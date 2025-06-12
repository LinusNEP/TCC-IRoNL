"""
This script is based on the paper, "The Conversation is the Command: Interacting with Real-World Autonomous Robots Through Natural Language": https://dl.acm.org/doi/abs/10.1145/3610978.3640723
Its usage is subject to the  Creative Commons Attribution International 4.0 License.
"""
import rospy
import rospkg
import os
import torch
import torchvision
import tf
import time
import math
import cv2
import numpy as np
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, PointStamped
from cv_bridge import CvBridge
from filterpy.kalman import KalmanFilter
from scipy.spatial import ConvexHull
from scipy import ndimage
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from transformers import CLIPProcessor, CLIPModel
import message_filters
from sensor_msgs import point_cloud2
from std_msgs.msg import Header

class ImageManager:
    def __init__(self):
        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None
        self.last_image_time = None
    def image_callback(self, rgb_msg, depth_msg):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
            self.last_image_time = rospy.Time.now()
        except Exception as e:
            rospy.logerr(f"[ImageManager] Failed to convert images: {e}")

class DepthEstimator:
    def __init__(self, fx, fy, cx, cy, mono_model, mono_transform, device):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.mono_model = mono_model
        self.mono_transform = mono_transform
        self.device = device
    def pixel_to_3d_point(self, u, v, depth):
        """
        Projects a 2D pixel (u, v) and depth value into 3D camera space.
        Assumes optical frame: +X right, +Y down, +Z forward.
        """
        if depth <= 0.0 or not np.isfinite(depth):
            return None  # invalid depth
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth
        #rospy.loginfo(f"[Projection] (u,v)=({u},{v}), depth={depth:.2f} → x={x:.2f}, y={y:.2f}, z={z:.2f}")
        return np.array([x, y, z])
    def get_valid_depth_value(self, depth_image, x, y, radius=5):
        h, w = depth_image.shape
        xs = slice(max(0, x-radius), min(w, x+radius+1))
        ys = slice(max(0, y-radius), min(h, y+radius+1))
        region = depth_image[ys, xs]
        vals = region[np.isfinite(region) & (region > 0)]
        return float(np.median(vals)) if vals.size else None
    def monocular_depth_estimate(self, rgb_image, mask):
        h, w = mask.shape
        img = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        input_batch = self.mono_transform(img).to(self.device)
        with torch.no_grad():
            prediction = self.mono_model(input_batch)
        depth_map = prediction.squeeze().cpu().numpy()
        depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
        depth_map = depth_map / np.max(depth_map)
        depth_map *= 4.0  # max depth = 4 meters
        return float(np.median(depth_map[mask]))

class ObjectTracker:
    def __init__(self, initial_pose, process_noise=1e-4):
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        self.kf.F = np.eye(6)
        self.kf.F[:3, 3:] = np.eye(3)
        self.kf.H = np.eye(3, 6)
        self.kf.R = np.eye(3) * 0.1
        self.kf.P[3:, 3:] *= 1000.0
        self.kf.Q[3:, 3:] = process_noise
        self.kf.x[:3] = initial_pose
    def update(self, measurement):
        self.kf.predict()
        self.kf.update(measurement)

class PerceptionModule:
    def __init__(self, data_logger=None):
        self.data_logger = data_logger
        self.listener = tf.TransformListener()
        self.image_manager = ImageManager()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        sam_checkpoint = rospy.get_param("models/sam_checkpoint", "")
        self.sam_model = sam_model_registry["vit_b"](checkpoint=sam_checkpoint).to(self.device)
        self.sam_predictor = SamPredictor(self.sam_model)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam_model)
        self.mono_model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small').to(self.device)
        self.mono_transform = torch.hub.load('intel-isl/MiDaS', 'transforms').small_transform
        self.depth_estimator = DepthEstimator(
            fx=rospy.get_param("perception/camera_fx", 268),
            fy=rospy.get_param("perception/camera_fy", 268),
            cx=rospy.get_param("perception/camera_cx", 464),
            cy=rospy.get_param("perception/camera_cy", 400),
            mono_model=self.mono_model,
            mono_transform=self.mono_transform,
            device=self.device
        )
        self.base_frame = rospy.get_param("perception/base_frame", "camera_face")
        self.camera_frame = rospy.get_param("perception/camera_frame", "camera_optical_face")
        self.object_labels = rospy.get_param("perception/default_class_labels", [])
        self.color_labels = rospy.get_param("perception/color_labels", [])
        self.tracked_objects = {}
        self.next_track_id = 0
        self.image_publisher = rospy.Publisher("/llm_image_output", Image, queue_size=10)
        self.setup_subscribers()
        self.energy_threshold = rospy.get_param("/perception/energy_threshold", 0.35)
        self.mask_quality_thresh = rospy.get_param("/perception/mask_quality", 0.5)
        self.detection_confidence_threshold = rospy.get_param("/perception/detection_confidence_threshold", 0.2)

    def setup_subscribers(self):
        rgb_topic = rospy.get_param("topics/camera_color", "/camera_face/color/image_raw")
        depth_topic = rospy.get_param("topics/camera_depth", "/camera_face/depth/image_raw")
        rgb_sub = message_filters.Subscriber(rgb_topic, Image)
        depth_sub = message_filters.Subscriber(depth_topic, Image)
        ats = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=10, slop=0.1)
        ats.registerCallback(self.image_manager.image_callback)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)

    def odom_callback(self, msg):
        self.robot_pose = msg.pose.pose

    def detect_objects(self, prob_thresh=0.2, color_thresh=0.15):
        rospy.loginfo("[Perception] Starting object detection...")
        if self.image_manager.rgb_image is None:
            rospy.logwarn("[Perception] No RGB image available for detection.")
            return []
        if self.image_manager.depth_image is None:
            rospy.logwarn("[Perception] No depth image available yet.")
            return None
        if rospy.Time.now() - self.image_manager.last_image_time > rospy.Duration(2.0):
            rospy.logwarn("[Perception] Depth image is stale. Falling back to monocular.")
            return None
        detections = []
        valid_masks = 0
        rejected = {'quality': 0, 'energy': 0, 'confidence': 0, 'pose': 0}
        masks = self.generate_robust_masks()
        if not isinstance(masks, list):
            rospy.logerr("[Perception] Mask generation returned invalid type.")
            return detections
        for mask in masks:
            valid_masks += 1
            if not isinstance(mask, np.ndarray) or mask.dtype != bool:
                rospy.logwarn("[Perception] Invalid mask format encountered.")
                rejected['quality'] += 1
                continue
            try:
                obj_label, obj_conf, color_label, color_conf, energy = self.classify_with_uncertainty(mask)
                rospy.loginfo(f"[Perception] Object candidate: {obj_label} ({obj_conf:.2f}), "
                            f"Color: {color_label} ({color_conf:.2f}), Energy: {energy:.2f}")
                if energy > self.energy_threshold:
                    rospy.loginfo(f"[Perception] Rejecting due to energy threshold: {energy:.2f}")
                    rejected['energy'] += 1
                    continue
                if obj_conf < prob_thresh or color_conf < color_thresh:
                    rospy.loginfo(f"[Perception] Rejecting due to low confidence: obj_conf={obj_conf:.2f}, color_conf={color_conf:.2f}")
                    rejected['confidence'] += 1
                    continue
                centroid = self.mask_centroid(mask)
                depth_val = self.get_mask_depth(mask, centroid)
                if depth_val is None:
                    rospy.logwarn("[Perception] Depth estimation failed for mask.")
                    rejected['pose'] += 1
                    continue
                pose = self.robust_transform(centroid, depth_val)
                if not pose:
                    rospy.logwarn("[Perception] Pose transform failed for mask.")
                    rejected['pose'] += 1
                    continue
                detections.append({
                    "label": str(obj_label),
                    "confidence": float(obj_conf),
                    "color": str(color_label),
                    "color_confidence": float(color_conf),
                    "pose": pose,
                    "mask": mask.copy() 
                })
                if rospy.get_param("~debug_masks", False):
                    self.save_debug_mask(mask, obj_label)
            except Exception as e:
                rospy.logerr(f"[Perception] Failed to process mask: {str(e)}")
                rejected['quality'] += 1
                continue
        rospy.loginfo(f"[Perception] Mask statistics: Total={valid_masks}, "
                    f"Accepted={len(detections)}, "
                    f"Rejected: quality={rejected['quality']}, energy={rejected['energy']}, "
                    f"confidence={rejected['confidence']}, pose={rejected['pose']}")
        return detections
    
    def generate_masks(self, rgb_image):
        """Generate masks using SAM."""
        try:
            self.sam_predictor.set_image(rgb_image)
            output = self.sam_predictor.predict(
                point_coords=None, point_labels=None, multimask_output=True
            )
            masks = output['masks'] if isinstance(output, dict) else output[0]
            return [mask for mask in masks if self.compute_mask_quality(mask) > 0.5]
        except Exception as e:
            rospy.logerr(f"Mask generation failed: {e}")
            return []
        
    def classify_mask(self, rgb_image, mask):
        """Classify object and color labels using CLIP."""
        masked_img = rgb_image.copy()
        masked_img[~mask] = 0
        inputs = self.clip_processor(images=masked_img, text=self.object_labels + self.color_labels, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.clip_model(**inputs)
            probs = out.logits_per_image.softmax(dim=1)
        idx = probs.argmax(dim=1).item()

        num_obj = len(self.object_labels)
        if idx < num_obj:
            label = self.object_labels[idx]
            label_conf = probs[0, idx].item()
            color = "unknown"
            color_conf = 0.0
        else:
            label = "unknown"
            label_conf = 0.0
            color = self.color_labels[idx - num_obj]
            color_conf = probs[0, idx].item()
        energy = -torch.logsumexp(out.logits_per_image, dim=1).item()
        return label, label_conf, color, color_conf, energy
    
    def mask_centroid(self, mask):
        ys, xs = np.where(mask)
        if not len(xs):
            return (0, 0)
        return (int(xs.mean()), int(ys.mean()))
    
    def compute_mask_quality(self, mask):
        h, w = mask.shape
        if np.sum(mask) < 100:
            return 0.0
        y, x = np.where(mask)
        try:
            hull = ConvexHull(np.stack((x, y), axis=1))
            return np.sum(mask) / hull.volume
        except:
            return 0.5
        
    def pixel_to_base_frame(self, centroid, depth):
        try:
            camera_point = self.depth_estimator.pixel_to_3d_point(centroid[0], centroid[1], depth)
            now = rospy.Time.now()
            self.listener.waitForTransform(self.base_frame, self.camera_frame, now, rospy.Duration(1.0))
            pt = PointStamped()
            pt.header.stamp = now
            pt.header.frame_id = self.camera_frame
            pt.point.x, pt.point.y, pt.point.z = camera_point
            base_point = self.listener.transformPoint(self.base_frame, pt)
            return base_point.point
        except Exception as e:
            rospy.logerr(f"TF transform failed: {e}")
            if self.data_logger:
                self.data_logger.log(f"TF transform failed: {e}")
            return None
        
    def send_latest_image(self):
        if self.image_manager.rgb_image is not None:
            img_msg = self.image_manager.bridge.cv2_to_imgmsg(self.image_manager.rgb_image, "bgr8")
            self.image_publisher.publish(img_msg)
        else:
            rospy.logwarn("No RGB image available to publish.")

    def get_object_locations(self):
        detections = self.detect_objects()
        locs = {}
        for det in detections:
            label = det["label"].lower()
            pos = det["position"]
            entry = {
                "x": pos.x,
                "y": pos.y,
                "z": pos.z,
                "confidence": det["confidence"],
                "color": det["color"]
            }
            locs.setdefault(label, []).append(entry)
        return locs
    
    def get_detected_objects(self):
        detections = self.detect_objects()
        return [det["label"] for det in detections]
    
    def get_object_pose(self, object_name, prob_thresh=0.2):
        detections = self.detect_objects()
        for det in detections:
            if det["label"].lower() == object_name.lower() and det["confidence"] >= prob_thresh:
                pose = PoseStamped()
                pose.header.stamp = rospy.Time.now()
                pose.header.frame_id = self.base_frame
                pose.pose.position = det["position"]
                pose.pose.orientation.w = 1.0  # No rotation
                return pose
        rospy.logwarn(f"Object '{object_name}' not found with sufficient confidence.")
        return None
    
    def get_angle_to_object(self, object_name):
        pose = self.get_object_pose(object_name)
        if pose is None:
            return None
        if not hasattr(self, 'robot_pose') or self.robot_pose is None:
            rospy.logwarn("Robot pose not available yet.")
            return None
        obj_x = pose.pose.position.x
        obj_y = pose.pose.position.y
        q = self.robot_pose.orientation
        _, _, robot_yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        robot_yaw_deg = math.degrees(robot_yaw)
        angle_rad = math.atan2(obj_y, obj_x)
        angle_deg = math.degrees(angle_rad)
        relative_angle = angle_deg - robot_yaw_deg
        return relative_angle

    def generate_robust_masks(self):
        """Robust mask generation with fallback."""
        try:
            rgb_image = self.image_manager.rgb_image
            if rgb_image is None:
                rospy.logwarn("[Perception] No RGB image for mask generation.")
                return []
            height, width = rgb_image.shape[:2]
            if height < 100 or width < 100:
                rospy.logwarn(f"[Perception] Image size too small: {width}x{height}")
                return []
            image_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            valid_masks = []
            # Point-prompted segmentation
            try:
                self.sam_predictor.set_image(image_rgb)
                grid_size = 7 if min(height, width) > 500 else 5
                prompt_points = [
                    (int(width * i / (grid_size+1)), int(height * j / (grid_size+1)))
                    for i, j in np.ndindex(grid_size+1, grid_size+1)
                ][1:-1]
                prompt_points.append((width//2, height//2))
                negative_points = [(0, 0), (width, height)]
                all_pts = prompt_points + negative_points
                labels_pt = [1]*len(prompt_points) + [0]*len(negative_points)
                output = self.sam_predictor.predict(
                    point_coords=np.array(all_pts),
                    point_labels=np.array(labels_pt),
                    multimask_output=True
                )
                if isinstance(output, tuple):
                    masks, scores, _ = output
                elif isinstance(output, dict):
                    masks = output["masks"]
                    scores = output["iou_predictions"]
                else:
                    raise ValueError(f"Unexpected SAM output type: {type(output)}")
                for mask, score in zip(masks, scores):
                    if score < 0.4:
                        continue
                    if mask.shape != (height, width):
                        mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
                    if self.compute_mask_quality(mask) > self.mask_quality_thresh:
                        valid_masks.append(mask)
            except Exception as e:
                rospy.logerr(f"[Perception] Prompt-based segmentation failed: {str(e)}")
                valid_masks = []
            if not valid_masks:
                rospy.logwarn("[Perception] Using fallback AutomaticMaskGenerator.")
                try:
                    self.mask_generator = SamAutomaticMaskGenerator(self.sam_model)
                    full_masks = self.mask_generator.generate(image_rgb)
                    for m in full_masks:
                        mask = m["segmentation"]
                        if mask.shape != (height, width):
                            mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
                        if self.compute_mask_quality(mask) > self.mask_quality_thresh:
                            valid_masks.append(mask)
                except Exception as e:
                    rospy.logerr(f"[Perception] Automatic mask generation failed: {str(e)}")
            if rospy.get_param("~debug_masks", False) and valid_masks:
                self.visualize_masks(rgb_image, valid_masks)
            return valid_masks
        except Exception as e:
            rospy.logerr(f"[Perception] generate_robust_masks failed: {str(e)}")
            return []
        
    def classify_with_uncertainty(self, mask):
        """Classify an object and its color inside a mask using CLIP, and estimate uncertainty."""
        try:
            if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.sum() == 0:
                rospy.logwarn("[Perception] Empty or invalid mask passed to classify_with_uncertainty.")
                return ("unknown", 0.0, "unknown", 0.0, 1.0) 
            rgb_image = self.image_manager.rgb_image
            if rgb_image is None:
                rospy.logwarn("[Perception] No RGB image available during classification.")
                return ("unknown", 0.0, "unknown", 0.0, 1.0)
            clip_size = (224, 224)
            resized_rgb = cv2.resize(rgb_image, clip_size, interpolation=cv2.INTER_LINEAR)
            resized_mask = cv2.resize(mask.astype(np.uint8), clip_size, interpolation=cv2.INTER_NEAREST).astype(bool)
            masked_rgb = resized_rgb.copy()
            masked_rgb[~resized_mask] = 0
            inputs = self.clip_processor(
                images=masked_rgb,
                text=self.object_labels + self.color_labels,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits = outputs.logits_per_image 
                probs = logits.softmax(dim=1)
            energy = -torch.logsumexp(logits, dim=1).item()
            num_obj_labels = len(self.object_labels)
            obj_probs = probs[:, :num_obj_labels]
            color_probs = probs[:, num_obj_labels:]
            obj_conf, obj_idx = obj_probs.max(dim=1)
            color_conf, color_idx = color_probs.max(dim=1)
            obj_label = self.object_labels[obj_idx.item()] if num_obj_labels > 0 else "unknown"
            color_label = self.color_labels[color_idx.item()] if len(self.color_labels) > 0 else "unknown"
            return (obj_label, obj_conf.item(), color_label, color_conf.item(), energy)
        except Exception as e:
            rospy.logerr(f"[Perception] classify_with_uncertainty failed: {str(e)}")
            return ("unknown", 0.0, "unknown", 0.0, 1.0)

    def get_mask_depth(self, mask, centroid, fallback_to_mono=True):
        """Estimate depth at the centroid of a mask cleanly."""
        try:
            depth_image = self.image_manager.depth_image
            rgb_image = self.image_manager.rgb_image
            if depth_image is None:
                rospy.logwarn("[Perception] No depth image available for depth estimation.")
                return None
            x, y = int(centroid[0]), int(centroid[1])
            depth_val = self.depth_estimator.get_valid_depth_value(depth_image, x, y)
            if depth_val is not None and 0.1 < depth_val < 5.0:
                return depth_val
            if fallback_to_mono and rgb_image is not None and mask is not None:
                rospy.logwarn("[Perception] Falling back to monocular depth estimation.")
                return self.depth_estimator.monocular_depth_estimate(rgb_image, mask)
        except Exception as e:
            rospy.logerr(f"[Perception] get_mask_depth failed: {str(e)}")
        return None
    
    def robust_transform(self, centroid, depth_val, retries=3, wait_time=0.2):
        """Robustly transform a pixel centroid with depth into the world frame."""
        try:
            for attempt in range(retries):
                try:
                    camera_point = self.depth_estimator.pixel_to_3d_point(centroid[0], centroid[1], depth_val)
                    now = rospy.Time.now()
                    self.listener.waitForTransform(self.base_frame, self.camera_frame, now, rospy.Duration(1.0))
                    pt = PointStamped()
                    pt.header.stamp = now
                    pt.header.frame_id = self.camera_frame
                    pt.point.x, pt.point.y, pt.point.z = camera_point
                    base_point = self.listener.transformPoint(self.base_frame, pt)
                    pose = PoseStamped()
                    pose.header.stamp = now
                    pose.header.frame_id = self.base_frame
                    pose.pose.position = base_point.point
                    pose.pose.orientation.w = 1.0  
                    return pose
                except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as tf_err:
                    rospy.logwarn(f"[Perception] TF transform attempt {attempt+1}/{retries} failed: {tf_err}")
                    rospy.sleep(wait_time)
        except Exception as e:
            rospy.logerr(f"[Perception] robust_transform failed: {str(e)}")
        return None 
