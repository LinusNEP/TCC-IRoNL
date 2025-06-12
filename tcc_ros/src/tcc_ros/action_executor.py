#!/usr/bin/env python
"""
This script is based on the paper, "The Conversation is the Command: Interacting with Real-World Autonomous Robots Through Natural Language": https://dl.acm.org/doi/abs/10.1145/3610978.3640723
Its usage is subject to the  Creative Commons Attribution International 4.0 License.
"""
import rospy
import actionlib
from geometry_msgs.msg import Twist, Quaternion, PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry
import tf
import math
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
from dynamic_reconfigure.client import Client
from tcc_ros.destination_resolver import DestinationResolver
import rospkg, os

class ActionExecutor:
    def __init__(self, data_logger, perception_module, llm_interface):
        self.data_logger = data_logger
        self.perception_module = perception_module
        self.llm_interface = llm_interface
        self.dyn_client = Client("/move_base/DWAPlannerROS", timeout=5)
        cmd_vel_topic = rospy.get_param("topics/cmd_vel", "/cmd_vel")
        odom_topic    = rospy.get_param("topics/odom", "/odom")
        self.cmd_vel_publisher = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)
        rospy.Subscriber(odom_topic, Odometry, self.odom_callback)
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.response_publisher = None 
        self.tts_publisher = rospy.Publisher('/tts_text', String, queue_size=10)
        rospy.loginfo("Waiting for move_base action server...")
        self.move_base_client.wait_for_server()
        rospy.loginfo("Connected to move_base action server.")
        self.linear_speed  = rospy.get_param("speeds/default_linear_speed", 0.2)
        self.angular_speed = rospy.get_param("speeds/default_angular_speed", 0.5)
        self.max_speed = rospy.get_param("speeds/maximum_speed", 1.0)
        self.min_speed = rospy.get_param("speeds/minimum_speed", 0.2)
        self.current_position = None
        self.current_orientation = None
        self.image_publisher = rospy.Publisher('/llm_image_output', Image, queue_size=10)
        self.bridge = CvBridge()
        self.detected_objects = {}
        self.latest_detections = []
        pkg_path  = rospkg.RosPack().get_path('tcc_ros')
        yaml_path = os.path.join(pkg_path, "config/config.yaml")
        self.dest_resolver = DestinationResolver(yaml_path)
        self.perception_module = None
        self.emergency_stop = False
        self.emergency_stop_sub = rospy.Subscriber('/emergency_stop', String, self.handle_emergency_stop)

    def odom_callback(self, msg):
        self.current_position = msg.pose.pose.position
        self.current_orientation = msg.pose.pose.orientation

    def speak_and_respond(self, text):
        if not self.response_publisher:
            return
        self.response_publisher.publish(String(data=text))
        self.tts_publisher.publish(String(data=text))

    def update_detected_objects(self):
        raw_objects = self.perception_module.get_object_locations()
        def normalize_label(label):
            ignore_words = ["detected", "the", "a", "an", "object", "thing"]
            for word in ignore_words:
                label = label.replace(word, "")
            return label.strip().lower()
        self.detected_objects = {}
        for label, pose in raw_objects.items():
            normalized_label = normalize_label(label)
            self.detected_objects[normalized_label] = pose
        rospy.loginfo(f"[ActionExecutor] Updated detected objects: {list(self.detected_objects.keys())}")

    def execute_actions(self, actions):
        for idx, action in enumerate(actions, start=1):
            if action:
                rospy.loginfo(f"Executing Action {idx}: {action['action']}")
                self.execute_action(action)
                rospy.sleep(0.5)  

    def execute_action(self, action):
        action_type = action.get('action')
        parameters = action.get('parameters', {})
        for key, value in parameters.items():
            if isinstance(value, str):
                try:
                    parameters[key] = float(value)
                except ValueError:
                    pass
        try:
            if action_type == 'SEND_IMAGE':
                self.send_image()
            elif action_type == 'NAVIGATE_TO_DESTINATION':
                destination = action.get('destination_name')
                detected_objects = self.perception_module.get_detected_objects() if self.perception_module else []
                if destination.lower() in [obj.lower() for obj in detected_objects]:
                    rospy.loginfo(f"Interpreting destination '{destination}' as a detected object.")
                    self.execute_action({'action': 'MOVE_TO_OBJECT', 'object_name': destination})
                    return
                speed = action.get('speed', None)
                if destination:
                    rospy.loginfo(f"Navigating to destination: {destination}")
                    self.navigate_to(destination, speed=speed)
                else:
                    rospy.logwarn("No destination provided for NAVIGATE_TO_DESTINATION action.")
            elif action_type == 'GO_TO_COORDINATES':
                coordinates = action.get('coordinates')
                speed = action.get('speed', None)
                if coordinates:
                    self.send_navigation_goal(coordinates, speed=speed)
                else:
                    rospy.logwarn("Coordinates not provided for GO_TO_COORDINATES action.")
            elif action_type == 'FORWARD':
                self.move_forward(action.get('distance', 1.0), speed=action.get('speed', None))
                self.speak_and_respond(f"Moved forward {action.get('distance', 1.0)} meters.")
            elif action_type == 'BACKWARD':
                self.move_backward(action.get('distance', 1.0), speed=action.get('speed', None))
                self.speak_and_respond(f"Moved backward {action.get('distance', 1.0)} meters.")
            elif action_type == 'TURN_LEFT':
                self.turn('left', action.get('angle', 90.0), speed=action.get('speed', None))
                self.speak_and_respond(f"Turned left {action.get('angle', 90.0)} degrees.")
            elif action_type == 'TURN_RIGHT':
                self.turn('right', action.get('angle', 90.0), speed=action.get('speed', None))
                self.speak_and_respond(f"Turned right {action.get('angle', 90.0)} degrees.")
            elif action_type == 'ROTATE':
                rospy.loginfo(f"Executing rotate action with parameters: {action}")
                self.rotate(angle=action.get('angle', 360.0))
                self.speak_and_respond(f"Rotated {action.get('angle', 360.0)} degrees.")
            elif action_type == 'CIRCULAR_MOTION':
                rospy.loginfo(f"Executing circular motion with parameters: {action}")
                radius = action.get('radius', 1.0)
                speed = action.get('speed', self.linear_speed)
                speed_unit = action.get('speed_unit', 'm/s')
                angle = action.get('angle', 360.0)
                self.move_in_circle(radius=radius,
                                    speed=speed,
                                    speed_unit=speed_unit,
                                    angle=angle)
                if angle < 360.0:
                    self.speak_and_respond(
                        f"Moved in an arc of {angle} degrees with a radius of {radius} meters at {speed} {speed_unit}."
                    )
                else:
                    self.speak_and_respond(
                        f"Moved in a circle of radius {radius} meters at {speed} {speed_unit}."
                    )
            elif action_type == 'ROTATE_TO_FACE':
                obj_name = action.get('object_name')
                self.rotate_to_face(obj_name)
            elif action_type == 'DESCRIBE_SURROUNDINGS':
                self.describe_and_update_surroundings()
            elif action_type == 'REPORT_COORDINATES':
                self.report_coordinates()
            elif action_type == 'REPORT_OBJECT_LOCATIONS':
                self.report_object_locations()
            elif action_type == 'REPORT_ORIENTATION':
                self.report_orientation()
            elif action_type == 'MOVE_TO_OBJECT':
                object_name = action.get('object_name')
                if object_name.lower().startswith('detected '):
                    object_name = object_name[9:].strip()
                color = action.get('object_color', None)
                rospy.loginfo(f"Attempting to move towards object: {object_name} (Color: {color})")
                self.update_detected_objects()
                detected_objects = self.detected_objects.get(object_name, [])
                if color:
                    obj_dict = next((obj for obj in detected_objects
                                     if obj.get('color', '').lower() == color.lower()), None)
                else:
                    obj_dict = detected_objects[0] if detected_objects else None
                if obj_dict:
                    coords = {'x': obj_dict['x'], 'y': obj_dict['y'], 'z': obj_dict['z']}
                    rospy.loginfo(f"Moving towards {object_name} at {coords}")
                    self.send_navigation_goal_for_object(coords, object_name)
                else:
                    rospy.logwarn(f"Object '{object_name}' not found.")
                    self.speak_and_respond(f"Object '{object_name}' not found.")
            
            elif action_type == 'NAVIGATE_AROUND_OBJECT':
                object_name = action.get('object_name')
                clearance = action.get('clearance', 0.5)  # Default 0.5m clearance
                self.navigate_around_object(object_name, clearance)
                        
            elif action_type == 'WAIT':
                rospy.loginfo("Received WAIT action.")
                self.cancel_current_goal()
                duration = action.get('duration', 0)
                if duration > 0:
                    rospy.sleep(duration)
                self.speak_and_respond(f"Waited for {duration} seconds as requested.")

            elif action_type == 'STOP':
                rospy.loginfo("Received STOP command - halting all movement")
                self.cancel_current_goal()
                self.cmd_vel_publisher.publish(Twist())
                self.speak_and_respond("All movement has been stopped.")
                        
            elif action_type == 'UNABLE_TO_PERFORM':
                reason = action.get('reason', 'I cannot perform this action.')
                self.speak_and_respond(reason)
            else:
                rospy.logwarn(f"Unknown action: {action_type}")
                self.speak_and_respond(f"Unknown action: {action_type}")
        except Exception as e:
            rospy.logerr(f"Error executing action {action_type}: {e}")
            self.speak_and_respond(f"Failed to execute action: {action_type}")

    def send_image(self):
        try:
            self.speak_and_respond("Here is the image of my current surroundings. Please check the image display area.")
            self.perception_module.send_latest_image()
        except Exception as e:
            rospy.logerr(f"Failed to send image: {e}")
            self.speak_and_respond("An error occurred while attempting to send the image.")

    def send_navigation_goal_for_object(self, destination_label):
        try:
            def normalize_label(label):
                ignore_words = ["detected", "the", "a", "an", "object", "thing"]
                for word in ignore_words:
                    label = label.replace(word, "")
                return label.strip().lower()
            normalized_label = normalize_label(destination_label)
            if not self.detected_objects:
                rospy.logwarn("[ActionExecutor] No detected_objects cache — falling back to latest_detections.")
                self.detected_objects = {}
                for det in self.latest_detections:
                    label = normalize_label(det.get("label", ""))
                    pose = det.get("pose", None)
                    if label and pose:
                        self.detected_objects[label] = pose
            if not self.detected_objects:
                rospy.logwarn("[ActionExecutor] No detected objects available at all.")
                self.speak_and_respond("I have no objects to navigate to.")
                return
            rospy.loginfo(f"[ActionExecutor] Available objects: {list(self.detected_objects.keys())}")
            pose = self.detected_objects.get(normalized_label)
            if not pose:
                import difflib
                candidates = list(self.detected_objects.keys())
                best_match = difflib.get_close_matches(normalized_label, candidates, n=1)
                if best_match:
                    pose = self.detected_objects[best_match[0]]
                    rospy.logwarn(f"[ActionExecutor] Approximated destination '{destination_label}' to '{best_match[0]}'.")
                    normalized_label = best_match[0]
            if not pose:
                rospy.logwarn(f"[ActionExecutor] Destination '{destination_label}' not found after normalization.")
                self.speak_and_respond(f"I couldn't find any object called {destination_label}.")
                return
            rospy.loginfo(f"[ActionExecutor] Navigating toward: {normalized_label}")
            self.speak_and_respond(f"I'm heading toward the {normalized_label}.")
            if self.move_to_pose(pose):
                self.speak_and_respond(f"I have arrived at the {normalized_label}.")
            else:
                self.speak_and_respond(f"I couldn't reach the {normalized_label}.")
        except Exception as e:
            rospy.logerr(f"[ActionExecutor] Failed to send navigation goal: {str(e)}")
            self.speak_and_respond("An error occurred while trying to navigate to the object.")

    def navigate_around_object(self, object_name, clearance=0.5):
        try:
            obj_pose = self.perception_module.get_object_pose(object_name)
            if not obj_pose:
                raise ValueError(f"Object '{object_name}' not detected")
            obj_x = obj_pose.pose.position.x
            obj_y = obj_pose.pose.position.y 
            waypoints = [
                (obj_x + clearance, obj_y),
                (obj_x, obj_y + clearance),
                (obj_x - clearance, obj_y),
                (obj_x, obj_y - clearance)
            ]
            for wx, wy in waypoints:
                goal = MoveBaseGoal()
                goal.target_pose.header.frame_id = "map"
                goal.target_pose.header.stamp = rospy.Time.now()
                goal.target_pose.pose.position.x = wx
                goal.target_pose.pose.position.y = wy
                goal.target_pose.pose.orientation = Quaternion(*tf.transformations.quaternion_from_euler(0, 0, 0))
                self.move_base_client.send_goal(goal)
                self.move_base_client.wait_for_result()
            self.speak_and_respond(f"Completed navigation around {object_name}")
        except Exception as e:
            rospy.logerr(f"Circular navigation failed: {e}")
            self.speak_and_respond(f"Failed to navigate around {object_name}.")

    def describe_and_update_surroundings(self):
        try:
            detections = self.perception_module.detect_objects(
                prob_thresh=self.perception_module.detection_confidence_threshold,
                color_thresh=self.perception_module.detection_confidence_threshold
            )
            if not detections:
                rospy.logwarn("[Perception] No objects detected at normal thresholds. Retrying with relaxed thresholds...")
                detections = self.perception_module.detect_objects(prob_thresh=0.05, color_thresh=0.05)
            if not detections:
                self.speak_and_respond("No objects detected in my surroundings.")
                return
            self.latest_detections = detections
            response = []
            for idx, det in enumerate(detections):
                label = det.get('label', 'unknown')
                pos = det.get('pose', None)
                color = det.get('color', 'unknown')
                confidence = det.get('confidence', 0.0)
                if pos and hasattr(pos, 'pose'):
                    response.append(
                        f"{label} {idx+1}: at (x={pos.pose.position.x:.2f}, y={pos.pose.position.y:.2f}) with "
                        f"confidence: {confidence * 100:.0f}% ." #, color: {color}"
                    )
                else:
                    response.append(f"{label} {idx+1}: position unknown")
            if response:
                self.speak_and_respond("I can see:\n" + "\n".join(response))
            else:
                self.speak_and_respond("I looked around but could not confidently detect objects.")
        except Exception as e:
            rospy.logerr(f"Description failed: {str(e)}")
            self.speak_and_respond("An error occurred while describing the surroundings.")

    def report_object_locations(self):
        try:
            if not self.latest_detections:
                self.speak_and_respond("I have not detected any objects yet.")
                return
            response = "Here are the detected objects and their locations:\n"
            for idx, det in enumerate(self.latest_detections):
                label = det.get('label', 'unknown')
                pose = det.get('pose', None)
                if pose and hasattr(pose, 'pose'):
                    x = pose.pose.position.x
                    y = pose.pose.position.y
                    z = pose.pose.position.z
                    response += f"{label} {idx+1}: x={x:.2f}, y={y:.2f}, z={z:.2f}\n"
                else:
                    response += f"{label} {idx+1}: Location unknown\n"
            self.speak_and_respond(response)
        except Exception as e:
            rospy.logerr(f"[ActionExecutor] Failed to report object locations: {str(e)}")
            self.speak_and_respond("Something went wrong while retrieving object locations.")

    def move_forward(self, distance, speed=None):
        try:
            actual_speed = speed if speed else self.linear_speed
            self.speak_and_respond(f"Moving forward {distance} m at {actual_speed} m/s.")
            twist = Twist()
            twist.linear.x = actual_speed
            duration = abs(distance / actual_speed)
            rate = rospy.Rate(10)
            start = rospy.Time.now()
            while (rospy.Time.now() - start).to_sec() < duration and not self.emergency_stop:
                if self.emergency_stop:
                    self.cmd_vel_publisher.publish(Twist())
                    rospy.logwarn("Emergency stop triggered during move_forward")
                    break
                self.cmd_vel_publisher.publish(twist)
                rate.sleep()
            self.cmd_vel_publisher.publish(Twist())
            if self.emergency_stop:
                self.emergency_stop = False  # Reset after handling
        except Exception as e:
            rospy.logerr(f"move forward failed: {e}")
            raise e

    def move_backward(self, distance, speed=None):
        try:
            actual_speed = speed if speed else self.linear_speed
            self.speak_and_respond(f"Moving backward {distance} m at {actual_speed} m/s.")
            twist = Twist()
            twist.linear.x = -abs(actual_speed)
            duration = abs(distance / actual_speed)
            rate = rospy.Rate(10)
            start = rospy.Time.now()
            while (rospy.Time.now() - start).to_sec() < duration and not self.emergency_stop:
                if self.emergency_stop:
                    self.cmd_vel_publisher.publish(Twist())
                    rospy.logwarn("Emergency stop triggered during move_backward")
                    break
                self.cmd_vel_publisher.publish(twist)
                rate.sleep()
            self.cmd_vel_publisher.publish(Twist())
            if self.emergency_stop:
                self.emergency_stop = False
        except Exception as e:
            rospy.logerr(f"move backward failed: {e}")
            raise e

    def turn(self, direction, angle, speed=None):
        try:
            actual_speed_deg = speed if speed else math.degrees(self.angular_speed)
            self.speak_and_respond(f"Turning {direction} {angle}° at {actual_speed_deg} deg/s.")
            angular_speed = math.radians(actual_speed_deg)
            if direction.lower() == 'left':
                twist_z = angular_speed
            else:
                twist_z = -angular_speed
            twist = Twist()
            twist.angular.z = twist_z
            duration = math.radians(abs(angle)) / abs(angular_speed)
            rate = rospy.Rate(10)
            start = rospy.Time.now()
            while (rospy.Time.now() - start).to_sec() < duration and not self.emergency_stop:
                if self.emergency_stop:
                    self.cmd_vel_publisher.publish(Twist())
                    rospy.logwarn("Emergency stop triggered during turn")
                    break
                self.cmd_vel_publisher.publish(twist)
                rate.sleep()
            self.cmd_vel_publisher.publish(Twist())
            if self.emergency_stop:
                self.emergency_stop = False
        except Exception as e:
            rospy.logerr(f"turn {direction} failed: {e}")
            raise e
    
    def rotate(self, angle, speed=None, speed_unit='deg/s'):
        try:
            self.speak_and_respond(f"Rotating {angle} degrees.")
            if speed and speed_unit in ['deg/s', 'rad/s']:
                actual_speed = self.convert_speed(speed, speed_unit)
            else:
                actual_speed = self.angular_speed
            twist = Twist()
            direction = 1.0 if angle >= 0 else -1.0
            twist.angular.z = direction * abs(actual_speed)
            duration = math.radians(abs(angle)) / abs(actual_speed)
            rate = rospy.Rate(10)
            start = rospy.Time.now()           
            while (rospy.Time.now() - start).to_sec() < duration and not self.emergency_stop:
                if self.emergency_stop:
                    self.cmd_vel_publisher.publish(Twist())
                    rospy.logwarn("Emergency stop triggered during rotate")
                    break
                self.cmd_vel_publisher.publish(twist)
                rate.sleep()
            self.cmd_vel_publisher.publish(Twist())
            if self.emergency_stop:
                self.emergency_stop = False
        except Exception as e:
            rospy.logerr(f"Failed to rotate: {e}")
            raise e

    def move_in_circle(self, radius, speed, speed_unit='m/s', angle=360.0, clockwise=False):
        try:
            actual_speed = self.convert_speed(speed, speed_unit)
            twist = Twist()
            twist.linear.x = abs(actual_speed)
            twist.angular.z = abs(actual_speed) / radius
            if clockwise:
                twist.angular.z = -twist.angular.z
            circumference_fraction = (abs(angle) / 360.0) * (2 * math.pi * radius)
            duration = abs(circumference_fraction / actual_speed)
            self.speak_and_respond(f"Moving in circle of radius={radius}, speed={speed} {speed_unit}, angle={angle}.")
            rate = rospy.Rate(10)
            start = rospy.Time.now()
            while (rospy.Time.now() - start).to_sec() < duration and not self.emergency_stop:
                if self.emergency_stop:
                    self.cmd_vel_publisher.publish(Twist())
                    rospy.logwarn("Emergency stop triggered during circular motion")
                    break
                self.cmd_vel_publisher.publish(twist)
                rate.sleep()   
            self.cmd_vel_publisher.publish(Twist())
            if self.emergency_stop:
                self.emergency_stop = False
        except Exception as e:
            rospy.logerr(f"Failed to move in circle: {e}")
            raise e

    def convert_speed(self, value, unit):
        """ Converts a numeric speed in 'deg/s', 'rad/s', etc. to m/s or rad/s. """
        if unit == 'deg/s':
            return math.radians(value)
        elif unit == 'rad/s':
            return value
        elif unit == 'm/s':
            return value
        elif unit == 'km/h':
            return value / 3.6
        elif unit == 'mph':
            return value * 0.44704
        elif unit == 'ft/s':
            return value * 0.3048
        else:
            raise ValueError(f"Unsupported speed unit: {unit}")

    def rotate_to_face(self, object_name):
        try:
            self.speak_and_respond(f"Rotating to face {object_name}.")
            if self.perception_module:
                angle_to_object = self.perception_module.get_angle_to_object(object_name)
                if angle_to_object is not None:
                    self.rotate(angle_to_object)
                    self.speak_and_respond(f"Rotated to face {object_name}.")
                else:
                    msg = f"Could not locate object '{object_name}' to rotate towards."
                    rospy.logwarn(msg)
                    if self.response_publisher:
                        self.speak_and_respond(msg)
            else:
                msg = "Perception module is not available."
                rospy.logwarn(msg)
                if self.response_publisher:
                    self.speak_and_respond(msg)
        except Exception as e:
            rospy.logerr(f"Failed to rotate to face {object_name}: {e}")
            raise e

    def navigate_to(self, destination_name, speed=None):
        try:
            slug, how = self.dest_resolver.resolve(destination_name)
            if slug is None:
                msg = f"Destination '{destination_name}' not found."
                rospy.logwarn(msg)
                self.speak_and_respond(msg)
                return
            goal_coords = self.dest_resolver.coords[slug]
            rospy.loginfo(f"[dest‑resolve] '{destination_name}' → '{slug}' via {how}  → {goal_coords}")
            if speed:
                self.adjust_navigation_speed(speed)
            self.send_navigation_goal(
                goal_coords,
                destination_name=destination_name,
                speed=speed
            )
            if speed:
                self.reset_navigation_speed()
        except Exception as e:
            rospy.logerr(f"Failed to navigate to '{destination_name}': {e}")
            self.speak_and_respond(f"Navigation to {destination_name} failed.")
            raise

    def adjust_navigation_speed(self, speed):
        try:
            rospy.loginfo(f"Adjusting nav speed to {speed} m/s.")
            params = {
                'max_vel_x': speed,
                'min_vel_x': -speed,
                'max_vel_trans': speed,
                'min_vel_trans': 0.2,
            }
            self.dyn_client.update_configuration(params)
        except Exception as e:
            rospy.logerr(f"Failed to adjust navigation speed: {e}")

    def reset_navigation_speed(self):
        try:
            rospy.loginfo("Resetting nav speed to default.")
            default_speed = self.linear_speed
            params = {
                'max_vel_x': default_speed,
                'min_vel_x': -default_speed,
                'max_vel_trans': default_speed,
                'min_vel_trans': 0.2,
            }
            self.dyn_client.update_configuration(params)
        except Exception as e:
            rospy.logerr(f"Failed to reset navigation speed: {e}")

    def send_navigation_goal(self, coordinates, destination_name=None, speed=None):
        try:
            x, y, z = coordinates['x'], coordinates['y'], coordinates.get('z', 0.0)
            self.speak_and_respond(f"Navigating to the {destination_name}."if destination_name else f"Sending navigation goal to the coordinates x={x}, y={y}, z={z}")
            if speed:
                self.adjust_navigation_speed(speed)
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "map"
            goal.target_pose.header.stamp = rospy.Time.now()
            goal.target_pose.pose.position.x = x
            goal.target_pose.pose.position.y = y
            goal.target_pose.pose.position.z = z
            quaternion = tf.transformations.quaternion_from_euler(0, 0, 0)
            goal.target_pose.pose.orientation = Quaternion(*quaternion)
            self.move_base_client.send_goal(goal)
            rate = rospy.Rate(10)
            while not rospy.is_shutdown():
                if self.emergency_stop:
                    rospy.logwarn("Emergency stop detected, cancelling goal immediately.")
                    self.move_base_client.cancel_all_goals()
                    self.cmd_vel_publisher.publish(Twist())
                    break
                state = self.move_base_client.get_state()
                if state in [actionlib.GoalStatus.SUCCEEDED,
                            actionlib.GoalStatus.ABORTED,
                            actionlib.GoalStatus.REJECTED,
                            actionlib.GoalStatus.PREEMPTED]:
                    break
                rate.sleep()
            if speed:
                self.reset_navigation_speed()
            final_state = self.move_base_client.get_state()
            if final_state == actionlib.GoalStatus.SUCCEEDED:
                rospy.loginfo("Navigation success.")
                self.speak_and_respond(f"Arrived at {destination_name} successfully." if destination_name else "Navigation to the given coordinates was successful.")     
            elif final_state == actionlib.GoalStatus.PREEMPTED:
                rospy.loginfo("Navigation preempted (interrupted).")
                self.speak_and_respond("Navigation was interrupted.")
            else:
                rospy.logwarn("Navigation failed.")
                self.speak_and_respond(f"Navigation to '{destination_name}' failed." if destination_name else "Navigation to the given coordinates failed.")
        except Exception as e:
            rospy.logerr(f"Failed to send navigation goal: {e}")
            raise e

    def report_coordinates(self):
        try:
            if self.current_position and self.current_orientation:
                x_rounded = round(self.current_position.x, 2)
                y_rounded = round(self.current_position.y, 2)
                z_rounded = round(self.current_position.z, 2)
                msg = (f"My current coordinates are x: {x_rounded}, "
                    f"y: {y_rounded}, z: {z_rounded}.")
            else:
                msg = "Current position data is unavailable."
            self.speak_and_respond(msg)
        except Exception as e:
            rospy.logerr("Failed to report coordinates.")
            self.speak_and_respond("Failed to report coordinates.")
            raise e

    def report_orientation(self):
        try:
            if self.current_orientation:
                q = self.current_orientation
                euler = tf.transformations.euler_from_quaternion((q.x, q.y, q.z, q.w))
                yaw_deg = math.degrees(euler[2])
                direction = self.get_cardinal_direction(yaw_deg)
                coords_info = self.get_coordinates()
                response = (f"My current orientation is {yaw_deg:.2f} degrees, "
                            f"facing {direction}. {coords_info}")
            else:
                response = "Orientation data is unavailable."
            self.speak_and_respond(response)
        except Exception as e:
            rospy.logerr("Failed to report orientation.")
            self.speak_and_respond("Failed to report orientation.")
            raise e

    def get_cardinal_direction(self, yaw_degrees):
        """Return approximate cardinal direction given yaw in degrees."""
        if -22.5 <= yaw_degrees < 22.5:
            return "north"
        elif 22.5 <= yaw_degrees < 67.5:
            return "northeast"
        elif 67.5 <= yaw_degrees < 112.5:
            return "east"
        elif 112.5 <= yaw_degrees < 157.5:
            return "southeast"
        elif 157.5 <= yaw_degrees or yaw_degrees < -157.5:
            return "south"
        elif -157.5 <= yaw_degrees < -112.5:
            return "southwest"
        elif -112.5 <= yaw_degrees < -67.5:
            return "west"
        elif -67.5 <= yaw_degrees < -22.5:
            return "northwest"
        else:
            return "unknown"

    def get_coordinates(self):
        if self.current_position:
            return (f"My current coordinates are x: {self.current_position.x:.2f}, "
                    f"y: {self.current_position.y:.2f}, z: {self.current_position.z:.2f}.")
        else:
            return "Current position data is unavailable."

    def get_destinations(self):
        mapping = {}
        for slug, info in self.dest_resolver.raw.items(): 
            display = info.get("display_name", slug.replace('_', ' ').title())
            mapping[display] = slug
        return mapping

    def cancel_current_goal(self):
        rospy.loginfo("Cancelling current navigation goal and stopping movement.")
        try:
            if self.move_base_client.get_state() in [actionlib.GoalStatus.ACTIVE, actionlib.GoalStatus.PENDING]:
                self.move_base_client.cancel_all_goals()
            self.cmd_vel_publisher.publish(Twist())
            self.reset_navigation_speed()
        except Exception as e:
            rospy.logerr(f"Error during goal cancellation: {e}")

    def handle_emergency_stop(self, msg):
        """Handle emergency stop commands from any source."""
        if msg.data.lower() in ['stop', 'emergency', 'halt']:
            self.emergency_stop = True
            self.cancel_current_goal()
            rospy.logwarn("EMERGENCY STOP ACTIVATED!")
