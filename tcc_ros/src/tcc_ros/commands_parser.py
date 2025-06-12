#!/usr/bin/env python
"""
This script is based on the paper, "The Conversation is the Command: Interacting with Real-World Autonomous Robots Through Natural Language": https://dl.acm.org/doi/abs/10.1145/3610978.3640723
Its usage is subject to the  Creative Commons Attribution International 4.0 License.
"""
import re
import json
import rospy
from typing import Dict, List, Union, Optional
from std_msgs.msg import String
from sensor_msgs.msg import Image
import rospkg
import os

class CommandParser:
    def __init__(self, action_executor):
        self.action_executor = action_executor
        self.patterns = self._load_patterns()
        self.linear_speed = rospy.get_param("speeds/default_linear_speed", 0.2)
        self.angular_speed = rospy.get_param("speeds/default_angular_speed", 0.5)
        self.maximum_speed = rospy.get_param("speeds/maximum_speed", 1.0)
        self.minimum_speed = rospy.get_param("speeds/minimum_speed", 0.2)

    def _load_patterns(self) -> Dict[str, str]:
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('tcc_ros')
        full_path = os.path.join(package_path, 'config', 'action_dictionary.json')
        with open(full_path, "r") as file:
            data = json.load(file)
            return data["Action_Dictionary"]["patterns"]

    def parse_input(self, llm_output: Dict) -> Dict:
        if llm_output['type'] == 'RESPONSE':
            return {'type': 'RESPONSE', 'content': llm_output['content']}
        elif llm_output['type'] == 'ACTIONS':
            actions = self.parse_actions(llm_output['content'])
            return {'type': 'ACTIONS', 'content': actions}
        else:
            rospy.logwarn("Unrecognized llm_output type.")
            return {'type': 'UNKNOWN', 'content': llm_output['content']}
        
    def parse_actions(self, llm_output: str) -> List[Dict]:
        actions = []
        lines = re.split(r'\n|;', llm_output)
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if match := re.match(r'Action\s*\d+:\s*(.+)', line, re.IGNORECASE):
                action_description = match.group(1).strip()
                if parsed := self._parse_action(action_description):
                    actions.extend(parsed) if isinstance(parsed, list) else actions.append(parsed)
        return actions

    def _parse_movement(
        self,
        action_type: str,
        match: re.Match,
        default_value: float,
        is_angular: bool = False
    ) -> Dict:
        """Helper for parsing movement commands with optional parameters."""
        value = float(match.group(1)) if match.group(1) else default_value
        param = 'angle' if is_angular else 'distance'
        action = {'action': action_type, param: value}
        if speed_group := (match.group(3) if action_type in ('FORWARD', 'BACKWARD') else match.group(3)):
            action['speed'] = float(speed_group)
        return action

    def _parse_circular_motion(self, match: re.Match, description: str) -> Dict:
        radius = 1.0  # Default radius
        if match.group(1):
            radius = float(match.group(1))
        speed_input = match.group(2)
        speed_unit = match.group(3) or 'm/s'
        if not speed_input:
            speed = self.linear_speed
        elif speed_input.lower() in ('maximum', 'max'):
            speed = self.maximum_speed
        elif speed_input.lower() in ('minimum', 'min'):
            speed = self.minimum_speed
        else:
            try:
                speed = float(speed_input)
            except (TypeError, ValueError):
                speed = self.linear_speed

        # Angle determination
        angle = 360.0  # Default full circle
        desc_lower = description.lower() 
        if "half" in desc_lower or "semi" in desc_lower:
            angle = 180.0
        elif "arc" in desc_lower:
            if angle_match := re.search(r'arc\s*(?:of)?\s*(\d+)\s*degrees?', description, re.IGNORECASE):
                angle = float(angle_match.group(1))
            else:
                angle = 90.0
        elif angle_match := re.search(r'(\d+)\s*degrees?', description, re.IGNORECASE):
            angle = float(angle_match.group(1))
        return {
            'action': 'CIRCULAR_MOTION',
            'radius': radius,
            'speed': speed,
            'speed_unit': speed_unit,
            'angle': angle,
            'direction': 'clockwise'
        }

    def _parse_action(self, description: str) -> Optional[Union[Dict, List]]:
        desc_lower = description.lower().strip()

        # 1. Non-parametric, non-movement, and information retrival commands
        if re.fullmatch(self.patterns["send_image"], desc_lower):
            return {'action': 'SEND_IMAGE'}
        if re.fullmatch(r'describe surroundings[.,]?', desc_lower):
            return {'action': 'DESCRIBE_SURROUNDINGS'}
        if re.fullmatch(r'report coordinates[.,]?', desc_lower):
            return {'action': 'REPORT_COORDINATES'}
        if re.fullmatch(r'report object locations[.,]?', desc_lower):
            return {'action': 'REPORT_OBJECT_LOCATIONS'}
        if re.fullmatch(r'report orientation[.,]?', desc_lower):
            return {'action': 'REPORT_ORIENTATION'}
        if re.fullmatch(self.patterns["stop"], desc_lower):
            return {'action': 'STOP'}

        # 2. Parametric movement-based commands
        if match := re.fullmatch(self.patterns["move_forward"], desc_lower):
            return self._parse_movement('FORWARD', match, 1.0)
        if match := re.fullmatch(self.patterns["move_backward"], desc_lower):
            return self._parse_movement('BACKWARD', match, 1.0)
        if match := re.fullmatch(self.patterns["turn_left"], desc_lower):
            return self._parse_movement('TURN_LEFT', match, 90.0, True)
        if match := re.fullmatch(self.patterns["turn_right"], desc_lower):
            return self._parse_movement('TURN_RIGHT', match, 90.0, True)
        if match := re.fullmatch(self.patterns["rotate"], desc_lower):
            angle = float(match.group(1)) if match.group(1) else 360.0
            return {'action': 'ROTATE', 'angle': angle}

        # 3. Object-based navigation
        if match := re.fullmatch(self.patterns["rotate_to_face"], desc_lower):
            return {'action': 'ROTATE_TO_FACE', 'object_name': match.group(1).strip()}
        if match := re.fullmatch(self.patterns["move_to_object"], desc_lower):
            return {'action': 'MOVE_TO_OBJECT', 'object_name': match.group(1).strip()}

        # 4. Goal-directed navigation
        if match := re.fullmatch(self.patterns["navigate_to_destination"], desc_lower, re.IGNORECASE):
            action = {'action': 'NAVIGATE_TO_DESTINATION', 'destination_name': match.group(1).strip()}
            if speed := match.group(2):
                action['speed'] = float(speed)
            return action
        if match := re.fullmatch(self.patterns["navigate_around_object"] + '|' + 
                                self.patterns["navigate_around_generic"], desc_lower, re.IGNORECASE):
            clearance = float(match.group(2)) if match.group(2) else 0.5
            return {
                'action': 'NAVIGATE_AROUND_OBJECT',
                'object_name': match.group(1).strip(),
                'clearance': clearance
            }

        # 5. Coordinate-based naviagtion
        if match := re.search(self.patterns["go_to_coordinates"], desc_lower):
            x = float(match.group(1))
            y = float(match.group(3))
            z = float(match.group(5)) if match.group(5) else 0.0 
            speed = None
            if match.group(6):
                try:
                    speed = float(match.group(6))
                except (TypeError, ValueError):
                    pass
            action = {'action': 'GO_TO_COORDINATES', 'coordinates': {'x': x, 'y': y, 'z': z}}
            if speed:
                action['speed'] = speed
            return action

        # 6. Pattern motions (Circle)
        if match := re.fullmatch(self.patterns["circular_motion"], desc_lower):
            return self._parse_circular_motion(match, description)
        elif re.search(r'^(?:move|go|drive|circle)\s+(?:in\s+an?\s+)?(?:arc|circle|half\s*circle|semi\s*circle)', desc_lower, re.IGNORECASE):
            class DummyMatch:
                def group(self, num):
                    return None
            return self._parse_circular_motion(DummyMatch(), description)

        # 7. Time-based commands
        if match := re.fullmatch(self.patterns["wait"], desc_lower):
            duration = 0
            if duration_str := match.group(1):
                if time_match := re.match(r'(\d+(\.\d+)?)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?)', duration_str, re.IGNORECASE):
                    amount, unit = float(time_match.group(1)), time_match.group(3).lower()
                    duration = amount * {'sec': 1, 'min': 60, 'hour': 3600, 'hr': 3600}.get(unit[:3], 1)
            return {'action': 'WAIT', 'duration': duration}

        # 8. Fallback: Generic action patterns
        if generic_actions := re.findall(
            r'(move forward|move backward|turn left|turn right|rotate)\s*(\d+(\.\d+)?)?', 
            desc_lower
        ):
            parsed = []
            for verb, num, _ in generic_actions:
                action_map = {
                    'move forward': ('FORWARD', 1.0),
                    'move backward': ('BACKWARD', 1.0),
                    'turn left': ('TURN_LEFT', 90.0),
                    'turn right': ('TURN_RIGHT', 90.0),
                    'rotate': ('ROTATE', 360.0)
                }
                action_type, default = action_map[verb]
                value = float(num) if num else default
                if 'move' in verb:
                    parsed.append({'action': action_type, 'distance': value})
                else:
                    parsed.append({'action': action_type, 'angle': value})
            return parsed
        rospy.logwarn("Action not recognized: " + description)
        return None