"""
This script is based on the paper, "The Conversation is the Command: Interacting with Real-World Autonomous Robots Through Natural Language": https://dl.acm.org/doi/abs/10.1145/3610978.3640723
Its usage is subject to the  Creative Commons Attribution International 4.0 License.
"""
#!/usr/bin/env python
import rospy
import openai
from typing import Dict, Any
from tcc_ros.llm_provider import LLMProviderFactory

class LLMInterface:
    def __init__(self, api_key: str, destinations: Dict[str, Any]):
        openai.api_key = api_key
        self.LLM_PROVIDER = rospy.get_param("models/llm_provider", "openai")
        self.MODEL_NAME = rospy.get_param("models/llm_name", "gpt-4")
        self.MAX_TOKENS = rospy.get_param("models/llm_max_tokens", 500)
        self.TEMPERATURE = rospy.get_param("models/llm_temperature", 0)
        self.llm_endpoint = rospy.get_param("models/llm_endpoint", "")  # Optional
        self.llm_timeout = rospy.get_param("models/llm_timeout_seconds", 30)
        self.destinations = destinations
        self.llm_client = LLMProviderFactory.create_provider(
            self.LLM_PROVIDER,
            api_key,
            self.MODEL_NAME,
            self.MAX_TOKENS,
            self.TEMPERATURE
        )

    def call_llm(self, system_message: str, prompt: str) -> str:
        if self.LLM_PROVIDER == "llama.cpp":
            wrapped_prompt = f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\nInput: {prompt}\nAnswer:[/INST]"
            return self.llm_client.chat("", wrapped_prompt)
        else:
            return self.llm_client.chat(system_message, prompt)
        
    def get_system_message_status(self, current_yaw: str, cardinal_direction: str, 
                                  position_x: str, position_y: str, position_z: str) -> str:
        return (
            "You are TCC-Robo, a physical ROS based mobile robot. You are equipped with sensors and actuators. Your maximum and minimum movement speed are 1.0 m/s and 0.2 m/s respectively, "
            "and your maximum and minimum rotation speed are 90 deg/s and 0 deg/s respectively."
            "Answer queries about you or your capabilities.\n"
            "You have access to the following information about the robot's state: "
            f"Current orientation (yaw): {current_yaw} degrees, facing {cardinal_direction}. "
            f"Current position: x = {position_x}, y = {position_y}, z = {position_z}. "
            "Answer any user queries related to the robot's current status. "
            "When given a command, you should:\n"
            "1. Interpret the user's intent.\n"
            "2. If it's a command, generate a list of actions in natural language.\n"
            "3. If it's a query about your capabilities or status, provide a concise answer.\n"
            "Remember to include any speed or radius parameters specified by the user in your responses."
        )

    def get_system_message_commands(self) -> str:
        return (
            "Your task is to interpret the user's command and convert it into one of the following actions: "
            "Move forward, move backward, turn left, turn right, rotate, move in a circle, move in an arc, navigate to specific coordinates, "
            "describe surroundings, send image, report coordinates, report object locations, move to object, wait for a specified period, stop all movement, move in any given geometric pattern, etc."
        )

    def get_system_message_navigation(self) -> str:
        destinations_list = ', '.join(self.destinations.keys())
        return (
            "You can navigate to specific coordinates, to named destinations from the following list: "
            f"{destinations_list}, or to objects detected in your surroundings. Note that some named destinations like toilet or restroom, passage or corridor, lift or elevator are the same. Always note that your start location or pose is (0,0,0).\n"
            "### Fixed Destinations:\n"
            "If the user requests a fixed location (e.g., 'Navigate to the kitchen'), move to the predefined coordinates.\n"
            "### Detected Objects:\n"
            "If the user requests to navigate to a detected object (e.g., 'Navigate to the detected chair' or 'Go to the person'), "
            "check if the object was recently detected. If found, navigate to it; otherwise, respond with: "
            "'The object is no longer visible, I cannot navigate to it.'"
        )

    def get_example_messages(self) -> str:
        return (
            "\n**Examples:**\n"
            "User: What can you see around you?\n"
            "TCC-Robo: Action 1: Describe surroundings.\n"
            "\nUser: Move forward 2 meters at 0.2m/s and then turn right at 30 deg/s.\n"
            "TCC-Robo:\n"
            "Action 1: Move forward 2 meters at 0.2 m/s.\n"
            "Action 2: Turn right 90 degrees at 30 deg/s.\n"
            "\nUser: Navigate to the kitchen at 0.5 m/s.\n"
            "TCC-Robo:\n"
            "Action 1: Navigate to the kitchen at 0.5 m/s.\n"
            "\nUser: Report me your current location.\n"
            "TCC-Robo:\n"
            "Action 1: Report coordinates.\n"
            "\nUser: Go to the detected object.\n"
            "TCC-Robo:\n"
            "Action 1: Navigate to the detected object.\n"
            "\nUser: Move toward the chair you detected.\n"
            "TCC-Robo:\n"
            "Action 1: Navigate to the detected chair.\n"
            "\nUser: Rotate to face the coffee table and move forward 1 meter.\n"
            "TCC-Robo:\n"
            "Action 1: Rotate to face the coffee table.\n"
            "Action 2: Move forward 1 meter.\n"
            "\nUser: Turn left 90 degrees, move forward 4 meters, head to the kitchen, describe the surroundings, "
            "navigate to a detected object, report the coordinates of your current location, and report object locations.\n"
            "TCC-Robo:\n"
            "Action 1: Turn left 90 degrees.\n"
            "Action 2: Move forward 4 meters.\n"
            "Action 3: Navigate to the kitchen.\n"
            "Action 4: Describe surroundings.\n"
            "Action 5: Go to the detected object.\n"
            "Action 6: Report coordinates.\n"
            "Action 7: Report object locations.\n"
            "\nUser: Head to the location x:0, y:0, z:0.\n"
            "TCC-Robo:\n"
            "Action 1: Go to coordinates x:0, y:0, z:0.\n"
            "\nUser: Report the robot status, current yaw, cardinal direction, and robot poses.\n"
            "TCC-Robo:\n"
            "Action 1: Report orientation.\n"
            "\nUser: Send me a photo of your surroundings.\n"
            "TCC-Robo:\n"
            "Action 1: Send image.\n"
            "\nUser: Move in a circle with a radius of 2 meters at 0.5 m/s.\n"
            "TCC-Robo:\n"
            "Action 1: Move in a circle of radius 2 meters at 0.5 m/s.\n"
            "\nUser: Drive in a circle at 0.3 m/s.\n"
            "TCC-Robo:\n"
            "Action 1: Move in a circle of radius 1 meter at 0.3 m/s.\n"
            "\nUser: Move clockwise in a circle of diameter 1 meter at 1 m/s.\n"
            "TCC-Robo:\n"
            "Action 1: Move in a circle of radius 0.5 meters at 1 m/s.\n"
            "\nUser: Go in an arc of 45 degrees with a radius of 1 meter.\n"
            "TCC-Robo:\n"
            "Action 1: Move in an arc of 45 degrees with a radius of 1 meter.\n"
            "\nUser: Pause for at least 15 seconds.\n"
            "TCC-Robo:\n"
            "Action 1: Wait for 15 seconds.\n"
            "\nUser: Move in a 'Z' shape pattern with horizontal lengths of 1 meter and diagonal lengths of 2 meters.\n"
            "TCC-Robo:\n"
            "Action 1: Move forward 1m.\n"
            "Action 2: Turn left 135 degrees.\n"
            "Action 3: Move forward 2m.\n"
            "Action 4: Turn right 135 degrees.\n"
            "Action 5: Move forward 1m.\n"
            "\nUser: Stop immediately!\n"
            "TCC-Robo:\n"
            "Action 1: STOP.\n"
            "\nUser: Emergency halt!\n"
            "TCC-Robo:\n"
            "Action 1: STOP.\n"
        )

    def build_combined_system_message(self, input_params: Dict[str, str]) -> str:
        current_yaw = input_params.get("current_yaw", "unknown")
        cardinal_direction = input_params.get("cardinal_direction", "unknown")
        position_x = input_params.get("position_x", "unknown")
        position_y = input_params.get("position_y", "unknown")
        position_z = input_params.get("position_z", "unknown")
        parts = [
            self.get_system_message_status(current_yaw, cardinal_direction, position_x, position_y, position_z),
            self.get_system_message_commands(),
            self.get_system_message_navigation(),
            self.get_example_messages(),
        ]
        return "\n".join(parts)

    def process_input(self, input_text: str, **input_params) -> Dict[str, Any]:
        combined_system_message = self.build_combined_system_message(input_params)
        generated_text = self.call_llm(combined_system_message, input_text)
        rospy.loginfo("LLM output: %s", generated_text)
        if self.is_query_response(generated_text):
            return {'type': 'RESPONSE', 'content': generated_text}
        else:
            return {'type': 'ACTIONS', 'content': generated_text}

    def is_query_response(self, response_text: str) -> bool:
        return 'Action' not in response_text
