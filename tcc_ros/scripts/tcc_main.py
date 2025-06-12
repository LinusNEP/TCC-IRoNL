"""
This script is based on the paper, "The Conversation is the Command: Interacting with Real-World Autonomous Robots Through Natural Language": https://dl.acm.org/doi/abs/10.1145/3610978.3640723
Its usage is subject to the  Creative Commons Attribution International 4.0 License.
"""
#!/usr/bin/env python
import os
import rospy
from std_msgs.msg import String
#from tcc_ros.vlm_node_YOLO import PerceptionModule
from tcc_ros.vlm_node_SAM import PerceptionModule
from tcc_ros.llm_node import LLMInterface
from tcc_ros.commands_parser import CommandParser
from tcc_ros.action_executor import ActionExecutor
from tcc_ros.data_logger import DataLogger, LogEntry
from tcc_ros.llm_provider import LLMProviderFactory
from datetime import datetime

class RobotController:
    def __init__(self):
        rospy.init_node('robot_controller')
        self.data_logger = DataLogger()
        llm_provider_name = rospy.get_param("models/llm_provider", "openai")
        llm_api_key_from_config = rospy.get_param("models/llm_api_key", None)
        api_key_mapping = {
            "openai": "OPENAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "claude": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "llama.cpp": None  # No API key needed
        }
        api_key_env_var = api_key_mapping.get(llm_provider_name)
        api_key_value = None
        if api_key_env_var:
            if llm_api_key_from_config:
                rospy.loginfo(f"Using LLM API key from config.yaml for provider: {llm_provider_name}")
                api_key_value = llm_api_key_from_config
            else:
                api_key_value = os.getenv(api_key_env_var)
                if not api_key_value:
                    raise ValueError(f"API key for {llm_provider_name} not found. Please set {api_key_env_var} as an environment variable or in config.yaml.")
            os.environ[api_key_env_var] = api_key_value
        else:
            rospy.loginfo(f"No API key required for provider: {llm_provider_name}")       
        self.perception = PerceptionModule(self.data_logger)
        self.llm_interface = LLMInterface(api_key=api_key_value, destinations={})
        self.response_publisher = rospy.Publisher('/llm_output', String, queue_size=10)
        self.tts_publisher = rospy.Publisher('/tts_text', String, queue_size=10)
        self.action_executor = ActionExecutor(
            self.data_logger,
            self.perception,
            self.llm_interface,
        )
        self.action_executor.response_publisher = self.response_publisher
        self.action_executor.perception_module = self.perception
        self.llm_interface.destinations = self.action_executor.get_destinations()
        self.command_parser = CommandParser(self.action_executor)
        rospy.Subscriber('/llm_input', String, self.handle_input)

    def handle_input(self, msg):
        input_text = msg.data.strip()
        llm_output = self.llm_interface.process_input(input_text)
        self.response_publisher.publish(String(data=llm_output['content']))
        parsed_output = self.command_parser.parse_input(llm_output)
        if parsed_output['type'] == 'ACTIONS':
            actions = parsed_output['content']
            valid_actions = [a for a in actions if ...]
            self.action_executor.execute_actions(valid_actions)    
        else:
            rospy.logwarn("Unexpected output type from LLM.")

if __name__ == '__main__':
    controller = None
    try:
        controller = RobotController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        if controller and controller.data_logger:
            controller.data_logger.close()
