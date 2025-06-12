"""
This script is based on the paper, "The Conversation is the Command: Interacting with Real-World Autonomous Robots Through Natural Language": https://dl.acm.org/doi/abs/10.1145/3610978.3640723
Its usage is subject to the  Creative Commons Attribution International 4.0 License.
"""
import csv
import os
from datetime import datetime

class LogEntry:
    def __init__(self):
        self.timestamp = datetime.now().isoformat()
        self.input_command = ''  # c_n
        self.input_timestamp = ''  # t_n^r
        self.response_timestamp = ''  # t_n^s
        self.true_action_sequence = ''  # a_n
        self.predicted_action_sequence = ''  # \hat{a}_n
        #self.action_feedback = ''  # Feedbacks
        self.execution_success = ''  # s_n

class DataLogger:
    def __init__(self, log_file='robot_log.csv'):
        self.log_file = log_file
        file_exists = os.path.isfile(self.log_file)
        self.file = open(self.log_file, 'a', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.file)
        if not file_exists:
            self.csv_writer.writerow([
                'Timestamp',
                'Input Command',
                'Input Timestamp',
                'Response Timestamp',
                'True Action Sequence',
                'Predicted Action Sequence',
                #'Action Feedback',
                'Execution Success'
            ])
            self.file.flush()

    def log_entry(self, log_entry):
        self.csv_writer.writerow([
            log_entry.timestamp,
            log_entry.input_command,
            log_entry.input_timestamp,
            log_entry.response_timestamp,
            log_entry.true_action_sequence,
            log_entry.predicted_action_sequence,
            #log_entry.action_feedback,
            log_entry.execution_success
        ])
        self.file.flush()

    def close(self):
        self.file.close()
