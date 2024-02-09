#!/usr/bin/env python
import tkinter as tk
from std_msgs.msg import String
import rospy
import actionlib
from actionlib import SimpleActionClient
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
import time

class ChatInterface(tk.Tk):
    def __init__(self):
        super().__init__()

        rospy.init_node('llm_input_publisher', anonymous=True)
        self.llm_input_publisher = rospy.Publisher('/llm_input', String, queue_size=10)
        rospy.Subscriber('/llm_output', String, self.add_llm_response)

        self.title("Chat With Robot")
        self.geometry("400x500")

        self.chat_history = tk.Text(self, wrap=tk.WORD, state=tk.DISABLED, bd=0, bg="lightgray", padx=10, pady=10)
        self.chat_history.pack(fill=tk.BOTH, expand=True)

        self.user_input_entry = tk.Entry(self, bd=0, bg="white", font=("Arial", 12))
        self.user_input_entry.pack(fill=tk.X, padx=10, pady=5)

        self.send_button = tk.Button(self, text="Send", command=self.send_message, bd=0, bg="lightpink", activebackground="lightblue")
        self.send_button.pack()

        self.user_input_entry.bind("<Return>", self.send_message)

        self.you_icon = tk.PhotoImage(file="you_icon.png")
        self.llm_icon = tk.PhotoImage(file="llm_icon.png")

    def add_llm_response(self, data):
        response = data.data
        self.add_message("Robot", response, color="lightblue", is_user=False)

    def send_message(self, event=None):
        user_input = self.user_input_entry.get()
        if user_input:
            self.llm_input_publisher.publish(String(data=user_input))
            self.add_message("You", user_input, color="lightgreen", is_user=True)
            self.user_input_entry.delete(0, tk.END)

    def add_message(self, sender, message, color, is_user):
        if is_user:
            icon = self.you_icon
        else:
            icon = self.llm_icon

        formatted_message = f"{sender}: {message}\n"
        message_tag = f"{sender}_tag"

        self.chat_history.config(state=tk.NORMAL)

        if icon:
            self.chat_history.image_create(tk.END, image=icon)
            self.chat_history.insert(tk.END, " ")

        self.chat_history.insert(tk.END, formatted_message, message_tag)
        self.chat_history.tag_config(message_tag, lmargin1=20, lmargin2=20, rmargin=20, background=color, foreground="black", font=("Arial", 10, "bold"))

        self.chat_history.insert(tk.END, "\n")

        self.chat_history.config(state=tk.DISABLED)

if __name__ == '__main__':
    chat_interface = ChatInterface()
    chat_interface.mainloop()
    
