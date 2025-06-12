"""
This script is based on the paper, "The Conversation is the Command: Interacting with Real-World Autonomous Robots Through Natural Language": https://dl.acm.org/doi/abs/10.1145/3610978.3640723
Its usage is subject to the  Creative Commons Attribution International 4.0 License.
"""
#!/usr/bin/env python
import tkinter as tk
from tkinter import ttk
from std_msgs.msg import String
from sensor_msgs.msg import Image
import rospy
from cv_bridge import CvBridge, CvBridgeError
import cv2
from PIL import Image as PILImage, ImageTk
import io
import speech_recognition as sr
import pyttsx3
from gtts import gTTS
import tempfile
import os
import threading
import time  
import traceback 

class TTSInterface:
    def __init__(self, tts_topic='/tts_text'):
        self.tts_subscriber = rospy.Subscriber(tts_topic, String, self.handle_tts)
        self.is_speaking = False
        self.engine = pyttsx3.init()

    def handle_tts(self, msg):
        text = msg.data.strip()
        if text:
            self.is_speaking = True
            try:
                tts = gTTS(text=text, lang="en")
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=True) as fp:
                    tts.save(fp.name)
                    os.system(f"mpg123 -q {fp.name}")
            except Exception as e:
                rospy.logerr(f"Google TTS error: {e}")
                self.engine.say(text)
                self.engine.runAndWait()
            finally:
                self.is_speaking = False

class ChatInterface(tk.Tk):
    def __init__(self):
        super().__init__()
        rospy.init_node('ChatGUI', anonymous=True)
        self.llm_input_publisher = rospy.Publisher('/llm_input', String, queue_size=10)
        rospy.Subscriber('/llm_output', String, self.add_llm_response)
        rospy.Subscriber('/llm_image_output', Image, self.add_llm_image)
        self.bridge = CvBridge()
        self.title("TCC - Chat With Robot in Natural Language")
        self.geometry("550x650")
        self.configure(bg='white')
        self.chat_history = tk.Text(self, wrap=tk.WORD, state=tk.DISABLED, bg="#f0f0f0",
                                    padx=10, pady=10, font=("Arial", 12))
        self.chat_history.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.chat_history.tag_configure("sender", font=("Arial", 10, "bold"))
        self.chat_history.tag_configure("user_msg", foreground="green", background="#e6ffe6",
                                        lmargin1=10, lmargin2=10)
        self.chat_history.tag_configure("bot_msg", foreground="blue", background="#e6f0ff",
                                        lmargin1=10, lmargin2=10)
        self.input_frame = tk.Frame(self, bg="white")
        self.input_frame.pack(fill=tk.X, padx=10, pady=5)
        self.user_input_entry = tk.Entry(self.input_frame, bd=1, bg="white", font=("Arial", 12))
        self.user_input_entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.user_input_entry.bind("<Return>", self.send_message)
        self.send_button = tk.Button(self.input_frame, text="Send", command=self.send_message, bd=1,
                                     bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), activebackground="#45a049")
        self.send_button.pack(side=tk.RIGHT, padx=(0, 5))
        self.listen_button = tk.Button(self.input_frame, text="Listen", command=self.start_listening_thread, bd=1,
                                       bg="#2196F3", fg="white", font=("Arial", 12, "bold"), activebackground="#1E88E5")
        self.listen_button.pack(side=tk.RIGHT, padx=(0, 5))
        self.tts_interface = TTSInterface(tts_topic='/tts_text')
        script_dir = os.path.dirname(os.path.abspath(__file__))
        try:
            self.you_icon = tk.PhotoImage(file=os.path.join(script_dir, "you_icon.png"))
            self.llm_icon = tk.PhotoImage(file=os.path.join(script_dir, "llm_icon.png"))
        except Exception as e:
            rospy.logerr(f"Error loading icons: {e}")
            self.you_icon = tk.PhotoImage(width=1, height=1)
            self.llm_icon = tk.PhotoImage(width=1, height=1)
        self.image_refs = []
        self.recognizer = sr.Recognizer()
        self.listening = False
        self.listen_thread = None
        self.microphone = sr.Microphone() 
        with self.microphone as source:
            rospy.loginfo("Calibrating microphone for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            rospy.loginfo(f"Energy threshold set to: {self.recognizer.energy_threshold}")

    def start_listening_thread(self):
        """Start voice recognition in a separate thread"""
        if self.tts_interface.is_speaking:
            self.add_message("System", "Please wait, I'm still speaking.", "bot_msg", False)
            return      
        if not self.listening:
            self.listening = True
            self.listen_button.config(text="Stop", bg="#FF5722")
            self.add_message("System", "Listening... Speak now", "bot_msg", False)
            self.listen_thread = threading.Thread(target=self.listen_and_transcribe)
            self.listen_thread.daemon = True
            self.listen_thread.start()
        else:
            self.stop_listening()

    def stop_listening(self):
        """Stop the listening process"""
        self.listening = False
        self.listen_button.config(text="Listen", bg="#2196F3")

    def listen_and_transcribe(self):
        try:
            if not self.microphone:
                rospy.logerr("No microphone available")
                self.after(0, self.add_message, "System", "No microphone detected", "bot_msg", False)
                return
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                rospy.loginfo(f"Adjusted energy threshold: {self.recognizer.energy_threshold}")
                time.sleep(0.5)  # Allow microphone to stabilize
                self.after(0, self.add_message, "System", "Listening... Speak now", "bot_msg", False)
                try:
                    audio = self.recognizer.listen(
                        source,
                        timeout=10,
                        phrase_time_limit=15
                    )
                    rospy.loginfo("Audio captured, recognizing...")
                    self.process_audio(audio)
                except sr.WaitTimeoutError:
                    self.after(0, self.add_message, "System", "No speech detected. Try speaking louder or closer to the microphone.", "bot_msg", False)
                except sr.UnknownValueError:
                    self.after(0, self.add_message, "System", "Could not understand audio (no match).", "bot_msg", False)
                except sr.RequestError as e:
                    rospy.logerr(f"Speech recognition API error: {e}")
                    self.after(0, self.add_message, "System", f"Speech service error: {e}", "bot_msg", False)
                except Exception as e:
                    rospy.logerr(f"Audio capture error: {e}")
                    rospy.logerr(traceback.format_exc())  # Detailed traceback
                    self.after(0, self.add_message, "System", f"Audio capture error: {e}", "bot_msg", False)
        except Exception as e:
            rospy.logerr(f"Error in voice recognition thread: {e}")
            rospy.logerr(traceback.format_exc())
            self.after(0, self.add_message, "System", f"Voice system error: {e}", "bot_msg", False)
        finally:
            self.after(0, self.stop_listening)

    def process_audio(self, audio):
        try:
            try:
                with open("/tmp/debug_audio.wav", "wb") as f:
                    f.write(audio.get_wav_data())
                rospy.loginfo("Saved debug audio to /tmp/debug_audio.wav")
            except Exception as e:
                rospy.logwarn(f"Could not save debug audio: {e}")
            result = self.recognizer.recognize_google(audio, show_all=True)
            rospy.loginfo(f"Google SR raw result: {result}")
            transcript = None
            if isinstance(result, dict) and 'alternative' in result and result['alternative']:
                transcript = result['alternative'][0].get('transcript', '')
            elif isinstance(result, str):
                transcript = result
            elif hasattr(result, 'alternative') and result.alternative:
                transcript = result.alternative[0].transcript
            if transcript:
                rospy.loginfo(f"Resolved transcript: {transcript}")
                self.publish_user_input(transcript)
            else:
                rospy.logwarn("No valid transcript found in response")
                self.after(0, self.add_message, "System", 
                        "Could not understand audio. Please try again.", "bot_msg", False)
        except sr.UnknownValueError:
            self.after(0, self.add_message, "System", "Could not understand audio (no match).", "bot_msg", False)
        except sr.RequestError as e:
            rospy.logerr(f"Speech recognition API error: {e}")
            self.after(0, self.add_message, "System", f"Speech service error: {e}", "bot_msg", False)
        except Exception as e:
            rospy.logerr(f"Audio processing error: {e}")
            rospy.logerr(traceback.format_exc())
            self.after(0, self.add_message, "System", f"Audio processing error: {e}", "bot_msg", False)
    
    def publish_user_input(self, text):
        self.llm_input_publisher.publish(String(data=text))
        self.add_message("You", text, "user_msg", True)

    def add_llm_response(self, data):
        response = data.data
        self.after(0, self.add_message, "TCC-Robo", response, "bot_msg", False)

    def add_llm_image(self, img_msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            rospy.loginfo("Image converted to OpenCV format.")
            pil_img = PILImage.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            pil_img = pil_img.resize((300, 300))
            tk_img = ImageTk.PhotoImage(pil_img)
            self.image_refs.append(tk_img)
            self.after(0, self.add_image_to_chat, tk_img)
        except CvBridgeError as e:
            rospy.logerr(f"Failed to convert image: {e}")
        except Exception as e:
            rospy.logerr(f"Unexpected error in add_llm_image: {e}")

    def add_image_to_chat(self, tk_img):
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.image_create(tk.END, image=tk_img)
        self.chat_history.insert(tk.END, "\n")
        self.chat_history.config(state=tk.DISABLED)
        self.update_idletasks()

    def send_message(self, event=None):
        user_input = self.user_input_entry.get().strip()
        if user_input:
            self.publish_user_input(user_input)
            self.user_input_entry.delete(0, tk.END)

    def add_message(self, sender, message, tag, is_user):
        icon = self.you_icon if is_user else self.llm_icon
        self.chat_history.config(state=tk.NORMAL)
        if icon:
            self.chat_history.image_create(tk.END, image=icon)
            self.chat_history.insert(tk.END, " ")
        formatted = f"{sender}: {message}\n"
        self.chat_history.insert(tk.END, formatted, ("sender", tag))
        self.chat_history.insert(tk.END, "\n")
        self.chat_history.config(state=tk.DISABLED)
        self.chat_history.see(tk.END)

    def on_closing(self):
        self.listening = False
        if self.listen_thread and self.listen_thread.is_alive():
            self.listen_thread.join(timeout=1.0)
        self.destroy()

if __name__ == '__main__':
    try:
        chat_interface = ChatInterface()
        chat_interface.protocol("WM_DELETE_WINDOW", chat_interface.on_closing)
        chat_interface.mainloop()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down Chat GUI.")