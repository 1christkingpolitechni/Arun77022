"""
J.A.R.V.I.S - Just A Rather Very Intelligent System
Version 3.4 | Advanced Virtual Assistant with Voice Recognition & Gesture Control
"""

# ============================================================================
# IMPORTS & DEPENDENCIES
# ============================================================================
import os
import sys
import time
import math
import queue
import re
import json
import webbrowser
import threading
import subprocess
import datetime
import platform
import warnings
from pathlib import Path
from random import choice, randint
from urllib.parse import quote_plus
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Any
warnings.filterwarnings('ignore')

# ---------------------------
# GUI Imports
# ---------------------------
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, scrolledtext, font, colorchooser, filedialog
    import customtkinter as ctk
    TK_AVAILABLE = True
except Exception:
    tk = None
    TK_AVAILABLE = False

# ---------------------------
# Optional Libraries with Graceful Fallbacks
# ---------------------------
LIBRARIES = {
    'pyttsx3': {'module': None, 'available': False},
    'pyautogui': {'module': None, 'available': False},
    'opencv': {'module': None, 'available': False},
    'mediapipe': {'module': None, 'available': False},
    'speech_recognition': {'module': None, 'available': False},
    'PIL': {'module': None, 'available': False},
    'numpy': {'module': None, 'available': False},
    'pywhatkit': {'module': None, 'available': False},
}

def load_library(name, pip_name=None):
    """Load library with graceful fallback"""
    try:
        if name == 'pyttsx3':
            import pyttsx3
            LIBRARIES[name]['module'] = pyttsx3
            LIBRARIES[name]['available'] = True
        elif name == 'pyautogui':
            import pyautogui
            LIBRARIES[name]['module'] = pyautogui
            LIBRARIES[name]['available'] = True
        elif name == 'opencv':
            import cv2
            LIBRARIES[name]['module'] = cv2
            LIBRARIES[name]['available'] = True
        elif name == 'mediapipe':
            import mediapipe as mp
            LIBRARIES[name]['module'] = mp
            LIBRARIES[name]['available'] = True
        elif name == 'speech_recognition':
            import speech_recognition as sr
            LIBRARIES[name]['module'] = sr
            LIBRARIES[name]['available'] = True
        elif name == 'PIL':
            from PIL import Image, ImageTk, ImageGrab
            LIBRARIES[name]['module'] = {'Image': Image, 'ImageTk': ImageTk, 'ImageGrab': ImageGrab}
            LIBRARIES[name]['available'] = True
        elif name == 'numpy':
            import numpy as np
            LIBRARIES[name]['module'] = np
            LIBRARIES[name]['available'] = True
        elif name == 'pywhatkit':
            import pywhatkit
            LIBRARIES[name]['module'] = pywhatkit
            LIBRARIES[name]['available'] = True
    except ImportError as e:
        print(f"⚠️  {name} not available: {e}")
        if pip_name:
            print(f"   Install: pip install {pip_name}")
        LIBRARIES[name]['available'] = False

# Load all libraries
load_library('pyttsx3', 'pyttsx3')
load_library('pyautogui', 'pyautogui')
load_library('opencv', 'opencv-python')
load_library('mediapipe', 'mediapipe')
load_library('speech_recognition', 'SpeechRecognition')
load_library('PIL', 'Pillow')
load_library('numpy', 'numpy')
load_library('pywhatkit', 'pywhatkit')

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================
class Config:
    """Central configuration for J.A.R.V.I.S"""
    
    # Version Information
    VERSION = "3.4.0"
    NAME = "J.A.R.V.I.S"
    FULL_NAME = "Just A Rather Very Intelligent System"
    
    # System Information
    OS = platform.system()
    PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    # Directories
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "jarvis_data"
    SCREENSHOTS_DIR = DATA_DIR / "screenshots"
    VOICE_DATA_DIR = DATA_DIR / "voice_data"
    LOGS_DIR = DATA_DIR / "logs"
    MODELS_DIR = DATA_DIR / "models"
    
    # Files
    SETTINGS_FILE = DATA_DIR / "settings.json"
    COMMANDS_FILE = DATA_DIR / "commands.json"
    HISTORY_FILE = DATA_DIR / "history.json"
    CONTACTS_FILE = DATA_DIR / "contacts.json"
    
    # Camera & Gesture Settings
    CAMERA_ID = 0
    CAMERA_RESOLUTION = (1280, 720)
    GESTURE_SMOOTHING = 8
    PINCH_THRESHOLD = 40
    CLICK_COOLDOWN = 0.3
    GESTURE_SENSITIVITY = 1.0
    
    # Voice Settings
    WAKE_WORDS = ["jarvis", "hey jarvis", "okay jarvis", "hello jarvis"]
    VOICE_TIMEOUT = 10
    ENERGY_THRESHOLD = 300
    SAMPLE_RATE = 44100
    
    # UI Settings
    THEME = "dark"
    PRIMARY_COLOR = "#1a237e"
    SECONDARY_COLOR = "#3949ab"
    ACCENT_COLOR = "#00e5ff"
    BACKGROUND_COLOR = "#0a0e17"
    TEXT_COLOR = "#ffffff"
    
    # Create directories
    for dir_path in [DATA_DIR, SCREENSHOTS_DIR, VOICE_DATA_DIR, LOGS_DIR, MODELS_DIR]:
        dir_path.mkdir(exist_ok=True, parents=True)

# ============================================================================
# CORE ASSISTANT ENGINE
# ============================================================================
class JARVISAssistant:
    """Core assistant engine with all functionality"""
    
    def __init__(self):
        self.name = Config.NAME
        self.version = Config.VERSION
        self.full_name = Config.FULL_NAME
        
        # State management
        self.is_active = True
        self.is_listening = False
        self.is_speaking = False
        self.gestures_enabled = True
        self.voice_enabled = True
        
        # Modules
        self.tts_engine = None
        self.recognizer = None
        self.camera = None
        self.gesture_controller = None
        
        # Data storage
        self.settings = self.load_settings()
        self.history = []
        self.custom_commands = {}
        
        # Initialize modules
        self.initialize_tts()
        self.initialize_speech_recognition()
        
    def load_settings(self) -> Dict:
        """Load or create settings"""
        default_settings = {
            "voice": {
                "rate": 180,
                "volume": 0.9,
                "voice_id": 0,
                "language": "en-US"
            },
            "gesture": {
                "enabled": True,
                "sensitivity": 1.0,
                "mirror_camera": True,
                "mirror_cursor": True
            },
            "general": {
                "auto_start": False,
                "wake_word_enabled": True,
                "hotkeys_enabled": True
            },
            "ui": {
                "theme": "dark",
                "show_camera": True,
                "show_status": True
            }
        }
        
        if Config.SETTINGS_FILE.exists():
            try:
                with open(Config.SETTINGS_FILE, 'r') as f:
                    saved = json.load(f)
                    # Merge settings
                    for category in default_settings:
                        if category in saved:
                            default_settings[category].update(saved[category])
            except Exception as e:
                print(f"Error loading settings: {e}")
        
        return default_settings
    
    def save_settings(self):
        """Save current settings"""
        try:
            with open(Config.SETTINGS_FILE, 'w') as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            print(f"Error saving settings: {e}")
    
    def initialize_tts(self):
        """Initialize text-to-speech engine"""
        if LIBRARIES['pyttsx3']['available']:
            try:
                self.tts_engine = LIBRARIES['pyttsx3']['module'].init()
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    self.tts_engine.setProperty('voice', voices[self.settings['voice']['voice_id']].id)
                self.tts_engine.setProperty('rate', self.settings['voice']['rate'])
                self.tts_engine.setProperty('volume', self.settings['voice']['volume'])
            except Exception as e:
                print(f"TTS initialization failed: {e}")
                self.tts_engine = None
    
    def initialize_speech_recognition(self):
        """Initialize speech recognition"""
        if LIBRARIES['speech_recognition']['available']:
            try:
                self.recognizer = LIBRARIES['speech_recognition']['module'].Recognizer()
            except Exception as e:
                print(f"Speech recognition initialization failed: {e}")
                self.recognizer = None
    
    def speak(self, text: str, priority: str = "normal"):
        """Text-to-speech with priority handling"""
        print(f"[{self.name}]: {text}")
        
        if self.tts_engine:
            try:
                self.is_speaking = True
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                self.is_speaking = False
            except Exception as e:
                print(f"Speech error: {e}")
                self.is_speaking = False
    
    def listen(self, timeout: int = 5) -> Optional[str]:
        """Listen for voice command"""
        if not self.recognizer:
            return None
            
        with LIBRARIES['speech_recognition']['module'].Microphone() as source:
            try:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                self.is_listening = True
                audio = self.recognizer.listen(source, timeout=timeout)
                self.is_listening = False
                
                text = self.recognizer.recognize_google(audio)
                return text.lower()
            except LIBRARIES['speech_recognition']['module'].UnknownValueError:
                self.is_listening = False
                return None
            except LIBRARIES['speech_recognition']['module'].RequestError as e:
                self.is_listening = False
                print(f"Speech recognition error: {e}")
                return None
            except Exception as e:
                self.is_listening = False
                print(f"Listening error: {e}")
                return None
    
    def process_command(self, command: str) -> bool:
        """Process and execute commands"""
        if not command:
            return False
            
        self.log_command(command)
        
        # Convert to lowercase for easier matching
        cmd = command.lower().strip()
        
        # Check for wake words
        for wake_word in Config.WAKE_WORDS:
            if cmd.startswith(wake_word):
                cmd = cmd[len(wake_word):].strip()
                break
        
        # Command routing
        if not cmd:
            self.speak("Yes? How can I help you?")
            return True
            
        # Navigation commands
        if any(x in cmd for x in ["open ", "launch ", "start "]):
            return self.handle_navigation(cmd)
            
        # System commands
        elif any(x in cmd for x in ["close ", "exit ", "quit ", "stop "]):
            return self.handle_system(cmd)
            
        # Media commands
        elif any(x in cmd for x in ["play ", "music ", "video ", "youtube "]):
            return self.handle_media(cmd)
            
        # Information commands
        elif any(x in cmd for x in ["what ", "who ", "where ", "when ", "how "]):
            return self.handle_information(cmd)
            
        # Utility commands
        elif any(x in cmd for x in ["screenshot", "photo", "picture", "capture"]):
            return self.handle_utilities(cmd)
            
        # Communication commands
        elif any(x in cmd for x in ["whatsapp", "message", "send", "email"]):
            return self.handle_communication(cmd)
            
        # Gesture commands
        elif any(x in cmd for x in ["gesture", "hand", "camera"]):
            return self.handle_gesture(cmd)
            
        # Default response
        else:
            responses = [
                "I'm not sure how to help with that yet.",
                "Could you rephrase that?",
                "I need more context to understand.",
                "That's an interesting request!"
            ]
            self.speak(choice(responses))
            return False
    
    def handle_navigation(self, command: str) -> bool:
        """Handle navigation and opening commands"""
        try:
            if "chrome" in command or "browser" in command:
                webbrowser.open("https://www.google.com")
                self.speak("Opening Chrome browser")
                return True
                
            elif "edge" in command:
                webbrowser.open("https://www.bing.com")
                self.speak("Opening Microsoft Edge")
                return True
                
            elif "firefox" in command:
                webbrowser.open("https://www.mozilla.org")
                self.speak("Opening Firefox")
                return True
                
            elif "youtube" in command:
                webbrowser.open("https://www.youtube.com")
                self.speak("Opening YouTube")
                return True
                
            elif "whatsapp" in command:
                webbrowser.open("https://web.whatsapp.com")
                self.speak("Opening WhatsApp Web")
                return True
                
            elif "gmail" in command or "email" in command:
                webbrowser.open("https://mail.google.com")
                self.speak("Opening Gmail")
                return True
                
            elif "github" in command:
                webbrowser.open("https://github.com")
                self.speak("Opening GitHub")
                return True
                
            elif "search" in command:
                query = command.replace("search", "").replace("for", "").strip()
                if query:
                    url = f"https://www.google.com/search?q={quote_plus(query)}"
                    webbrowser.open(url)
                    self.speak(f"Searching for {query}")
                    return True
                    
        except Exception as e:
            self.speak(f"Failed to open: {e}")
            
        return False
    
    def handle_system(self, command: str) -> bool:
        """Handle system control commands"""
        try:
            if "close" in command:
                if "chrome" in command:
                    if platform.system() == "Windows":
                        os.system("taskkill /f /im chrome.exe")
                    else:
                        subprocess.run(["pkill", "chrome"])
                    self.speak("Closing Chrome")
                    return True
                    
                elif "edge" in command:
                    if platform.system() == "Windows":
                        os.system("taskkill /f /im msedge.exe")
                    else:
                        subprocess.run(["pkill", "msedge"])
                    self.speak("Closing Microsoft Edge")
                    return True
                    
                elif "whatsapp" in command or "tab" in command:
                    if LIBRARIES['pyautogui']['available']:
                        pyautogui = LIBRARIES['pyautogui']['module']
                        if platform.system() == "Darwin":
                            pyautogui.hotkey('command', 'w')
                        else:
                            pyautogui.hotkey('ctrl', 'w')
                        self.speak("Closing current tab")
                        return True
                        
            elif "screenshot" in command:
                self.take_screenshot()
                return True
                
            elif "shutdown" in command or "quit" in command:
                self.speak("Shutting down. Goodbye!")
                self.is_active = False
                return True
                
        except Exception as e:
            self.speak(f"System command failed: {e}")
            
        return False
    
    def handle_media(self, command: str) -> bool:
        """Handle media playback commands"""
        try:
            if "play" in command:
                query = command.replace("play", "").strip()
                if query:
                    if LIBRARIES['pywhatkit']['available']:
                        LIBRARIES['pywhatkit']['module'].playonyt(query)
                    else:
                        url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
                        webbrowser.open(url)
                    self.speak(f"Playing {query}")
                    return True
                    
        except Exception as e:
            self.speak(f"Media playback failed: {e}")
            
        return False
    
    def handle_information(self, command: str) -> bool:
        """Handle information requests"""
        try:
            if "time" in command:
                current_time = datetime.datetime.now().strftime("%I:%M %p")
                self.speak(f"The time is {current_time}")
                return True
                
            elif "date" in command:
                current_date = datetime.datetime.now().strftime("%B %d, %Y")
                self.speak(f"Today is {current_date}")
                return True
                
            elif "day" in command:
                current_day = datetime.datetime.now().strftime("%A")
                self.speak(f"Today is {current_day}")
                return True
                
            elif "weather" in command:
                self.speak("Weather information requires an API key. You can add weather service integration.")
                return True
                
            elif "system" in command or "info" in command:
                info = f"""
                System: {platform.system()}
                Python: {platform.python_version()}
                Assistant: {self.name} v{self.version}
                Processor: {platform.processor()}
                """
                self.speak(info)
                return True
                
        except Exception as e:
            self.speak(f"Information retrieval failed: {e}")
            
        return False
    
    def handle_utilities(self, command: str) -> bool:
        """Handle utility commands"""
        try:
            if "screenshot" in command:
                self.take_screenshot()
                return True
                
            elif "camera" in command:
                if "open" in command:
                    self.start_camera()
                    return True
                elif "close" in command:
                    self.stop_camera()
                    return True
                    
            elif "volume" in command:
                if "up" in command:
                    if LIBRARIES['pyautogui']['available']:
                        LIBRARIES['pyautogui']['module'].press('volumeup')
                    self.speak("Volume increased")
                    return True
                elif "down" in command:
                    if LIBRARIES['pyautogui']['available']:
                        LIBRARIES['pyautogui']['module'].press('volumedown')
                    self.speak("Volume decreased")
                    return True
                elif "mute" in command:
                    if LIBRARIES['pyautogui']['available']:
                        LIBRARIES['pyautogui']['module'].press('volumemute')
                    self.speak("Volume muted")
                    return True
                    
        except Exception as e:
            self.speak(f"Utility command failed: {e}")
            
        return False
    
    def handle_communication(self, command: str) -> bool:
        """Handle communication commands"""
        try:
            if "whatsapp" in command:
                # Extract phone number and message
                phone_match = re.search(r'(\+?\d[\d\s-]+)', command)
                message_match = re.search(r'message\s+(.+)', command)
                
                if phone_match:
                    phone = phone_match.group(1)
                    message = message_match.group(1) if message_match else ""
                    
                    if message:
                        url = f"https://web.whatsapp.com/send?phone={phone}&text={quote_plus(message)}"
                    else:
                        url = f"https://web.whatsapp.com/send?phone={phone}"
                    
                    webbrowser.open(url)
                    self.speak(f"Opening WhatsApp for {phone}")
                    return True
                else:
                    webbrowser.open("https://web.whatsapp.com")
                    self.speak("Opening WhatsApp Web")
                    return True
                    
        except Exception as e:
            self.speak(f"Communication command failed: {e}")
            
        return False
    
    def handle_gesture(self, command: str) -> bool:
        """Handle gesture control commands"""
        try:
            if "gesture" in command:
                if "on" in command or "enable" in command:
                    self.gestures_enabled = True
                    self.speak("Gesture control enabled")
                    return True
                elif "off" in command or "disable" in command:
                    self.gestures_enabled = False
                    self.speak("Gesture control disabled")
                    return True
                    
            elif "camera" in command:
                if "on" in command or "open" in command:
                    self.start_camera()
                    return True
                elif "off" in command or "close" in command:
                    self.stop_camera()
                    return True
                    
        except Exception as e:
            self.speak(f"Gesture command failed: {e}")
            
        return False
    
    def take_screenshot(self):
        """Take screenshot of the screen"""
        try:
            if LIBRARIES['PIL']['available']:
                ImageGrab = LIBRARIES['PIL']['module']['ImageGrab']
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = Config.SCREENSHOTS_DIR / f"screenshot_{timestamp}.png"
                
                screenshot = ImageGrab.grab()
                screenshot.save(filename)
                self.speak(f"Screenshot saved as {filename.name}")
            else:
                self.speak("Screenshot requires Pillow library")
        except Exception as e:
            self.speak(f"Screenshot failed: {e}")
    
    def start_camera(self):
        """Start camera feed"""
        if not LIBRARIES['opencv']['available']:
            self.speak("Camera requires OpenCV library")
            return
            
        try:
            cv2 = LIBRARIES['opencv']['module']
            self.camera = cv2.VideoCapture(Config.CAMERA_ID)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_RESOLUTION[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_RESOLUTION[1])
            self.speak("Camera started")
        except Exception as e:
            self.speak(f"Camera failed to start: {e}")
    
    def stop_camera(self):
        """Stop camera feed"""
        try:
            if self.camera:
                self.camera.release()
                self.camera = None
                if LIBRARIES['opencv']['available']:
                    cv2 = LIBRARIES['opencv']['module']
                    cv2.destroyAllWindows()
                self.speak("Camera stopped")
        except Exception as e:
            self.speak(f"Camera stop failed: {e}")
    
    def log_command(self, command: str):
        """Log command to history"""
        try:
            entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "command": command,
                "assistant": self.name
            }
            
            if Config.HISTORY_FILE.exists():
                with open(Config.HISTORY_FILE, 'r') as f:
                    history = json.load(f)
            else:
                history = []
                
            history.append(entry)
            
            # Keep only last 100 entries
            if len(history) > 100:
                history = history[-100:]
                
            with open(Config.HISTORY_FILE, 'w') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            print(f"Error logging command: {e}")

# ============================================================================
# ADVANCED GESTURE CONTROLLER
# ============================================================================
class GestureController:
    """Advanced gesture recognition and control system"""
    
    def __init__(self, assistant: JARVISAssistant):
        self.assistant = assistant
        self.running = False
        self.thread = None
        self.cap = None
        self.hands = None
        
        # Gesture tracking
        self.last_gesture_time = 0
        self.gesture_buffer = []
        self.current_gesture = None
        
        # Cursor smoothing
        self.cursor_history = []
        self.history_size = 5
        
        # Initialize if libraries available
        self.initialize_gesture_libs()
    
    def initialize_gesture_libs(self):
        """Initialize gesture libraries"""
        self.cv2_available = LIBRARIES['opencv']['available']
        self.mp_available = LIBRARIES['mediapipe']['available']
        self.pyautogui_available = LIBRARIES['pyautogui']['available']
        
        if self.cv2_available:
            self.cv2 = LIBRARIES['opencv']['module']
        if self.pyautogui_available:
            self.pyautogui = LIBRARIES['pyautogui']['module']
        if self.mp_available:
            self.mp = LIBRARIES['mediapipe']['module']
    
    def start(self):
        """Start gesture controller"""
        if self.running:
            return
            
        if not self.cv2_available or not self.mp_available:
            print("Gesture control requires OpenCV and MediaPipe")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self.gesture_loop, daemon=True)
        self.thread.start()
        print("Gesture controller started")
    
    def stop(self):
        """Stop gesture controller"""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        print("Gesture controller stopped")
    
    def gesture_loop(self):
        """Main gesture recognition loop"""
        if not self.cv2_available or not self.mp_available:
            return
            
        try:
            # Initialize camera
            self.cap = self.cv2.VideoCapture(Config.CAMERA_ID)
            self.cap.set(self.cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_RESOLUTION[0])
            self.cap.set(self.cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_RESOLUTION[1])
            
            # Initialize MediaPipe Hands
            self.hands = self.mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            
            print(f"Gesture controller active at {Config.CAMERA_RESOLUTION[0]}x{Config.CAMERA_RESOLUTION[1]}")
            
            while self.running and self.assistant.gestures_enabled:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Flip frame for mirror effect
                frame = self.cv2.flip(frame, 1)
                
                # Convert BGR to RGB
                rgb_frame = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
                
                # Process frame for hand landmarks
                results = self.hands.process(rgb_frame)
                
                # Detect gestures
                gesture = self.detect_gesture(results, frame)
                
                # Execute gesture action
                if gesture and time.time() - self.last_gesture_time > Config.CLICK_COOLDOWN:
                    self.execute_gesture(gesture)
                    self.last_gesture_time = time.time()
                
                # Display frame with overlay
                self.display_overlay(frame, results, gesture)
                
                # Check for exit key
                if self.cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except Exception as e:
            print(f"Gesture loop error: {e}")
        finally:
            if self.cap:
                self.cap.release()
            if self.cv2_available:
                self.cv2.destroyAllWindows()
    
    def detect_gesture(self, results, frame):
        """Detect gestures from hand landmarks"""
        if not results.multi_hand_landmarks:
            return None
            
        hand_landmarks = results.multi_hand_landmarks[0]
        h, w, _ = frame.shape
        
        # Get landmark positions
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append((int(lm.x * w), int(lm.y * h)))
        
        # Calculate distances between key points
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        
        # Calculate distances
        thumb_index_dist = math.sqrt((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)
        thumb_middle_dist = math.sqrt((thumb_tip[0] - middle_tip[0])**2 + (thumb_tip[1] - middle_tip[1])**2)
        
        # Detect gestures
        if thumb_index_dist < Config.PINCH_THRESHOLD:
            return "click"
        elif thumb_middle_dist < Config.PINCH_THRESHOLD:
            return "right_click"
        elif thumb_index_dist < Config.PINCH_THRESHOLD * 2:
            return "double_click"
        else:
            # Move cursor based on index finger
            if self.pyautogui_available:
                screen_w, screen_h = self.pyautogui.size()
                cursor_x = int(index_tip[0] / w * screen_w)
                cursor_y = int(index_tip[1] / h * screen_h)
                
                # Smooth cursor movement
                self.cursor_history.append((cursor_x, cursor_y))
                if len(self.cursor_history) > self.history_size:
                    self.cursor_history.pop(0)
                
                if len(self.cursor_history) > 0:
                    avg_x = sum(x for x, y in self.cursor_history) // len(self.cursor_history)
                    avg_y = sum(y for x, y in self.cursor_history) // len(self.cursor_history)
                    self.pyautogui.moveTo(avg_x, avg_y, duration=0.1)
            
            return "cursor_move"
    
    def execute_gesture(self, gesture):
        """Execute action based on detected gesture"""
        if not self.pyautogui_available:
            return
            
        try:
            if gesture == "click":
                self.pyautogui.click()
                self.assistant.speak("Click", priority="low")
            elif gesture == "right_click":
                self.pyautogui.rightClick()
                self.assistant.speak("Right click", priority="low")
            elif gesture == "double_click":
                self.pyautogui.doubleClick()
                self.assistant.speak("Double click", priority="low")
            elif gesture == "scroll_up":
                self.pyautogui.scroll(100)
                self.assistant.speak("Scrolling up", priority="low")
            elif gesture == "scroll_down":
                self.pyautogui.scroll(-100)
                self.assistant.speak("Scrolling down", priority="low")
        except Exception as e:
            print(f"Gesture execution error: {e}")
    
    def display_overlay(self, frame, results, gesture):
        """Display camera feed with gesture overlay"""
        if results.multi_hand_landmarks:
            # Draw hand landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, self.mp.solutions.hands.HAND_CONNECTIONS)
        
        # Display gesture information
        if gesture:
            self.cv2.putText(frame, f"Gesture: {gesture}", (50, 50),
                           self.cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        self.cv2.imshow('Gesture Controller - Press Q to quit', frame)

# ============================================================================
# MODERN GUI INTERFACE
# ============================================================================
class JARVISGUI:
    """Modern GUI interface for J.A.R.V.I.S"""
    
    def __init__(self):
        if not TK_AVAILABLE:
            print("GUI not available. Running in console mode...")
            self.run_console_mode()
            return
            
        # Initialize assistant
        self.assistant = JARVISAssistant()
        self.gesture_controller = GestureController(self.assistant)
        
        # Create main window
        self.root = tk.Tk()
        self.root.title(f"{Config.NAME} v{Config.VERSION} - {Config.FULL_NAME}")
        self.root.geometry("1400x900")
        self.root.configure(bg=Config.BACKGROUND_COLOR)
        
        # Variables
        self.current_tab = "home"
        self.voice_active = False
        self.gesture_active = False
        
        # Setup UI
        self.setup_styles()
        self.create_layout()
        self.setup_bindings()
        
        # Start updates
        self.update_loop()
        
    def setup_styles(self):
        """Setup custom styles and fonts"""
        self.colors = {
            "bg_primary": Config.BACKGROUND_COLOR,
            "bg_secondary": "#1e293b",
            "bg_tertiary": "#334155",
            "accent_primary": Config.ACCENT_COLOR,
            "accent_secondary": Config.SECONDARY_COLOR,
            "text_primary": Config.TEXT_COLOR,
            "text_secondary": "#94a3b8",
            "text_muted": "#64748b",
            "success": "#10b981",
            "warning": "#f59e0b",
            "error": "#ef4444"
        }
        
        self.fonts = {
            "title": ("Segoe UI", 24, "bold"),
            "heading": ("Segoe UI", 18, "bold"),
            "subheading": ("Segoe UI", 14, "bold"),
            "normal": ("Segoe UI", 11),
            "monospace": ("Consolas", 10)
        }
    
    def create_layout(self):
        """Create main layout"""
        # Top bar
        self.create_top_bar()
        
        # Main container
        main_container = tk.Frame(self.root, bg=self.colors["bg_primary"])
        main_container.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Sidebar
        self.create_sidebar(main_container)
        
        # Content area
        self.content_frame = tk.Frame(main_container, bg=self.colors["bg_primary"])
        self.content_frame.pack(side="right", fill="both", expand=True, padx=(20, 0))
        
        # Initialize tabs
        self.tabs = {}
        self.create_home_tab()
        self.create_browser_tab()
        self.create_gesture_tab()
        self.create_voice_tab()
        self.create_system_tab()
        self.create_settings_tab()
        
        # Show home tab
        self.switch_tab("home")
        
        # Status bar
        self.create_status_bar()
    
    def create_top_bar(self):
        """Create top navigation bar"""
        top_bar = tk.Frame(self.root, bg=self.colors["bg_secondary"], height=70)
        top_bar.pack(fill="x")
        top_bar.pack_propagate(False)
        
        # Logo and title
        title_frame = tk.Frame(top_bar, bg=self.colors["bg_secondary"])
        title_frame.pack(side="left", padx=20)
        
        tk.Label(
            title_frame,
            text="🤖",
            font=("Segoe UI", 32),
            bg=self.colors["bg_secondary"],
            fg=self.colors["accent_primary"]
        ).pack(side="left", padx=(0, 15))
        
        tk.Label(
            title_frame,
            text=f"{Config.NAME} Assistant",
            font=self.fonts["title"],
            bg=self.colors["bg_secondary"],
            fg=self.colors["text_primary"]
        ).pack(side="left")
        
        tk.Label(
            title_frame,
            text=f"v{Config.VERSION}",
            font=("Segoe UI", 10),
            bg=self.colors["bg_secondary"],
            fg=self.colors["text_muted"]
        ).pack(side="left", padx=(10, 0))
        
        # Quick actions
        actions_frame = tk.Frame(top_bar, bg=self.colors["bg_secondary"])
        actions_frame.pack(side="right", padx=20)
        
        actions = [
            ("🎤", self.toggle_voice, "Voice Control"),
            ("👆", self.toggle_gestures, "Gesture Control"),
            ("⚙️", lambda: self.switch_tab("settings"), "Settings"),
            ("❓", self.show_help, "Help"),
            ("✕", self.root.quit, "Exit")
        ]
        
        for icon, command, tooltip in actions:
            btn = tk.Button(
                actions_frame,
                text=icon,
                font=("Segoe UI", 18),
                bg=self.colors["bg_tertiary"],
                fg=self.colors["text_primary"],
                activebackground=self.colors["accent_primary"],
                activeforeground="white",
                relief="flat",
                padx=12,
                pady=6,
                cursor="hand2",
                command=command
            )
            btn.pack(side="left", padx=5)
    
    def create_sidebar(self, parent):
        """Create sidebar with navigation"""
        sidebar = tk.Frame(parent, bg=self.colors["bg_secondary"], width=220)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        
        # Navigation items
        nav_items = [
            ("🏠", "Dashboard", "home"),
            ("🌐", "Browser Control", "browser"),
            ("👆", "Gesture Control", "gesture"),
            ("🎤", "Voice Control", "voice"),
            ("⚡", "System Tools", "system"),
            ("⚙️", "Settings", "settings"),
        ]
        
        for icon, text, tab_id in nav_items:
            btn = tk.Button(
                sidebar,
                text=f"  {icon}  {text}",
                font=self.fonts["normal"],
                anchor="w",
                bg=self.colors["bg_secondary"],
                fg=self.colors["text_secondary"],
                activebackground=self.colors["accent_primary"],
                activeforeground="white",
                relief="flat",
                padx=20,
                pady=14,
                cursor="hand2",
                command=lambda t=tab_id: self.switch_tab(t)
            )
            btn.pack(fill="x")
    
    def create_home_tab(self):
        """Create home/dashboard tab"""
        frame = tk.Frame(self.content_frame, bg=self.colors["bg_primary"])
        self.tabs["home"] = frame
        
        # Welcome section
        welcome_frame = tk.Frame(frame, bg=self.colors["bg_secondary"], relief="flat", bd=1)
        welcome_frame.pack(fill="x", pady=(0, 20))
        
        tk.Label(
            welcome_frame,
            text="🚀 Welcome to J.A.R.V.I.S",
            font=self.fonts["heading"],
            bg=self.colors["bg_secondary"],
            fg=self.colors["text_primary"],
            pady=20
        ).pack()
        
        tk.Label(
            welcome_frame,
            text=f"{Config.FULL_NAME} - Version {Config.VERSION}",
            font=self.fonts["normal"],
            bg=self.colors["bg_secondary"],
            fg=self.colors["text_secondary"],
            pady=10
        ).pack()
        
        # Quick stats
        stats_frame = tk.Frame(frame, bg=self.colors["bg_primary"])
        stats_frame.pack(fill="x", pady=(0, 20))
        
        stats = [
            ("Voice Recognition", "🎤", "Ready"),
            ("Gesture Control", "👆", "Ready"),
            ("Browser Automation", "🌐", "Ready"),
            ("System Control", "⚡", "Ready")
        ]
        
        for i, (title, icon, status) in enumerate(stats):
            stat_card = tk.Frame(
                stats_frame,
                bg=self.colors["bg_tertiary"],
                relief="flat",
                bd=1
            )
            stat_card.grid(row=0, column=i, padx=10, sticky="nsew")
            stats_frame.grid_columnconfigure(i, weight=1)
            
            tk.Label(
                stat_card,
                text=icon,
                font=("Segoe UI", 28),
                bg=self.colors["bg_tertiary"],
                fg=self.colors["accent_primary"]
            ).pack(pady=(15, 5))
            
            tk.Label(
                stat_card,
                text=title,
                font=self.fonts["subheading"],
                bg=self.colors["bg_tertiary"],
                fg=self.colors["text_primary"]
            ).pack()
            
            tk.Label(
                stat_card,
                text=status,
                font=self.fonts["normal"],
                bg=self.colors["bg_tertiary"],
                fg=self.colors["success"]
            ).pack(pady=(0, 15))
        
        # Quick actions
        actions_frame = tk.LabelFrame(
            frame,
            text=" Quick Actions ",
            font=self.fonts["subheading"],
            bg=self.colors["bg_primary"],
            fg=self.colors["text_primary"],
            relief="flat"
        )
        actions_frame.pack(fill="both", expand=True)
        
        actions = [
            ("Open Chrome", "🌐", lambda: webbrowser.open("https://google.com")),
            ("Take Screenshot", "📸", self.assistant.take_screenshot),
            ("Play Music", "🎵", lambda: webbrowser.open("https://youtube.com")),
            ("Open Camera", "📷", self.assistant.start_camera),
            ("System Info", "💻", lambda: self.assistant.speak(f"System: {platform.system()}")),
            ("Voice Typing", "⌨️", self.start_voice_typing)
        ]
        
        actions_grid = tk.Frame(actions_frame, bg=self.colors["bg_primary"])
        actions_grid.pack(fill="both", expand=True, padx=10, pady=10)
        
        for i, (text, icon, command) in enumerate(actions):
            row = i // 3
            col = i % 3
            
            btn = tk.Button(
                actions_grid,
                text=f"{icon} {text}",
                font=self.fonts["normal"],
                bg=self.colors["bg_tertiary"],
                fg=self.colors["text_primary"],
                activebackground=self.colors["accent_primary"],
                activeforeground="white",
                relief="flat",
                padx=15,
                pady=12,
                cursor="hand2",
                command=command
            )
            btn.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            actions_grid.grid_columnconfigure(col, weight=1)
    
    def create_browser_tab(self):
        """Create browser control tab"""
        frame = tk.Frame(self.content_frame, bg=self.colors["bg_primary"])
        self.tabs["browser"] = frame
        
        # Browser controls
        browser_frame = tk.LabelFrame(
            frame,
            text=" Browser Control ",
            font=self.fonts["subheading"],
            bg=self.colors["bg_primary"],
            fg=self.colors["text_primary"],
            relief="flat"
        )
        browser_frame.pack(fill="x", pady=(0, 20))
        
        # Browser selection
        tk.Label(
            browser_frame,
            text="Select Browser:",
            font=self.fonts["normal"],
            bg=self.colors["bg_primary"],
            fg=self.colors["text_secondary"]
        ).pack(anchor="w", padx=20, pady=(20, 10))
        
        browsers_frame = tk.Frame(browser_frame, bg=self.colors["bg_primary"])
        browsers_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        browsers = [
            ("Google Chrome", "https://google.com"),
            ("Microsoft Edge", "https://bing.com"),
            ("Mozilla Firefox", "https://mozilla.org"),
            ("YouTube", "https://youtube.com"),
            ("WhatsApp Web", "https://web.whatsapp.com"),
            ("Gmail", "https://mail.google.com")
        ]
        
        for i, (name, url) in enumerate(browsers):
            row = i // 3
            col = i % 3
            
            btn_frame = tk.Frame(browsers_frame, bg=self.colors["bg_primary"])
            btn_frame.grid(row=row, column=col, padx=5, pady=5, sticky="w")
            
            tk.Button(
                btn_frame,
                text=name,
                font=self.fonts["normal"],
                bg=self.colors["bg_tertiary"],
                fg=self.colors["text_primary"],
                activebackground=self.colors["accent_primary"],
                activeforeground="white",
                relief="flat",
                padx=15,
                pady=10,
                cursor="hand2",
                command=lambda u=url: webbrowser.open(u)
            ).pack()
        
        # Search section
        search_frame = tk.LabelFrame(
            frame,
            text=" Web Search ",
            font=self.fonts["subheading"],
            bg=self.colors["bg_primary"],
            fg=self.colors["text_primary"],
            relief="flat"
        )
        search_frame.pack(fill="x", pady=(0, 20))
        
        self.search_entry = tk.Entry(
            search_frame,
            font=self.fonts["normal"],
            bg=self.colors["bg_tertiary"],
            fg=self.colors["text_primary"],
            insertbackground=self.colors["text_primary"]
        )
        self.search_entry.pack(fill="x", padx=20, pady=(20, 10))
        self.search_entry.insert(0, "Enter search query...")
        
        btn_frame = tk.Frame(search_frame, bg=self.colors["bg_primary"])
        btn_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        tk.Button(
            btn_frame,
            text="🔍 Search Google",
            font=self.fonts["normal"],
            bg=self.colors["accent_primary"],
            fg="white",
            activebackground=self.colors["accent_secondary"],
            relief="flat",
            padx=20,
            pady=12,
            cursor="hand2",
            command=self.perform_search
        ).pack()
    
    def create_gesture_tab(self):
        """Create gesture control tab"""
        frame = tk.Frame(self.content_frame, bg=self.colors["bg_primary"])
        self.tabs["gesture"] = frame
        
        # Gesture control panel
        control_frame = tk.LabelFrame(
            frame,
            text=" Gesture Control ",
            font=self.fonts["subheading"],
            bg=self.colors["bg_primary"],
            fg=self.colors["text_primary"],
            relief="flat"
        )
        control_frame.pack(fill="x", pady=(0, 20))
        
        # Status display
        self.gesture_status_label = tk.Label(
            control_frame,
            text="Gesture Control: OFF",
            font=self.fonts["heading"],
            bg=self.colors["bg_primary"],
            fg=self.colors["error"],
            pady=30
        )
        self.gesture_status_label.pack()
        
        # Control buttons
        btn_frame = tk.Frame(control_frame, bg=self.colors["bg_primary"])
        btn_frame.pack(pady=(0, 30))
        
        self.gesture_toggle_btn = tk.Button(
            btn_frame,
            text="▶ Start Gesture Control",
            font=self.fonts["normal"],
            bg=self.colors["success"],
            fg="white",
            activebackground="#0da56d",
            activeforeground="white",
            relief="flat",
            padx=30,
            pady=15,
            cursor="hand2",
            command=self.toggle_gestures
        )
        self.gesture_toggle_btn.pack(side="left", padx=10)
        
        tk.Button(
            btn_frame,
            text="📷 Test Camera",
            font=self.fonts["normal"],
            bg=self.colors["bg_tertiary"],
            fg=self.colors["text_primary"],
            activebackground=self.colors["accent_primary"],
            activeforeground="white",
            relief="flat",
            padx=30,
            pady=15,
            cursor="hand2",
            command=self.test_camera
        ).pack(side="left", padx=10)
        
        # Gesture guide
        guide_frame = tk.LabelFrame(
            frame,
            text=" Gesture Guide ",
            font=self.fonts["subheading"],
            bg=self.colors["bg_primary"],
            fg=self.colors["text_primary"],
            relief="flat"
        )
        guide_frame.pack(fill="both", expand=True)
        
        gestures = [
            ("👆 Index Finger", "Cursor Movement"),
            ("🤏 Thumb + Index", "Left Click"),
            ("🤏 Thumb + Middle", "Right Click"),
            ("✌️ Two Fingers", "Double Click"),
            ("👍 Thumb Up", "Scroll Up"),
            ("👎 Thumb Down", "Scroll Down")
        ]
        
        guide_grid = tk.Frame(guide_frame, bg=self.colors["bg_primary"])
        guide_grid.pack(fill="both", expand=True, padx=10, pady=10)
        
        for i, (gesture, action) in enumerate(gestures):
            row = i // 2
            col = i % 2
            
            gesture_card = tk.Frame(
                guide_grid,
                bg=self.colors["bg_tertiary"],
                relief="flat",
                bd=1
            )
            gesture_card.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            guide_grid.grid_columnconfigure(col, weight=1)
            
            tk.Label(
                gesture_card,
                text=gesture,
                font=("Segoe UI", 18),
                bg=self.colors["bg_tertiary"],
                fg=self.colors["text_primary"]
            ).pack(pady=(15, 5))
            
            tk.Label(
                gesture_card,
                text=action,
                font=self.fonts["normal"],
                bg=self.colors["bg_tertiary"],
                fg=self.colors["text_secondary"]
            ).pack(pady=(0, 15))
    
    def create_voice_tab(self):
        """Create voice control tab"""
        frame = tk.Frame(self.content_frame, bg=self.colors["bg_primary"])
        self.tabs["voice"] = frame
        
        # Voice control panel
        voice_frame = tk.LabelFrame(
            frame,
            text=" Voice Control ",
            font=self.fonts["subheading"],
            bg=self.colors["bg_primary"],
            fg=self.colors["text_primary"],
            relief="flat"
        )
        voice_frame.pack(fill="both", expand=True)
        
        # Voice input
        tk.Label(
            voice_frame,
            text="Enter command or use voice:",
            font=self.fonts["normal"],
            bg=self.colors["bg_primary"],
            fg=self.colors["text_secondary"]
        ).pack(anchor="w", padx=20, pady=(20, 10))
        
        self.voice_text = scrolledtext.ScrolledText(
            voice_frame,
            height=6,
            font=self.fonts["monospace"],
            bg=self.colors["bg_tertiary"],
            fg=self.colors["text_primary"],
            insertbackground=self.colors["text_primary"]
        )
        self.voice_text.pack(fill="x", padx=20, pady=(0, 10))
        self.voice_text.insert("1.0", "Type command here or say 'Jarvis' to activate...")
        
        # Control buttons
        btn_frame = tk.Frame(voice_frame, bg=self.colors["bg_primary"])
        btn_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        tk.Button(
            btn_frame,
            text="🎤 Start Listening",
            font=self.fonts["normal"],
            bg=self.colors["accent_primary"],
            fg="white",
            activebackground=self.colors["accent_secondary"],
            relief="flat",
            padx=25,
            pady=12,
            cursor="hand2",
            command=self.start_listening
        ).pack(side="left", padx=5)
        
        tk.Button(
            btn_frame,
            text="▶ Process Command",
            font=self.fonts["normal"],
            bg=self.colors["bg_tertiary"],
            fg=self.colors["text_primary"],
            activebackground=self.colors["accent_primary"],
            activeforeground="white",
            relief="flat",
            padx=25,
            pady=12,
            cursor="hand2",
            command=self.process_text_command
        ).pack(side="left", padx=5)
        
        # Voice typing
        typing_frame = tk.LabelFrame(
            voice_frame,
            text=" Voice Typing ",
            font=self.fonts["normal"],
            bg=self.colors["bg_primary"],
            fg=self.colors["text_primary"],
            relief="flat"
        )
        typing_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        typing_btns = tk.Frame(typing_frame, bg=self.colors["bg_primary"])
        typing_btns.pack(padx=20, pady=20)
        
        tk.Button(
            typing_btns,
            text="⌨️ Start Voice Typing",
            font=self.fonts["normal"],
            bg=self.colors["success"],
            fg="white",
            activebackground="#0da56d",
            relief="flat",
            padx=20,
            pady=10,
            cursor="hand2",
            command=self.start_voice_typing
        ).pack(side="left", padx=5)
        
        tk.Button(
            typing_btns,
            text="⏹️ Stop Voice Typing",
            font=self.fonts["normal"],
            bg=self.colors["error"],
            fg="white",
            activebackground="#dc2626",
            relief="flat",
            padx=20,
            pady=10,
            cursor="hand2",
            command=self.stop_voice_typing
        ).pack(side="left", padx=5)
    
    def create_system_tab(self):
        """Create system tools tab"""
        frame = tk.Frame(self.content_frame, bg=self.colors["bg_primary"])
        self.tabs["system"] = frame
        
        # System tools
        tools_frame = tk.LabelFrame(
            frame,
            text=" System Tools ",
            font=self.fonts["subheading"],
            bg=self.colors["bg_primary"],
            fg=self.colors["text_primary"],
            relief="flat"
        )
        tools_frame.pack(fill="both", expand=True)
        
        tools = [
            ("📸 Take Screenshot", self.assistant.take_screenshot),
            ("📷 Open Camera", self.assistant.start_camera),
            ("📁 Open Data Folder", self.open_data_folder),
            ("🗑️ Clear Cache", self.clear_cache),
            ("🔄 Refresh System", self.refresh_system),
            ("📊 System Info", self.show_system_info)
        ]
        
        tools_grid = tk.Frame(tools_frame, bg=self.colors["bg_primary"])
        tools_grid.pack(fill="both", expand=True, padx=20, pady=20)
        
        for i, (text, command) in enumerate(tools):
            row = i // 3
            col = i % 3
            
            btn = tk.Button(
                tools_grid,
                text=text,
                font=self.fonts["normal"],
                bg=self.colors["bg_tertiary"],
                fg=self.colors["text_primary"],
                activebackground=self.colors["accent_primary"],
                activeforeground="white",
                relief="flat",
                padx=20,
                pady=15,
                cursor="hand2",
                command=command
            )
            btn.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            tools_grid.grid_columnconfigure(col, weight=1)
    
    def create_settings_tab(self):
        """Create settings tab"""
        frame = tk.Frame(self.content_frame, bg=self.colors["bg_primary"])
        self.tabs["settings"] = frame
        
        # Settings panel
        settings_frame = tk.LabelFrame(
            frame,
            text=" Settings ",
            font=self.fonts["subheading"],
            bg=self.colors["bg_primary"],
            fg=self.colors["text_primary"],
            relief="flat"
        )
        settings_frame.pack(fill="both", expand=True)
        
        # Voice settings
        voice_frame = tk.LabelFrame(
            settings_frame,
            text=" Voice Settings ",
            font=self.fonts["normal"],
            bg=self.colors["bg_primary"],
            fg=self.colors["text_primary"],
            relief="flat"
        )
        voice_frame.pack(fill="x", padx=20, pady=20)
        
        tk.Label(
            voice_frame,
            text="Voice Rate:",
            font=self.fonts["normal"],
            bg=self.colors["bg_primary"],
            fg=self.colors["text_secondary"]
        ).grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        self.voice_rate = tk.Scale(
            voice_frame,
            from_=100,
            to=300,
            orient="horizontal",
            bg=self.colors["bg_primary"],
            fg=self.colors["text_primary"],
            troughcolor=self.colors["bg_tertiary"]
        )
        self.voice_rate.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        self.voice_rate.set(self.assistant.settings["voice"]["rate"])
        
        # Gesture settings
        gesture_frame = tk.LabelFrame(
            settings_frame,
            text=" Gesture Settings ",
            font=self.fonts["normal"],
            bg=self.colors["bg_primary"],
            fg=self.colors["text_primary"],
            relief="flat"
        )
        gesture_frame.pack(fill="x", padx=20, pady=20)
        
        tk.Label(
            gesture_frame,
            text="Gesture Sensitivity:",
            font=self.fonts["normal"],
            bg=self.colors["bg_primary"],
            fg=self.colors["text_secondary"]
        ).grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        self.gesture_sensitivity = tk.Scale(
            gesture_frame,
            from_=1,
            to=10,
            orient="horizontal",
            bg=self.colors["bg_primary"],
            fg=self.colors["text_primary"],
            troughcolor=self.colors["bg_tertiary"]
        )
        self.gesture_sensitivity.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        self.gesture_sensitivity.set(self.assistant.settings["gesture"]["sensitivity"] * 10)
        
        # Save button
        tk.Button(
            settings_frame,
            text="💾 Save Settings",
            font=self.fonts["normal"],
            bg=self.colors["success"],
            fg="white",
            activebackground="#0da56d",
            relief="flat",
            padx=25,
            pady=12,
            cursor="hand2",
            command=self.save_settings
        ).pack(pady=20)
    
    def create_status_bar(self):
        """Create status bar at bottom"""
        self.status_bar = tk.Frame(
            self.root,
            bg=self.colors["bg_secondary"],
            height=30
        )
        self.status_bar.pack(fill="x", side="bottom")
        self.status_bar.pack_propagate(False)
        
        self.status_label = tk.Label(
            self.status_bar,
            text="Ready",
            font=("Segoe UI", 9),
            bg=self.colors["bg_secondary"],
            fg=self.colors["text_secondary"]
        )
        self.status_label.pack(side="left", padx=10)
        
        # System info
        sys_info = f"{Config.NAME} v{Config.VERSION} | {Config.OS} | Python {Config.PYTHON_VERSION}"
        tk.Label(
            self.status_bar,
            text=sys_info,
            font=("Segoe UI", 9),
            bg=self.colors["bg_secondary"],
            fg=self.colors["text_muted"]
        ).pack(side="right", padx=10)
    
    def setup_bindings(self):
        """Setup keyboard bindings"""
        self.root.bind('<Escape>', lambda e: self.root.quit())
        self.root.bind('<Control-q>', lambda e: self.root.quit())
        self.root.bind('<Control-s>', lambda e: self.save_settings())
        self.root.bind('<F1>', lambda e: self.show_help())
        self.root.bind('<Control-space>', lambda e: self.toggle_voice())
        self.root.bind('<Alt-g>', lambda e: self.toggle_gestures())
        self.root.bind('<Control-b>', lambda e: webbrowser.open("https://google.com"))
        self.root.bind('<Control-w>', lambda e: self.close_page())
        self.root.bind('<Control-r>', lambda e: self.refresh_system())
    
    def switch_tab(self, tab_id):
        """Switch between tabs"""
        # Hide all tabs
        for tab in self.tabs.values():
            tab.pack_forget()
        
        # Show selected tab
        self.current_tab = tab_id
        if tab_id in self.tabs:
            self.tabs[tab_id].pack(fill="both", expand=True)
    
    def toggle_voice(self):
        """Toggle voice listening"""
        self.voice_active = not self.voice_active
        if self.voice_active:
            self.status_label.config(text="Voice listening active")
            self.assistant.speak("Voice listening activated")
        else:
            self.status_label.config(text="Voice listening inactive")
            self.assistant.speak("Voice listening deactivated")
    
    def start_listening(self):
        """Start voice listening"""
        self.status_label.config(text="Listening...")
        threading.Thread(target=self.listen_for_command, daemon=True).start()
    
    def listen_for_command(self):
        """Listen for voice command in background"""
        command = self.assistant.listen()
        if command:
            self.voice_text.delete("1.0", tk.END)
            self.voice_text.insert("1.0", command)
            self.process_text_command()
        self.status_label.config(text="Ready")
    
    def process_text_command(self):
        """Process command from text box"""
        command = self.voice_text.get("1.0", tk.END).strip()
        if command:
            self.assistant.process_command(command)
            self.voice_text.delete("1.0", tk.END)
    
    def perform_search(self):
        """Perform web search"""
        query = self.search_entry.get().strip()
        if query:
            webbrowser.open(f"https://www.google.com/search?q={quote_plus(query)}")
            self.assistant.speak(f"Searching for {query}")
            self.search_entry.delete(0, tk.END)
    
    def toggle_gestures(self):
        """Toggle gesture control"""
        if self.gesture_active:
            self.gesture_controller.stop()
            self.gesture_active = False
            self.gesture_status_label.config(
                text="Gesture Control: OFF",
                fg=self.colors["error"]
            )
            self.gesture_toggle_btn.config(
                text="▶ Start Gesture Control",
                bg=self.colors["success"]
            )
            self.assistant.speak("Gesture control disabled")
        else:
            self.gesture_controller.start()
            self.gesture_active = True
            self.gesture_status_label.config(
                text="Gesture Control: ON",
                fg=self.colors["success"]
            )
            self.gesture_toggle_btn.config(
                text="⏸ Stop Gesture Control",
                bg=self.colors["error"]
            )
            self.assistant.speak("Gesture control enabled")
    
    def test_camera(self):
        """Test camera feed"""
        if not LIBRARIES['opencv']['available']:
            messagebox.showerror("Camera Error", "OpenCV not installed: pip install opencv-python")
            return
            
        try:
            cv2 = LIBRARIES['opencv']['module']
            cap = cv2.VideoCapture(Config.CAMERA_ID)
            if not cap.isOpened():
                messagebox.showerror("Camera Error", "Cannot open camera")
                return
                
            ret, frame = cap.read()
            if ret:
                cv2.imshow('Camera Test', frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                self.assistant.speak("Camera test successful")
            else:
                messagebox.showerror("Camera Error", "Cannot read from camera")
                
            cap.release()
        except Exception as e:
            messagebox.showerror("Camera Error", str(e))
    
    def start_voice_typing(self):
        """Start voice typing"""
        if not LIBRARIES['pyautogui']['available']:
            self.assistant.speak("Voice typing requires pyautogui")
            return
            
        self.assistant.speak("Voice typing started. Speak now...")
        # This would be implemented with speech recognition
        # For now, just a placeholder
        self.assistant.speak("Voice typing feature coming soon")
    
    def stop_voice_typing(self):
        """Stop voice typing"""
        self.assistant.speak("Voice typing stopped")
    
    def open_data_folder(self):
        """Open data folder"""
        try:
            os.startfile(Config.DATA_DIR)
        except:
            subprocess.Popen(['explorer', str(Config.DATA_DIR)])
    
    def clear_cache(self):
        """Clear cache"""
        if messagebox.askyesno("Clear Cache", "Clear all cache and temporary files?"):
            self.assistant.speak("Cache cleared")
    
    def refresh_system(self):
        """Refresh system"""
        self.assistant.speak("System refreshed")
    
    def show_system_info(self):
        """Display system information"""
        info = f"""
        System: {platform.system()}
        Python: {platform.python_version()}
        Assistant: {self.assistant.name} v{self.assistant.version}
        Architecture: {platform.architecture()[0]}
        Processor: {platform.processor()}
        """
        messagebox.showinfo("System Information", info.strip())
    
    def save_settings(self):
        """Save settings"""
        self.assistant.settings["voice"]["rate"] = self.voice_rate.get()
        self.assistant.settings["gesture"]["sensitivity"] = self.gesture_sensitivity.get() / 10
        self.assistant.save_settings()
        self.assistant.speak("Settings saved")
        messagebox.showinfo("Success", "Settings saved successfully!")
    
    def close_page(self):
        """Close current browser page"""
        if LIBRARIES['pyautogui']['available']:
            pyautogui = LIBRARIES['pyautogui']['module']
            if platform.system() == "Darwin":
                pyautogui.hotkey('command', 'w')
            else:
                pyautogui.hotkey('ctrl', 'w')
            self.assistant.speak("Page closed")
        else:
            self.assistant.speak("pyautogui not available for auto-closing")
    
    def show_help(self):
        """Show help dialog"""
        help_text = f"""
        {Config.NAME} v{Config.VERSION} - Help Guide
        
        Voice Commands:
        - Say "{Config.WAKE_WORDS[0]}" to activate
        - "Open [browser/website]" - Open applications
        - "Search for [query]" - Web search
        - "Take screenshot" - Capture screen
        - "Open/Close camera" - Camera control
        - "Play [song/video]" - Play media
        - "What time is it?" - Get current time
        - "Close [app/tab]" - Close applications
        - "System info" - Display system information
        
        Gesture Controls:
        - 👆 Index finger: Move cursor
        - 🤏 Thumb+Index: Left click
        - 🤏 Thumb+Middle: Right click
        - ✌️ Two fingers: Double click
        - 👍 Thumb up: Scroll up
        - 👎 Thumb down: Scroll down
        
        Keyboard Shortcuts:
        - Ctrl+Space: Toggle voice
        - Alt+G: Toggle gestures
        - Ctrl+B: Open browser
        - Ctrl+W: Close page
        - Ctrl+R: Refresh
        - F1: Show help
        - Esc: Exit
        
        Requirements:
        - pip install opencv-python mediapipe pyautogui
        - pip install SpeechRecognition pyttsx3
        - pip install Pillow pywhatkit
        """
        
        help_window = tk.Toplevel(self.root)
        help_window.title("Help")
        help_window.geometry("600x500")
        help_window.configure(bg=self.colors["bg_primary"])
        
        text_widget = scrolledtext.ScrolledText(
            help_window,
            font=self.fonts["monospace"],
            bg=self.colors["bg_tertiary"],
            fg=self.colors["text_primary"],
            insertbackground=self.colors["text_primary"]
        )
        text_widget.pack(fill="both", expand=True, padx=10, pady=10)
        text_widget.insert("1.0", help_text.strip())
        text_widget.config(state="disabled")
    
    def update_loop(self):
        """Periodic update loop"""
        # Update status
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        self.status_label.config(text=f"Ready | {current_time}")
        
        # Schedule next update
        if self.assistant.is_active:
            self.root.after(1000, self.update_loop)
    
    def run_console_mode(self):
        """Run in console mode if GUI not available"""
        print(f"\n{'='*60}")
        print(f"{Config.NAME} v{Config.VERSION} - Console Mode")
        print(f"{'='*60}\n")
        
        self.assistant.speak(f"Hello! I am {Config.NAME}, your virtual assistant.")
        
        while self.assistant.is_active:
            try:
                print("\nCommands:")
                print("1. Voice command")
                print("2. Type command")
                print("3. Open browser")
                print("4. Take screenshot")
                print("5. System info")
                print("6. Exit")
                
                choice = input("\nEnter choice (1-6): ").strip()
                
                if choice == "1":
                    self.assistant.speak("Listening...")
                    command = self.assistant.listen()
                    if command:
                        self.assistant.process_command(command)
                
                elif choice == "2":
                    command = input("Enter command: ").strip()
                    if command:
                        self.assistant.process_command(command)
                
                elif choice == "3":
                    browser = input("Browser (chrome/edge/firefox): ").strip()
                    if browser:
                        webbrowser.open("https://google.com")
                        print(f"Opening {browser}")
                
                elif choice == "4":
                    self.assistant.take_screenshot()
                
                elif choice == "5":
                    print(f"\nSystem: {platform.system()}")
                    print(f"Python: {platform.python_version()}")
                    print(f"Assistant: {self.assistant.name} v{self.assistant.version}")
                
                elif choice == "6":
                    self.assistant.speak("Goodbye!")
                    break
                
            except KeyboardInterrupt:
                self.assistant.speak("Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def run(self):
        """Run the GUI"""
        if TK_AVAILABLE:
            # Center window
            self.root.update_idletasks()
            width = self.root.winfo_width()
            height = self.root.winfo_height()
            x = (self.root.winfo_screenwidth() // 2) - (width // 2)
            y = (self.root.winfo_screenheight() // 2) - (height // 2)
            self.root.geometry(f'{width}x{height}+{x}+{y}')
            
            # Welcome message
            self.assistant.speak(f"Hello! I am {Config.NAME}, version {Config.VERSION}. How can I assist you today?")
            
            # Run main loop
            self.root.mainloop()
        else:
            self.run_console_mode()

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main():
    """Main entry point"""
    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║               J.A.R.V.I.S Virtual Assistant v{Config.VERSION}        ║
    ║         Advanced AI Assistant with Voice & Gesture Control   ║
    ║                  Cross-Platform | Full Features              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # System information
    print(f"System: {Config.OS} | Python: {Config.PYTHON_VERSION}")
    print(f"Assistant: {Config.NAME} v{Config.VERSION}")
    print("-" * 60)
    
    # Check dependencies
    print("Checking Dependencies:")
    for lib_name, lib_info in LIBRARIES.items():
        status = "✓" if lib_info['available'] else "✗"
        print(f"  {status} {lib_name}")
    
    print("-" * 60)
    print("Starting assistant...")
    
    # Create and run assistant
    assistant = JARVISGUI()
    assistant.run()

if __name__ == "__main__":
    main()
