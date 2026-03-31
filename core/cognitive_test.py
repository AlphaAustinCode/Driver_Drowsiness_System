import random
import time
import threading
import cv2
import numpy as np
import speech_recognition as sr

class CognitiveAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.is_active = False
        
        # Test State
        self.word = ""
        self.color_bgr = (255, 255, 255)
        self.expected_answer = ""
        self.result = None  # Will be 'pass', 'fail', or 'timeout'
        
        # Don't spam the driver. Wait at least 45 seconds between tests.
        self.cooldown_until = 0

        # Colors formatted in BGR for OpenCV
        self.colors = {
            "red": (0, 0, 255),
            "blue": (255, 0, 0),
            "green": (0, 255, 0),
            "yellow": (0, 255, 255)
        }

    def trigger(self) -> bool:
        """Starts the test if not currently active and cooldown has passed."""
        if self.is_active or time.time() < self.cooldown_until:
            return False

        self.is_active = True
        self.result = None

        # Generate the Stroop Test (Word and Color must be different)
        words = list(self.colors.keys())
        self.word = random.choice(words)
        
        color_choices = [c for c in words if c != self.word]
        self.expected_answer = random.choice(color_choices)
        self.color_bgr = self.colors[self.expected_answer]

        print(f"\n[COGNITIVE TEST] Triggered!")
        print(f"[COGNITIVE TEST] Word on screen: '{self.word.upper()}'")
        print(f"[COGNITIVE TEST] Driver must say: '{self.expected_answer.upper()}'")

        # Start listening in a background thread so video doesn't freeze
        threading.Thread(target=self._listen_for_answer, daemon=True).start()
        return True

    def _listen_for_answer(self):
        """Listens to the mic and evaluates the response."""
        try:
            with sr.Microphone() as source:
                # 0.5 sec calibration for car background noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                print("[COGNITIVE TEST] Listening (3 seconds)...")
                # Strict 3-second window to answer
                audio = self.recognizer.listen(source, timeout=3.0, phrase_time_limit=3.0)
            
            print("[COGNITIVE TEST] Processing audio...")
            # Using Google's free speech-to-text API
            text = self.recognizer.recognize_google(audio).lower()
            print(f"[COGNITIVE TEST] Driver said: '{text}'")

            if self.expected_answer in text:
                self.result = "pass"
                print("[COGNITIVE TEST] Result: PASSED. Applying reward.")
            else:
                self.result = "fail"
                print("[COGNITIVE TEST] Result: FAILED. Applying penalty.")

        except sr.WaitTimeoutError:
            print("[COGNITIVE TEST] Result: TIMEOUT. Driver unresponsive.")
            self.result = "timeout"
        except sr.UnknownValueError:
            print("[COGNITIVE TEST] Result: UNINTELLIGIBLE. Applying penalty.")
            self.result = "fail"
        except Exception as e:
            print(f"[COGNITIVE TEST] Mic Error: {e}")
            self.result = "error"
            
        finally:
            self.is_active = False
            self.cooldown_until = time.time() + 45.0  # 45 second cooldown

    def pop_result(self):
        """Returns the result and clears it so it's only processed once."""
        res = self.result
        self.result = None
        return res

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draws the Stroop test in the center of the screen."""
        if not self.is_active:
            return frame
        
        h, w = frame.shape[:2]
        
        # Semi-transparent dark background for readability
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h//2 - 70), (w, h//2 + 70), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
        
        cv2.putText(frame, "QUICK CHECK: SAY THE COLOR OF THIS TEXT", 
                    (w//2 - 240, h//2 - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # The trick word
        text_size = cv2.getTextSize(self.word.upper(), cv2.FONT_HERSHEY_DUPLEX, 2.0, 4)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(frame, self.word.upper(), (text_x, h//2 + 40), 
                    cv2.FONT_HERSHEY_DUPLEX, 2.0, self.color_bgr, 4)
        
        return frame