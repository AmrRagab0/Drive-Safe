import time
import os
import threading

from scoring.state_machine import DrowsinessState
from config import ALERT_COOLDOWN_MILD, ALERT_COOLDOWN_MODERATE, ALERT_COOLDOWN_SEVERE


class AlertSystem:
    """Audio and TTS alert management with cooldowns."""

    def __init__(self):
        self._pygame_available = False
        self._tts_available = False
        self._tts_busy = False
        self.sounds = {}
        self.last_alert_time = {}

        self.cooldowns = {
            DrowsinessState.MILD: ALERT_COOLDOWN_MILD,
            DrowsinessState.MODERATE: ALERT_COOLDOWN_MODERATE,
            DrowsinessState.SEVERE: ALERT_COOLDOWN_SEVERE,
        }

        self._init_audio()
        self._init_tts()

    def _init_audio(self):
        try:
            import pygame
            pygame.mixer.init()
            self._pygame_available = True

            sounds_dir = os.path.join(os.path.dirname(__file__), "sounds")
            for state, filename in [
                (DrowsinessState.MILD, "chime_soft.wav"),
                (DrowsinessState.MODERATE, "alarm_medium.wav"),
                (DrowsinessState.SEVERE, "alarm_loud.wav"),
            ]:
                path = os.path.join(sounds_dir, filename)
                if os.path.exists(path):
                    self.sounds[state] = pygame.mixer.Sound(path)
        except Exception as e:
            print(f"[AlertSystem] Audio init failed: {e}. Alerts will be text-only.")

    def _init_tts(self):
        try:
            import pyttsx3
            self.tts = pyttsx3.init()
            self.tts.setProperty("rate", 150)
            self._tts_available = True
        except Exception as e:
            print(f"[AlertSystem] TTS init failed: {e}. Voice alerts disabled.")

    def alert(self, state):
        """Trigger alert for the given drowsiness state, respecting cooldowns.

        Returns:
            True if an alert was triggered, False if in cooldown.
        """
        if state == DrowsinessState.ALERT:
            return False

        now = time.time()
        last = self.last_alert_time.get(state, 0)
        cooldown = self.cooldowns.get(state, 0)

        if now - last < cooldown:
            return False

        self.last_alert_time[state] = now

        if state == DrowsinessState.MILD:
            self._play_sound(state)
            print("[ALERT] MILD: Soft chime")

        elif state == DrowsinessState.MODERATE:
            self._play_sound(state)
            self._speak("You appear drowsy. Consider taking a break.")
            print("[ALERT] MODERATE: Warning + voice")

        elif state == DrowsinessState.SEVERE:
            self._play_sound(state, loop=True)
            self._speak("Warning! Pull over safely now!")
            print("[ALERT] SEVERE: Loud alarm + urgent voice")

        return True

    def stop_alarm(self):
        """Stop any looping alarm sound."""
        if self._pygame_available:
            import pygame
            pygame.mixer.stop()

    def _play_sound(self, state, loop=False):
        if self._pygame_available and state in self.sounds:
            if loop:
                self.sounds[state].play(-1)
            else:
                self.sounds[state].play()

    def _speak(self, text):
        """Speak text in a background thread so it doesn't block detection."""
        if self._tts_available and not self._tts_busy:
            def _run():
                self._tts_busy = True
                try:
                    self.tts.say(text)
                    self.tts.runAndWait()
                except Exception:
                    pass
                finally:
                    self._tts_busy = False

            threading.Thread(target=_run, daemon=True).start()

    def cleanup(self):
        if self._pygame_available:
            import pygame
            pygame.mixer.quit()
