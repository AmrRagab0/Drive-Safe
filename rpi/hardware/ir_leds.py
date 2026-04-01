from config import IR_LED_PIN, IS_RPI


class IRLEDController:
    """GPIO control for IR LEDs. No-op on non-RPi platforms."""

    def __init__(self):
        self._available = False
        if IS_RPI:
            try:
                import RPi.GPIO as GPIO
                self.GPIO = GPIO
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(IR_LED_PIN, GPIO.OUT)
                self._available = True
            except Exception as e:
                print(f"[IRLEDs] GPIO init failed: {e}")
        else:
            print("[IRLEDs] Not on RPi — IR LED control disabled.")

    def on(self):
        if self._available:
            self.GPIO.output(IR_LED_PIN, self.GPIO.HIGH)

    def off(self):
        if self._available:
            self.GPIO.output(IR_LED_PIN, self.GPIO.LOW)

    def cleanup(self):
        if self._available:
            self.off()
            self.GPIO.cleanup(IR_LED_PIN)
