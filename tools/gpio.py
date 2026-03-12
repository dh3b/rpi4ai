from __future__ import annotations
from time import sleep
import RPi.GPIO as GPIO

from tools.registry import ToolRegistry, tool


def register_gpio_tools(registry: ToolRegistry) -> None:
    @tool(registry=registry)
    def move_stepper_to_angle(angle: float) -> str:
        """
        Moves a stepper motor to the specified angle in degrees.
        Value can range between 0 and 180 degrees.
        """

        if not (0 <= angle <= 180):
            return "Stepper motor was not moved. Angle must be between 0 and 180 degrees."
        duty = angle / 18 + 3
        
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(11, GPIO.OUT)

        pwm=GPIO.PWM(11, 50)
        pwm.start(duty)
        sleep(1)
        pwm.stop()
        GPIO.cleanup()

        return f"Stepper motor moved by {angle} degrees"