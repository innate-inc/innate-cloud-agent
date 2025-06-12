from src.primitives.types import Primitive
import math


class TurnAndMove(Primitive):
    @property
    def name(self):
        return "turn_and_move"

    def guidelines(self):
        return (
            "Use when you need to turn and then move forward. Provide: "
            + "1. angle: The angle to turn IN DEGREES (positive is counterclockwise, negative is clockwise)"
            + "2. distance: The distance to move forward after turning (in meters)."
            + "This is only to use when asked by the user, if you want to move to something you see in the image, use navigate_in_sight instead."
        )

    async def execute(self, angle: float, distance: float):
        # This primitive doesn't actually execute the movement itself
        # It will be converted to a navigate_to_position task in the Brain
        # Return the parameters for conversion
        angle = math.radians(angle)
        print(f"Turning and moving {angle} radians and {distance} meters")
        return {"angle": angle, "distance": distance}, True
