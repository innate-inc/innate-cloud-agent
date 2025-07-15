from src.primitives.types import Primitive
from src.brain_utils.unified_logger import unified_logger, LogLevel, LogSource
import math


class TurnAndMove(Primitive):
    @property
    def name(self):
        return "turn_and_move"

    def guidelines(self):
        return (
            "Use when you need to turn and/or move forward. Provide: "
            + "1. angle (optional): The angle to turn IN DEGREES (positive is counterclockwise, negative is clockwise). Defaults to 0 (no turn)."
            + "2. distance (optional): The distance to move forward after turning (in meters). Defaults to 0 (no movement)."
            + "This is a good primitive to explore, look around, and navigate precisely. Avoid big angles."
            +"When arriving at a new location, you might want to turn 60 degrees to the right, then 120 degrees to the left to look to your right and to your left to explore. "
        )

    async def execute(self, angle: float = 0.0, distance: float = 0.0):
        # This primitive doesn't actually execute the movement itself
        # It will be converted to a navigate_to_position task in the Brain
        # Return the parameters for conversion
        
        angle_rad = math.radians(angle)
        
        # Log to unified logger
        unified_logger.info(
            LogSource.PRIMITIVE,
            "turn_and_move",
            f"Executing turn_and_move: angle={angle}°, distance={distance}m",
            data={
                "angle_degrees": angle,
                "angle_radians": angle_rad,
                "distance_meters": distance,
            },
        )
        
        print(f"Turning and moving {angle_rad} radians and {distance} meters")
        return {"angle": angle_rad, "distance": distance}, True
