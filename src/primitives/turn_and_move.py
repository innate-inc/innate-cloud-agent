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
            + "1. angle: The angle to turn IN DEGREES. **IMPORTANT** positive is counterclockwise, negative is clockwise."
            + "2. distance : The distance to move forward after turning (in meters)."
        )

    async def execute(self, angle: float, distance: float):
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
