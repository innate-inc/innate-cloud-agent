from src.primitives.types import Primitive
import asyncio
import math


class TurnAndMove(Primitive):
    @property
    def name(self):
        return "turn_and_move"

    def guidelines(self):
        return (
            "Use when you need to turn and then move forward. Provide:\n"
            + "1. angle: The angle to turn IN RADIANS (positive is counterclockwise, negative is clockwise)\n"
            + "2. distance: The distance to move forward after turning (in meters)\n"
            + "This is simpler than navigate_to_position when you just want to turn and move forward."
        )

    async def execute(self, angle: float, distance: float):
        # This primitive doesn't actually execute the movement itself
        # It will be converted to a navigate_to_position task in the Brain
        # Return the parameters for conversion
        return {
            "angle": angle,
            "distance": distance
        }, True
