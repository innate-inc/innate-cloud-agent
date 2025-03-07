from src.primitives.types import Primitive
import asyncio


class NavigateToPosition(Primitive):
    @property
    def name(self):
        return "navigate_to_position"

    def guidelines(self):
        return (
            "Use when you need to navigate the robot to a position. Two modes are available:\n"
            + "1. Absolute mode: Navigate to the specified position using provided x, y coordinates, and theta (yaw) angle IN RADIANS.\n"
            + "2. Delta mode: Move relative to current position by the specified delta_x, delta_y, and delta_theta (in RADIANS).\n"
            + "Set is_delta=True to use delta mode, or is_delta=False (default) for absolute mode."
        )

    async def execute(self, x: float, y: float, theta: float, is_delta: bool = False):
        # Replace this simulated delay and print statements with actual navigation logic.
        if is_delta:
            print(f"Initiating relative movement: delta_x={x}, delta_y={y}, delta_theta={theta}")
            await asyncio.sleep(2)  # Simulate time delay for navigation.
            print(f"Relative movement complete. Moved by: delta_x={x}, delta_y={y}, delta_theta={theta}")
            return f"Completed relative movement by ({x}, {y}, {theta})", True
        else:
            print(f"Initiating navigation to absolute position: x={x}, y={y}, theta={theta}")
            await asyncio.sleep(2)  # Simulate time delay for navigation.
            print(f"Navigation complete. Arrived at position: x={x}, y={y}, theta={theta}")
            return f"Reached absolute position ({x}, {y}, {theta})", True
