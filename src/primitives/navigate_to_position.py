from src.primitives.types import Primitive
import asyncio


class NavigateToPosition(Primitive):
    @property
    def name(self):
        return "navigate_to_position"

    def guidelines(self):
        return (
            "Use when you need to navigate the robot to the specified position using provided x, y coordinates, and theta (yaw) angle IN RADIANS. "
            + "Can be used to navigate to a specific point in the map."
        )

    async def execute(self, x: float, y: float, theta: float):
        # Replace this simulated delay and print statements with actual navigation logic.
        print(f"Initiating navigation to position: x={x}, y={y}, theta={theta}")
        await asyncio.sleep(2)  # Simulate time delay for navigation.
        print(f"Navigation complete. Arrived at position: x={x}, y={y}, theta={theta}")
        return f"Reached position ({x}, {y}, {theta})", True
