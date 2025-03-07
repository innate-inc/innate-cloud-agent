from src.agents.types import PrimitiveDefinition
from src.primitives.types import Primitive
from typing import List


class NavigationHandler:
    def __init__(self, logger, primitives_list: List[Primitive]):
        self.logger = logger
        self.primitives_list = primitives_list

    async def handle_navigate_in_sight(
        self, vision_output, robot_coords, base64_img, depth_payload
    ):
        nav_in_sight = next(
            (prim for prim in self.primitives_list if prim.name == "navigate_in_sight"),
            None,
        )

        nav_in_sight.update_current_vars(
            current_x=robot_coords["x"],
            current_y=robot_coords["y"],
            current_yaw=robot_coords["theta"],
            image_b64=base64_img,
            depth_payload=depth_payload,
        )

        msg, result, navigation_command = await nav_in_sight.execute(
            **vision_output.next_task.inputs
        )

        # Only replace the output with a navigation task if the execution was successful
        if result:
            # Replace the output with a navigation_to_position primitive.
            navigation_to_position_task = PrimitiveDefinition(
                name="navigate_to_position",
                inputs={
                    "x": navigation_command["x"],
                    "y": navigation_command["y"],
                    "theta": navigation_command["theta"],
                },
            )
            vision_output.next_task = navigation_to_position_task

            self.logger.info(
                f"Converted navigate_in_sight to navigate_to_position with inputs: {navigation_to_position_task.inputs}. Initial coords were: {robot_coords}"
            )
        else:
            # If the execution failed, update the vision output to reflect the failure
            vision_output.stop_current_task = True
            vision_output.observation = f"Navigation in sight failed: {msg}"
            vision_output.next_task = None
            vision_output.to_tell_user = f"I couldn't navigate to the shelf: {msg}"

        return vision_output

    async def handle_navigate_through_memory(
        self, vision_output, connection_id
    ):
        # Find the NavigateThroughMemory primitive in the primitives_list
        navigate_through_memory = next(
            (prim for prim in self.primitives_list if prim.name == "navigate_through_memory"),
            None,
        )
        
        if navigate_through_memory is None:
            self.logger.error("NavigateThroughMemory primitive not found")
            vision_output.stop_current_task = True
            vision_output.observation = "Navigation through memory failed: primitive not found"
            vision_output.next_task = None
            vision_output.to_tell_user = "I couldn't navigate to that location: internal error"
            return vision_output
            
        # Execute the primitive to get navigation parameters
        description = vision_output.next_task.inputs.get("description", "")
        result, success, navigation_command = await navigate_through_memory.execute(
            description, connection_id
        )
        
        if success and navigation_command:
            # Replace the output with a navigate_to_position primitive
            navigation_to_position_task = PrimitiveDefinition(
                name="navigate_to_position",
                inputs=navigation_command,
            )
            vision_output.next_task = navigation_to_position_task
            
            self.logger.info(
                f"Converted navigate_through_memory to navigate_to_position with inputs: {navigation_command}"
            )
        else:
            # If the execution failed, update the vision output to reflect the failure
            vision_output.stop_current_task = True
            vision_output.observation = f"Navigation through memory failed: {result}"
            vision_output.next_task = None
            vision_output.to_tell_user = f"I couldn't navigate to that location: {result}"
            
        return vision_output
