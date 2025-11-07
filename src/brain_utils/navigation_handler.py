import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.agents.types import PrimitiveDefinition
from src.constants_robots import ROBOT_PARAMS_TO_USE
from src.primitives.types import Primitive
from typing import List
from math import atan, radians, tan, degrees
import math
from src.brain_utils.payload_decoders import decode_map_payload


SIM_VERTICAL_FOV = ROBOT_PARAMS_TO_USE["vertical_fov"]
SIM_CAMERA_RESOLUTION = ROBOT_PARAMS_TO_USE["camera_resolution"]
SIM_HORIZONTAL_FOV = degrees(
    2
    * atan(
        tan(radians(SIM_VERTICAL_FOV) / 2)
        * SIM_CAMERA_RESOLUTION[0]
        / SIM_CAMERA_RESOLUTION[1]
    )
)

MAURICE_OAK_D_VERTICAL_FOV = ROBOT_PARAMS_TO_USE["vertical_fov"]
MAURICE_OAK_D_HORIZONTAL_FOV = ROBOT_PARAMS_TO_USE["horizontal_fov"]

# Minimum distance (in meters) that the target position must be from obstacles
MIN_OBSTACLE_DISTANCE = ROBOT_PARAMS_TO_USE["min_obstacle_distance"]
ENABLE_VISUALIZATIONS = ROBOT_PARAMS_TO_USE["enable_visualizations"]


class NavigationHandler:
    def __init__(self, logger, primitives_list: List[Primitive]):
        self.logger = logger
        self.primitives_list = primitives_list

    async def handle_check_distance_and_orientation(
        self,
        vision_output,
        robot_coords,
        base64_img,
        depth_payload,
        map_payload,
        camera_info,
    ):
        has_canceled_task = False
        check_prim = next(
            (
                prim
                for prim in self.primitives_list
                if prim.name == "check_distance_and_orientation"
            ),
            None,
        )

        if not check_prim:
            self.logger.error("CheckDistanceAndOrientation primitive not found")
            vision_output.observation = (
                "CheckDistanceAndOrientation primitive not found, cannot perform check."
            )
            vision_output.next_task = None
            has_canceled_task = True
            return vision_output, has_canceled_task

        # Camera info is now always required from the payload
        horizontal_fov = camera_info["horizontal_fov"]
        vertical_fov = camera_info["vertical_fov"]

        check_prim.update_current_vars(
            current_x=robot_coords["x"],
            current_y=robot_coords["y"],
            current_yaw=robot_coords["theta"],
            image_b64=base64_img,
            depth_payload=depth_payload,
            horizontal_fov=horizontal_fov,
            vertical_fov=vertical_fov,
            camera_info=camera_info,
        )

        distance_meters = vision_output.next_task.inputs.get("distance_meters")
        target_description = vision_output.next_task.inputs.get("target_description")

        if distance_meters is None or target_description is None:
            vision_output.observation = "Missing 'distance_meters' or 'target_description' for check_distance_and_orientation."
            vision_output.next_task = None
            has_canceled_task = True
            return vision_output, has_canceled_task

        await check_prim.execute(
            distance_meters=distance_meters,
            target_description=target_description,
            map_payload=map_payload,
        )

        return vision_output, has_canceled_task

    async def handle_navigate_in_sight(
        self,
        vision_output,
        robot_coords,
        base64_img,
        depth_payload,
        map_payload,
        camera_info,
    ):
        has_canceled_task = False
        nav_in_sight = next(
            (prim for prim in self.primitives_list if prim.name == "navigate_in_sight"),
            None,
        )

        # Camera info is now always required from the payload
        horizontal_fov = camera_info["horizontal_fov"]
        vertical_fov = camera_info["vertical_fov"]

        nav_in_sight.update_current_vars(
            current_x=robot_coords["x"],
            current_y=robot_coords["y"],
            current_yaw=robot_coords["theta"],
            image_b64=base64_img,
            depth_payload=depth_payload,
            horizontal_fov=horizontal_fov,
            vertical_fov=vertical_fov,
            camera_info=camera_info,
        )

        # Extract input parameters
        target_object = vision_output.next_task.inputs.get("target_object")
        target_description = vision_output.next_task.inputs.get(
            "target_description", target_object
        )
        stop_in_front_of_target = vision_output.next_task.inputs.get(
            "stop_in_front_of_target", False
        )

        # Execute the primitive with the appropriate parameters
        msg, result, navigation_command = await nav_in_sight.execute(
            stop_in_front_of_target=stop_in_front_of_target,
            target_description=target_description,
            map_payload=map_payload,
        )

        # Only replace the output with a navigation task if the execution was successful
        if result and navigation_command is not None:
            # Check if the target position is too close to obstacles using the map
            if map_payload:
                # If we're using point selection, we already verified safety
                is_safe, safety_msg = self.check_position_safety(
                    navigation_command["x"], navigation_command["y"], map_payload
                )

                if not is_safe:
                    self.logger.warn(
                        f"Navigation (in sight) target at ({navigation_command['x']}, {navigation_command['y']}) "
                        f"is too close to obstacles: {safety_msg}"
                    )
                    # If not safe, update the vision output to reflect the safety issue
                    vision_output.stop_current_task = True
                    vision_output.observation = f"Navigation failed: {safety_msg}"
                    vision_output.next_task = None
                    vision_output.to_tell_user = (
                        f"I can't navigate to that position because it's too close to obstacles. "
                        f"{safety_msg}"
                    )
                    return vision_output

            # Get the primitive ID from the original task if it exists
            original_primitive_id = getattr(
                vision_output.next_task, "primitive_id", None
            )

            # Replace the output with a navigation_to_position primitive.
            navigation_to_position_task = PrimitiveDefinition(
                name="navigate_to_position",
                inputs={
                    "x": navigation_command["x"],
                    "y": navigation_command["y"],
                    "theta": navigation_command["theta"],
                },
                primitive_id=original_primitive_id,  # Preserve the ID
            )
            vision_output.next_task = navigation_to_position_task

            self.logger.info(
                f"Converted navigate_in_sight to navigate_to_position with inputs: "
                f"{navigation_to_position_task.inputs}. Initial coords were: {robot_coords}"
            )
        elif result and navigation_command is None:
            vision_output.stop_current_task = True
            vision_output.observation = (
                f"Navigation indicates we're already close enough to the target: {msg}"
            )
            vision_output.anticipation = None
            vision_output.next_task = None
            vision_output.to_tell_user = None
            print(
                f"Navigation in sight indicates we're already close enough to the target and we return this vision output: {vision_output}"
            )
        else:
            has_canceled_task = True
            # If the execution failed, update the vision output to reflect the failure
            vision_output.stop_current_task = True
            vision_output.observation = f"Navigation in sight failed: {msg}"
            vision_output.anticipation = f"I should use a different primitive to navigate, maybe turning and moving."
            vision_output.next_task = None
            print(
                f"Navigation in sight failed and we return this vision output: {vision_output}"
            )

        return vision_output, has_canceled_task

    def check_position_safety(self, target_x, target_y, map_payload):
        """
        Check if a target position is at a safe distance from obstacles.

        Args:
            target_x (float): Target X coordinate in world frame
            target_y (float): Target Y coordinate in world frame
            map_payload (dict): Map payload containing occupancy grid data

        Returns:
            tuple: (is_safe, message) where is_safe is a boolean and message is a string
        """
        try:
            # Decode the map payload
            map_array, map_info = decode_map_payload(map_payload)

            # Get map metadata
            resolution = map_info["resolution"]
            origin_x = map_info["origin_x"]
            origin_y = map_info["origin_y"]

            # Convert target position from world coordinates to grid coordinates
            grid_x = int((target_x - origin_x) / resolution)
            grid_y = int((target_y - origin_y) / resolution)

            # Check if the position is within map bounds
            if (
                grid_x < 0
                or grid_x >= map_info["width"]
                or grid_y < 0
                or grid_y >= map_info["height"]
            ):
                return False, "Target position is outside the map boundaries."

            # Calculate the radius in grid cells that corresponds to MIN_OBSTACLE_DISTANCE
            obstacle_radius = int(MIN_OBSTACLE_DISTANCE / resolution)

            # Create visualization if enabled
            if ENABLE_VISUALIZATIONS:
                self._visualize_safety_check(
                    map_array, map_info, grid_x, grid_y, obstacle_radius
                )

            # Define a search window
            min_x = max(0, grid_x - obstacle_radius)
            max_x = min(map_info["width"] - 1, grid_x + obstacle_radius)
            min_y = max(0, grid_y - obstacle_radius)
            max_y = min(map_info["height"] - 1, grid_y + obstacle_radius)

            # Check if any cell within the radius is an obstacle (value = 100)
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    # Calculate distance from this cell to the target cell
                    cell_distance = math.sqrt((x - grid_x) ** 2 + (y - grid_y) ** 2)

                    # If this cell is within our search radius and is an obstacle
                    if cell_distance <= obstacle_radius and map_array[y, x] == 100:
                        # Calculate actual world distance
                        world_distance = cell_distance * resolution
                        return False, (
                            f"Found obstacle at {world_distance:.2f}m from target, "
                            f"which is less than the minimum safe distance of "
                            f"{MIN_OBSTACLE_DISTANCE}m."
                        )

            return True, "Position is at a safe distance from obstacles."

        except Exception as e:
            self.logger.error(f"Error checking position safety: {e}")
            # If there's an error, err on the side of caution
            return False, f"Could not verify position safety: {e}"

    def _visualize_safety_check(
        self, map_array, map_info, grid_x, grid_y, obstacle_radius
    ):
        """
        Create a visualization of the safety check for debugging.

        Args:
            map_array (numpy.ndarray): Map data
            map_info (dict): Map metadata
            grid_x (int): Target X position in grid coordinates
            grid_y (int): Target Y position in grid coordinates
            obstacle_radius (int): Search radius in grid cells
        """
        try:

            # Create a copy of the map for visualization
            # Convert occupancy grid (-1, 0, 100) to an RGB image
            # -1: Unknown (gray), 0: Free (white), 100: Occupied (black)
            rgb_map = np.zeros(
                (map_array.shape[0], map_array.shape[1], 3), dtype=np.uint8
            )

            # Unknown space (gray)
            rgb_map[map_array == -1] = [128, 128, 128]

            # Free space (white)
            rgb_map[map_array == 0] = [255, 255, 255]

            # Occupied space (black)
            rgb_map[map_array == 100] = [0, 0, 0]

            # Flip the map vertically to match the visualization in image_processor
            rgb_map = np.flipud(rgb_map)

            # Create PIL image
            vis_img = Image.fromarray(rgb_map)
            draw = ImageDraw.Draw(vis_img)

            # Draw safety radius (green circle)
            # Need to flip y coordinate for drawing since map is flipped
            flipped_y = map_info["height"] - grid_y
            draw.ellipse(
                [
                    grid_x - obstacle_radius,
                    flipped_y - obstacle_radius,
                    grid_x + obstacle_radius,
                    flipped_y + obstacle_radius,
                ],
                outline=(0, 255, 0),  # Green
                width=2,
            )

            # Draw target position (red dot)
            target_radius = 5
            draw.ellipse(
                [
                    grid_x - target_radius,
                    flipped_y - target_radius,
                    grid_x + target_radius,
                    flipped_y + target_radius,
                ],
                fill=(255, 0, 0),  # Red
                outline=(0, 0, 0),  # Black outline
            )

            # Add text
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()

            safety_text = (
                f"Target: ({grid_x}, {grid_y}), "
                f"Safety radius: {obstacle_radius} cells "
                f"({MIN_OBSTACLE_DISTANCE}m)"
            )

            draw.text((10, 10), safety_text, font=font, fill=(255, 0, 0))

            # Save visualization
            os.makedirs("safety_checks", exist_ok=True)
            vis_img.save("safety_checks/safety_check.png")

            # self.logger.info(
            #     "Saved safety check visualization to safety_checks/safety_check.png"
            # )

        except Exception as e:
            self.logger.error(f"Error creating safety visualization: {e}")
            # Don't raise - this is just for debugging

    async def handle_navigate_through_memory(
        self, vision_output, connection_id, map_payload=None
    ):
        # Find the NavigateThroughMemory primitive in the primitives_list
        navigate_through_memory = next(
            (
                prim
                for prim in self.primitives_list
                if prim.name == "navigate_through_memory"
            ),
            None,
        )

        if navigate_through_memory is None:
            self.logger.error("NavigateThroughMemory primitive not found")
            vision_output.stop_current_task = True
            vision_output.observation = (
                "Navigation through memory failed: primitive not found"
            )
            vision_output.next_task = None
            vision_output.to_tell_user = (
                "I couldn't navigate to that location: internal error"
            )
            return vision_output

        # Execute the primitive to get navigation parameters
        description = vision_output.next_task.inputs.get("description", "")
        result, success, navigation_command = await navigate_through_memory.execute(
            description, connection_id
        )

        original_primitive_id = getattr(vision_output.next_task, "primitive_id", None)

        has_canceled_task = False

        if success and navigation_command:
            # Check if target position is safe if map_payload is available
            if map_payload:
                is_safe, safety_msg = self.check_position_safety(
                    navigation_command["x"], navigation_command["y"], map_payload
                )

                if not is_safe:
                    self.logger.warn(
                        f"Navigation (through memory) target at ({navigation_command['x']}, {navigation_command['y']}) "
                        f"is too close to obstacles: {safety_msg}"
                    )
                    # If not safe, update the vision output to reflect the safety issue
                    vision_output.stop_current_task = True
                    vision_output.observation = f"Navigation failed: {safety_msg}"
                    vision_output.next_task = None
                    vision_output.to_tell_user = (
                        f"I can't navigate to that position because it's too close to obstacles. "
                        f"{safety_msg}"
                    )
                    return vision_output

            # Replace the output with a navigate_to_position primitive
            navigation_to_position_task = PrimitiveDefinition(
                name="navigate_to_position",
                inputs=navigation_command,
                primitive_id=original_primitive_id,
            )
            vision_output.next_task = navigation_to_position_task

            self.logger.info(
                f"Converted navigate_through_memory to navigate_to_position with inputs: "
                f"{navigation_command}"
            )
        else:
            has_canceled_task = True
            # If the execution failed, update the vision output to reflect the failure
            vision_output.stop_current_task = True
            vision_output.observation = f"Navigation through memory failed: {result}"
            vision_output.next_task = None
            vision_output.to_tell_user = None

        return vision_output, has_canceled_task

    async def handle_turn_and_move(self, vision_output, robot_coords, map_payload=None):
        """
        Handle the turn_and_move primitive by converting it to a navigate_to_position task.

        This takes the angle to turn and distance to move forward, and calculates the
        resulting x, y, theta coordinates for a navigate_to_position task.
        """
        # Get the angle and distance from the inputs
        angle = vision_output.next_task.inputs.get("angle", 0.0)
        angle = radians(angle)
        distance = vision_output.next_task.inputs.get("distance", 0.0)

        # Get current robot coordinates
        current_x = robot_coords.get("x", 0.0)
        current_y = robot_coords.get("y", 0.0)
        current_theta = robot_coords.get("theta", 0.0)

        # Calculate the new theta (current + angle to turn)
        new_theta = current_theta + angle

        # Calculate the new x, y coordinates after moving forward
        # Using trigonometry: x = current_x + distance * cos(new_theta)
        #                     y = current_y + distance * sin(new_theta)
        new_x = current_x + distance * math.cos(new_theta)
        new_y = current_y + distance * math.sin(new_theta)

        has_canceled_task = False

        # Check if target position is safe if map_payload is available
        if map_payload:
            is_safe, safety_msg = self.check_position_safety(new_x, new_y, map_payload)

            is_safe = True

            if not is_safe:
                self.logger.warn(
                    f"Navigation (turn and move) target at ({new_x:.2f}, {new_y:.2f}) is too close to obstacles: "
                    f"{safety_msg}"
                )
                # If not safe, update the vision output to reflect the safety issue
                vision_output.stop_current_task = True
                vision_output.observation = f"Navigation failed: {safety_msg}. "
                vision_output.thoughts = f"I can't turn and move to that position because it's too close to obstacles."
                vision_output.next_task = None
                vision_output.to_tell_user = None
                return vision_output, True

        original_primitive_id = getattr(vision_output.next_task, "primitive_id", None)

        # Create a navigate_to_position task with the calculated coordinates
        navigation_to_position_task = PrimitiveDefinition(
            name="navigate_to_position",
            inputs={
                "x": new_x,
                "y": new_y,
                "theta": new_theta,
            },
            primitive_id=original_primitive_id,
        )

        # Update the vision output
        vision_output.next_task = navigation_to_position_task

        self.logger.info(
            f"Converted turn_and_move (angle={angle}, distance={distance}) to "
            f"navigate_to_position with inputs: x={new_x}, y={new_y}, theta={new_theta}. "
            f"Initial coords were: {robot_coords}"
        )

        return vision_output, has_canceled_task
