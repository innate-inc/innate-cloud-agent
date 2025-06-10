from datetime import datetime
from src.primitives.types import Primitive
import base64
import cv2
import numpy as np
from google import genai
from google.genai import types
import os
from pydantic import BaseModel
from enum import Enum

# Import utility modules
from src.primitives.visualization_utils import (
    annotate_camera_view_with_line,
    annotate_camera_view_with_corridors,
    create_map_visualization,
    save_navigation_visualizations,
)
from src.primitives.projection_utils import (
    angle_distance_to_image_coordinates,
    sample_valid_navigation_points,
    world_to_grid_coordinates,
)

# Utility to decode depth payload
from src.utils import decode_map_payload
from src.constants_robots import ROBOT_PARAMS_TO_USE

ROBOT_CAMERA_INFO = ROBOT_PARAMS_TO_USE["camera_info"]

# Gemini API constants
GEMINI_MODEL_NAME = "gemini-2.5-flash-preview-05-20"
GEMINI_TEMPERATURE = 0
GEMINI_TOP_P = 0.95
GEMINI_TOP_K = 64
GEMINI_MAX_OUTPUT_TOKENS = 8192

CORRIDOR_WIDTH = 30.0  # degrees


class Proximity(Enum):
    CLOSER = "CLOSER"
    FURTHER = "FURTHER"
    UNKNOWN = "UNKNOWN"


class OrientationResult(Enum):
    FACING = "FACING"
    NOT_FACING = "NOT_FACING"
    UNKNOWN = "UNKNOWN"


class ResponseSchema(BaseModel):
    proximity: Proximity
    reason: str


class OrientationResponseSchema(BaseModel):
    orientation: OrientationResult
    corridor_angle: int  # Mean angle of the corridor where target is located
    reason: str


class CheckDistanceAndOrientation(Primitive):
    def __init__(self):
        """
        Initialize Gemini client.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                self.genai_client = genai.Client(api_key=api_key)
            except Exception as e:
                self.genai_client = None
                print(f"Failed to initialize Gemini client: {e}")
        else:
            self.genai_client = None
            print(
                "Warning: GEMINI_API_KEY not found in environment variables. "
                "VLM capabilities will not be available."
            )

    @property
    def name(self):
        return "check_distance_and_orientation"

    def guidelines(self):
        return (
            f"Use this primitive to check if the robot is close enough to a target and if it is facing the target. "
            "You need to provide a target description and a distance in meters. Be specific about the target."
        )

    def update_current_vars(
        self,
        current_x: float,
        current_y: float,
        current_yaw: float,
        image_b64: str,
        depth_payload: dict,
        horizontal_fov: float,
        vertical_fov: float,
    ):
        self.current_x = current_x
        self.current_y = current_y
        self.current_yaw = current_yaw
        self.image_b64 = image_b64
        self.depth_payload = depth_payload
        self.horizontal_fov = horizontal_fov
        self.vertical_fov = vertical_fov
        self.pitch_deg = ROBOT_CAMERA_INFO["pitch_deg"]
        self.x_cam = ROBOT_CAMERA_INFO["x_cam"]
        self.height_cam = ROBOT_CAMERA_INFO["height_cam"]

    async def execute(
        self,
        distance_meters: float,
        target_description: str,
        map_payload: dict,
    ):
        """
        Execute the check_distance_and_orientation primitive.

        Args:
            distance_meters (float): The distance in meters to check against.
            target_description (str): Description of the target to check.
            map_payload (dict): Map payload from the robot.

        Returns:
            tuple: (message, success, data)
        """

        # Decode the map payload
        try:
            map_array, map_info = decode_map_payload(map_payload)
        except Exception as e:
            error_msg = f"Failed to decode map payload: {e}"
            print(error_msg)
            return error_msg, False, None

        # Decode the provided image from base64 into a cv2 image.
        try:
            image_bytes = base64.b64decode(self.image_b64)
            image_array = np.frombuffer(image_bytes, np.uint8)
            cv_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            image_height, image_width = cv_image.shape[:2]

            if cv_image is None:
                return "Failed to decode image", False, None
        except Exception as e:
            print(f"Exception decoding image: {e}")
            return "Image decode error", False, None

        # Sample valid navigation points at the specified distance for distance check
        result = sample_valid_navigation_points(
            self.current_x,
            self.current_y,
            self.current_yaw,
            map_array,
            map_info,
            self.horizontal_fov,
            min_obstacle_distance=0.0,
            distances=[distance_meters],
            angles_deg=np.linspace(-40, 41, 15).tolist(),
            check_map_location_valid=False,
        )

        (
            valid_points_absolute,
            valid_points_angle_distance,
            invalid_points_absolute,
            invalid_points_angle_distance,
        ) = result

        if not valid_points_absolute:
            msg = "Could not find any valid points at the specified distance."
            return msg, False, None

        # Create visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Convert robot position from world coordinates to grid coordinates
        robot_pixel_x, robot_pixel_y = world_to_grid_coordinates(
            self.current_x, self.current_y, map_info
        )
        robot_pos = (robot_pixel_x, robot_pixel_y, self.current_yaw)

        # Process and number the valid points consistently for both visualizations
        grid_valid_navigation_points = []
        camera_valid_navigation_points = []
        for i, ((point_x, point_y, point_theta), (angle, dist)) in enumerate(
            zip(valid_points_absolute, valid_points_angle_distance)
        ):
            point_id = i + 1

            # Convert to grid coordinates for map visualization
            pixel_x, pixel_y = world_to_grid_coordinates(point_x, point_y, map_info)
            grid_valid_navigation_points.append(
                (pixel_x, pixel_y, point_theta, point_id)
            )
            camera_valid_navigation_points.append((angle, dist, point_id))

        grid_invalid_navigation_points = []
        for i, ((point_x, point_y, point_theta), (angle, distance)) in enumerate(
            zip(
                invalid_points_absolute,
                invalid_points_angle_distance,
            )
        ):
            point_id = -(i + 1)
            pixel_x, pixel_y = world_to_grid_coordinates(point_x, point_y, map_info)
            grid_invalid_navigation_points.append(
                (pixel_x, pixel_y, point_theta, point_id)
            )

        # Create map visualization with grid coordinates
        map_vis = create_map_visualization(
            map_array,
            robot_pos,
            grid_valid_navigation_points,
            map_info,
            invalid_points=grid_invalid_navigation_points,
        )

        # Create a wrapper for angle_distance_to_image_coordinates
        def convert_to_image_coords(angle, dist):
            return angle_distance_to_image_coordinates(
                angle,
                dist,
                {
                    "width": image_width,
                    "height": image_height,
                    "horizontal_fov": self.horizontal_fov,
                    "vertical_fov": self.vertical_fov,
                    "pitch_deg": self.pitch_deg,
                    "x_cam": self.x_cam,
                    "height_cam": self.height_cam,
                },
            )

        # Create distance line annotation
        distance_annotated_image = annotate_camera_view_with_line(
            cv_image,
            camera_valid_navigation_points,
            convert_to_image_coords,
        )

        # Create orientation lines annotation (separate image)
        # Use corridor-based approach with 20-degree wide corridors
        orientation_annotated_image, corridors = annotate_camera_view_with_corridors(
            cv_image,  # Start with original image
            self.horizontal_fov,  # Pass the horizontal FOV
            convert_to_image_coords,
            {
                "width": image_width,
                "height": image_height,
                "horizontal_fov": self.horizontal_fov,
                "vertical_fov": self.vertical_fov,
                "pitch_deg": self.pitch_deg,
                "x_cam": self.x_cam,
                "height_cam": self.height_cam,
            },
            corridor_width=CORRIDOR_WIDTH,
        )

        # Log corridor information for debugging
        corridor_info = []
        for start_angle, end_angle in corridors:
            mean_angle = (start_angle + end_angle) / 2.0
            corridor_info.append(f"{mean_angle:+.0f}°")

        # Save both visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save distance image
        os.makedirs("navigation_visualizations", exist_ok=True)
        distance_path = f"navigation_visualizations/distance_check_{timestamp}.jpg"
        cv2.imwrite(distance_path, distance_annotated_image)

        # Save orientation image
        orientation_path = f"navigation_visualizations/corridor_check_{timestamp}.jpg"
        cv2.imwrite(orientation_path, orientation_annotated_image)

        # Also save map visualization
        save_navigation_visualizations(
            distance_annotated_image,  # Use distance image for combined save
            map_vis,
            timestamp,
            prefix="check_distance_corridors",
        )

        if not self.genai_client:
            msg = "Gemini client not available. Cannot perform check."
            return msg, False, None

        # Create prompts for Gemini - distance and orientation checks
        distance_prompt = f"""
The image shows a green line on the ground which represents a distance of {distance_meters} meters from the robot.
Is the target '{target_description}' closer or further away than this distance?
If it is under the line, it is closer. If it is above the line, it is further.

Respond with whether the target is "closer" or "further".
"""

        orientation_prompt = f"""
The image shows vertical lines that divide the view into corridors, each 20 degrees wide and centered on the robot's forward direction (0°).
Each corridor is labeled with its mean angle at the bottom.

Look at the target '{target_description}' and identify which corridor it appears in.
- If the target is in the center corridor (labeled 0°), the robot is "facing" the target.
- If the target is in any other corridor, the robot is "not_facing" the target.

Respond with:
1. Whether the robot is "facing" or "not_facing" the target
2. The mean angle of the corridor where the target appears (use the label shown at the bottom of that corridor)

Example: If the target is in the corridor labeled "+20°", respond with corridor_angle: 20
"""

        try:
            # Encode both images
            _, distance_img_encoded = cv2.imencode(".jpg", distance_annotated_image)
            distance_img_bytes = distance_img_encoded.tobytes()

            _, orientation_img_encoded = cv2.imencode(
                ".jpg", orientation_annotated_image
            )
            orientation_img_bytes = orientation_img_encoded.tobytes()

            distance_image_part = types.Part.from_bytes(
                data=distance_img_bytes, mime_type="image/jpeg"
            )
            orientation_image_part = types.Part.from_bytes(
                data=orientation_img_bytes, mime_type="image/jpeg"
            )

            # Create distance check message
            distance_message_parts = [distance_prompt, distance_image_part]

            # Create orientation check message
            orientation_message_parts = [orientation_prompt, orientation_image_part]

            # Make parallel calls to Gemini for both checks
            import asyncio

            async def distance_check():
                return self.genai_client.models.generate_content(
                    contents=distance_message_parts,
                    model=GEMINI_MODEL_NAME,
                    config=types.GenerateContentConfig(
                        temperature=GEMINI_TEMPERATURE,
                        top_p=GEMINI_TOP_P,
                        top_k=GEMINI_TOP_K,
                        max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                        thinking_config=types.ThinkingConfig(thinking_budget=256),
                        response_mime_type="application/json",
                        response_schema=ResponseSchema,
                    ),
                )

            async def orientation_check():
                return self.genai_client.models.generate_content(
                    contents=orientation_message_parts,
                    model=GEMINI_MODEL_NAME,
                    config=types.GenerateContentConfig(
                        temperature=GEMINI_TEMPERATURE,
                        top_p=GEMINI_TOP_P,
                        top_k=GEMINI_TOP_K,
                        max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                        thinking_config=types.ThinkingConfig(thinking_budget=256),
                        response_mime_type="application/json",
                        response_schema=OrientationResponseSchema,
                    ),
                )

            # Run both checks in parallel
            distance_response, orientation_response = await asyncio.gather(
                distance_check(), orientation_check()
            )

            distance_data = ResponseSchema.parse_raw(distance_response.text)
            orientation_data = OrientationResponseSchema.parse_raw(
                orientation_response.text
            )

            print(f"Gemini distance response: {distance_data}")
            print(f"Gemini orientation response: {orientation_data}")

            # Determine results
            is_close_enough = distance_data.proximity == Proximity.CLOSER
            is_facing = orientation_data.orientation == OrientationResult.FACING
            corridor_angle = orientation_data.corridor_angle

            # Calculate turn recommendation
            turn_recommendation = ""
            if not is_facing and corridor_angle != 0:
                turn_direction = "left" if corridor_angle < 0 else "right"
                turn_amount = abs(corridor_angle)
                turn_recommendation = f" The robot should turn {turn_direction} by approximately {turn_amount:.0f}° to face the target."

            # Create combined feedback message
            distance_status = (
                "close enough"
                if is_close_enough
                else f"further away than {distance_meters}m"
            )

            if is_facing or corridor_angle == 0:
                orientation_status = "facing the target"
            else:
                orientation_status = f"not facing the target (target is in the {corridor_angle}° corridor)"

            feedback_msg = f"The target {target_description} is {distance_status} and the robot is {orientation_status}.{turn_recommendation}"
            self._send_feedback(feedback_msg)

            # Return success if both conditions are met
            success = is_close_enough and is_facing

            return (
                feedback_msg,
                success,
                {
                    "is_close_enough": is_close_enough,
                    "is_facing": is_facing,
                    "corridor_angle": corridor_angle,
                    "turn_recommendation": turn_recommendation.strip(),
                    "distance_reason": distance_data.reason,
                    "orientation_reason": orientation_data.reason,
                },
            )

        except Exception as e:
            error_msg = f"Error calling Gemini API: {e}"
            print(error_msg)
            return error_msg, False, None
