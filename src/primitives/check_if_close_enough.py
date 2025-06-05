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
    annotate_camera_view,
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


class CheckIfCloseEnough(Primitive):
    def __init__(self):
        """
        Initialize Gemini client.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                self.genai_client = genai.Client(api_key=api_key)
                print("Gemini client initialized successfully.")
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
        return "check_if_close_enough"

    def guidelines(self):
        return (
            "Use this primitive to check if the robot is close enough to a target. "
            "You need to provide a target description and a distance in meters. "
            "It will determine if the target is closer or further than the specified distance."
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
        Execute the check_if_close_enough primitive.

        Args:
            distance_meters (float): The distance in meters to check against.
            target_description (str): Description of the target to check.
            map_payload (dict): Map payload from the robot.

        Returns:
            tuple: (message, success, data)
        """
        print(
            f"CheckIfCloseEnough: Starting check from ({self.current_x}, "
            f"{self.current_y}) for target '{target_description}' at distance {distance_meters}m."
        )

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

        # Sample valid navigation points at the specified distance
        result = sample_valid_navigation_points(
            self.current_x,
            self.current_y,
            self.current_yaw,
            map_array,
            map_info,
            self.horizontal_fov,
            min_obstacle_distance=0.20,
            distances=[distance_meters],
            angles_deg=[-40, -20, 0, 20, 40],
        )

        (
            valid_points_absolute,
            valid_points_angle_distance,
            invalid_points_absolute,
            invalid_points_angle_distance,
        ) = result

        if not valid_points_absolute:
            msg = "Could not find any valid points at the specified distance."
            print(msg)
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

        annotated_image = annotate_camera_view(
            cv_image,
            camera_valid_navigation_points,
            convert_to_image_coords,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_navigation_visualizations(
            annotated_image, map_vis, timestamp, prefix="check_close_enough"
        )

        if not self.genai_client:
            msg = "Gemini client not available. Cannot perform check."
            print(msg)
            return msg, False, None

        # Create prompt for Gemini
        user_prompt = f"""
The image shows several numbered green circles which represent a distance of {distance_meters} meters from the robot.
Is the target '{target_description}' closer or further away than this distance?

Respond with whether the target is "closer" or "further".
"""

        class Proximity(Enum):
            CLOSER = "CLOSER"
            FURTHER = "FURTHER"
            UNKNOWN = "UNKNOWN"

        class ResponseSchema(BaseModel):
            proximity: Proximity
            reason: str

        try:
            print("Calling Gemini to check proximity...")

            _, img_encoded = cv2.imencode(".jpg", annotated_image)
            img_bytes = img_encoded.tobytes()

            image_part = types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
            message_parts = [user_prompt, image_part]

            response = self.genai_client.models.generate_content(
                contents=message_parts,
                model=GEMINI_MODEL_NAME,
                generation_config=types.GenerateContentConfig(
                    temperature=GEMINI_TEMPERATURE,
                    top_p=GEMINI_TOP_P,
                    top_k=GEMINI_TOP_K,
                    max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                    response_mime_type="application/json",
                    response_schema=ResponseSchema,
                ),
            )

            response_data = ResponseSchema.parse_raw(response.text)

            print(f"Gemini response: {response_data}")

            if response_data.proximity == Proximity.CLOSER:
                feedback_msg = f"The target {target_description} is close enough"
                self._send_feedback(feedback_msg)
                return feedback_msg, True, {"is_close_enough": True}
            elif response_data.proximity == Proximity.FURTHER:
                feedback_msg = f"The target {target_description} is further away than {distance_meters}m"
                self.feedback_callback(feedback_msg)
                return feedback_msg, True, {"is_close_enough": False}
            else:
                feedback_msg = f"Could not determine proximity for {target_description}. Reason: {response_data.reason}"
                self.feedback_callback(feedback_msg)
                return feedback_msg, False, None

        except Exception as e:
            error_msg = f"Error calling Gemini API: {e}"
            print(error_msg)
            return error_msg, False, None
