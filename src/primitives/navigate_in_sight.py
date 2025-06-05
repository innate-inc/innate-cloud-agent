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

# Utility to decode depth payload (assumed defined in src/utils.py)
from src.utils import decode_map_payload
from src.constants_robots import ROBOT_PARAMS_TO_USE

ROBOT_CAMERA_INFO = ROBOT_PARAMS_TO_USE["camera_info"]

# Gemini API constants from navigate_through_memory.py
GEMINI_MODEL_NAME = "gemini-2.5-flash-preview-05-20"
GEMINI_TEMPERATURE = 0
GEMINI_TOP_P = 0.95
GEMINI_TOP_K = 64
GEMINI_MAX_OUTPUT_TOKENS = 8192


class NavigateInSight(Primitive):
    def __init__(self):
        """
        Initialize Gemini client.
        """
        # Initialize Gemini API client
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                # Attempt to create the client, which might also handle configuration.
                # If genai.configure is still needed, it should be called before client instantiation.
                # However, the example in navigate_through_memory.py suggests client takes api_key directly.
                # genai.configure(api_key=api_key) # Assuming configure is not needed if client takes key
                self.genai_client = genai.Client(api_key=api_key)
                print("Gemini client initialized successfully.")
            except Exception as e:
                self.genai_client = None
                print(f"Failed to initialize Gemini client: {e}")
        else:
            self.genai_client = None
            print(
                "Warning: GEMINI_API_KEY not found in environment variables. "
                "Point selection with VLM will not be available."
            )

    @property
    def name(self):
        return "navigate_in_sight"

    def guidelines(self):
        return (
            "To use to navigate to an object or target in sight. Is a much better "
            "primitive than navigate_to_position to use when it's to navigate to a "
            "target in sight. Provide a target object name, such as 'shelf', 'table', "
            "'chair', etc.\n\n"
            "Make sure you precise if it's on the target, or in front of it, or behind it, or to the left or right of it."
            "For example, if we want to pick up an object, we want to in front of it, but not on it.\n\n"
            "After using it, you can use it again to get closer or pursue navigation "
            "in sight if you deem it necessary. Can be very helpful to follow paths or "
            "navigate to a target that is far."
            "It is also naturally used in conjunction with the 'turn_and_move' "
            "primitive to turn and potentially look around if the target is not "
            "immediately visible but you know it might be around."
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
        target_description: str = None,
        map_payload: dict = None,
    ):
        """
        Execute the navigate_in_sight primitive using point selection.

        Args:
            target_description (str, optional): Description of where to navigate
            map_payload (dict, optional): Map payload from the robot

        Returns:
            tuple: (message, success, navigation_command)
        """
        print(
            f"NavigateInSight: Starting navigation from ({self.current_x}, "
            f"{self.current_y})"
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
                print("Failed to decode image into cv_image.")
                return "Failed to decode image", False, None
        except Exception as e:
            print(f"Exception decoding image: {e}")
            return "Image decode error", False, None

        # Sample valid navigation points using the utility function
        result = sample_valid_navigation_points(
            self.current_x,
            self.current_y,
            self.current_yaw,
            map_array,
            map_info,
            self.horizontal_fov,
            min_obstacle_distance=0.20,
            distances=[0.5, 1.0, 2.0],
            angles_deg=[-40, -20, 0, 20, 40],
        )

        # Unpack the tuple of absolute points and angle-distance points
        (
            valid_navigation_points_absolute,
            valid_navigation_points_angle_distance,
            invalid_navigation_points_absolute,
            invalid_navigation_points_angle_distance,
        ) = result

        # Create visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Convert robot position from world coordinates to grid coordinates
        robot_pixel_x, robot_pixel_y = world_to_grid_coordinates(
            self.current_x, self.current_y, map_info
        )
        robot_pos = (robot_pixel_x, robot_pixel_y, self.current_yaw)

        # Create point mapping from valid navigation points
        # This ensures the same point ID is used in both camera and map views
        point_mapping = {}
        grid_valid_navigation_points = []
        camera_valid_navigation_points = []

        # Process and number the valid points consistently for both visualizations
        for i, ((point_x, point_y, point_theta), (angle, distance)) in enumerate(
            zip(
                valid_navigation_points_absolute, valid_navigation_points_angle_distance
            )
        ):
            point_id = i + 1

            # Convert to grid coordinates for map visualization
            pixel_x, pixel_y = world_to_grid_coordinates(point_x, point_y, map_info)
            grid_valid_navigation_points.append(
                (pixel_x, pixel_y, point_theta, point_id)
            )
            camera_valid_navigation_points.append((angle, distance, point_id))

            # Store in point mapping for navigation command creation
            point_mapping[str(point_id)] = {
                "angle_distance": (angle, distance),
                "x": point_x,
                "y": point_y,
                "theta": point_theta,
            }

        print(f"Generated {len(point_mapping)} valid navigation points")

        # Prepare invalid points for visualization
        grid_invalid_navigation_points = []
        camera_invalid_navigation_points = []

        for i, ((point_x, point_y, point_theta), (angle, distance)) in enumerate(
            zip(
                invalid_navigation_points_absolute,
                invalid_navigation_points_angle_distance,
            )
        ):
            # Using negative IDs for invalid points to distinguish them if needed,
            # though visualization utils might just use color.
            point_id = -(i + 1)
            pixel_x, pixel_y = world_to_grid_coordinates(point_x, point_y, map_info)
            grid_invalid_navigation_points.append(
                (pixel_x, pixel_y, point_theta, point_id)
            )
            camera_invalid_navigation_points.append((angle, distance, point_id))

        print(
            f"Generated {len(grid_invalid_navigation_points)} invalid navigation points"
        )

        # Create map visualization with grid coordinates
        map_vis = create_map_visualization(
            map_array,
            robot_pos,
            grid_valid_navigation_points,
            map_info,
            invalid_points=grid_invalid_navigation_points,
        )

        # Create a wrapper function for angle_distance_to_image_coordinates
        def convert_to_image_coords(angle, distance):
            return angle_distance_to_image_coordinates(
                angle,
                distance,
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

        # Save visualizations
        save_navigation_visualizations(annotated_image, map_vis, timestamp)

        # If no valid points are found, exit early but still save visualizations for debugging
        if not point_mapping:
            print(
                "No valid navigation points found. "
                "Visualizations saved for debugging."
            )
            return (
                "Could not find any valid navigation points, try again with a different primitive.",
                False,
                None,
            )

        # If there's only one valid point, just use that
        if len(point_mapping) == 1:
            selected_point_id = list(point_mapping.keys())[0]
            selected_point = point_mapping[selected_point_id]

            navigation_command = {
                "x": selected_point["x"],
                "y": selected_point["y"],
                "theta": selected_point["theta"],
            }

            print(
                f"Only one valid point found, automatically selecting point "
                f"{selected_point_id}"
            )
            return (
                f"Navigation to point {selected_point_id} initiated",
                True,
                navigation_command,
            )

        # Create prompt for Gemini to select a navigation point
        user_prompt = f"""
I need to navigate to: {target_description}

The image shows several numbered green circles. Each circle represents a safe 
location I can navigate to.
Which numbered point should I navigate to based on the description?

Please respond with the number of the best point (1, 2, 3, etc) if you found one.

If no point is found, or if you think you're already close enough, return that you
found no point, set the point_id to 0, and give the reason you didn't pick a point.

If there's a need for clarification, explain in the explanation field.
"""

        # Use the GenerativeAI package directly, like in navigate_through_memory.py
        try:
            # Check if API key is available and client was initialized
            if not self.genai_client:
                print("Warning: Gemini client not available. " "Using default point 1.")
                selected_point_id = "1"
            else:
                print("Calling Gemini to select a navigation point...")

                # Convert CV2 image to JPEG bytes
                _, img_encoded = cv2.imencode(".jpg", annotated_image)
                img_bytes = img_encoded.tobytes()

                # Create image part for Gemini
                image_part = types.Part.from_bytes(
                    data=img_bytes,
                    mime_type="image/jpeg",
                )

                # Create content parts: user prompt and the image part
                message_parts = [user_prompt, image_part]

                class PointSelectionReason(Enum):
                    FOUND_MATCHING_POINT = "FOUND_MATCHING_POINT"
                    NO_POINT_AVAILABLE = "NO_POINT_AVAILABLE"
                    ALREADY_CLOSE_ENOUGH = "ALREADY_CLOSE_ENOUGH"
                    OTHER = "OTHER"

                class ResponseSchema(BaseModel):
                    found_a_point: bool
                    point_id: int
                    reason: PointSelectionReason
                    explanation: str

                # Call Gemini model using the client
                response = self.genai_client.models.generate_content(
                    contents=message_parts,
                    model=GEMINI_MODEL_NAME,
                    config=types.GenerateContentConfig(
                        temperature=GEMINI_TEMPERATURE,
                        top_p=GEMINI_TOP_P,
                        top_k=GEMINI_TOP_K,
                        max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                        response_mime_type="application/json",
                        response_schema=ResponseSchema,
                    ),
                )
                response_text = response.text

                print(f"Gemini response: {response_text}")

                # Extract the number from the response
                import re

                numbers = re.findall(r"\\d+", response_text)
                if numbers:
                    selected_point_id = numbers[0]
                    print(f"Extracted selected point ID: {selected_point_id}")
                else:
                    # Default to the first point if no number found
                    selected_point_id = "1"
                    print(
                        f"No point number found in response, defaulting to point "
                        f"{selected_point_id}"
                    )
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            # Default to the first point
            selected_point_id = "1"
            print(f"Error with Gemini API, defaulting to point {selected_point_id}")

        # Get the selected point
        if selected_point_id in point_mapping:
            selected_point = point_mapping[selected_point_id]

            navigation_command = {
                "x": selected_point["x"],
                "y": selected_point["y"],
                "theta": selected_point["theta"],
            }

            print(
                f"Selected navigation point {selected_point_id}: {navigation_command}"
            )
            return (
                f"Navigation to point {selected_point_id} initiated",
                True,
                navigation_command,
            )
        else:
            # If the selected point is not valid, use the first available one
            if point_mapping:
                return (
                    "Picked an invalid point, try again with a valid point",
                    False,
                    None,
                )
            else:
                return (
                    "No valid navigation points found. Use a different primitive to navigate.",
                    False,
                    None,
                )
