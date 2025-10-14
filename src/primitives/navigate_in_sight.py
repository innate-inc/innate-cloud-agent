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
MIN_OBSTACLE_DISTANCE = ROBOT_PARAMS_TO_USE["min_obstacle_distance"]
ENABLE_VISUALIZATIONS = ROBOT_PARAMS_TO_USE["enable_visualizations"]

# Gemini API constants from navigate_through_memory.py
GEMINI_MODEL_NAME = "gemini-2.5-flash-lite-preview-06-17"
GEMINI_TEMPERATURE = 0
GEMINI_TOP_P = 0.95
GEMINI_TOP_K = 64
GEMINI_MAX_OUTPUT_TOKENS = 8192
THINKING_BUDGET = 512


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
        try:
            guidelines_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                "guidelines", 
                "navigate_in_sight_guidelines.md"
            )
            with open(guidelines_path, 'r') as f:
                content = f.read()
                # Remove the markdown header and return just the content
                return content.replace("# Navigate In Sight Guidelines\n\n", "")
        except Exception as e:
            # Fallback to hardcoded guidelines if file not found
            return (
                "To use to navigate to an object or target in sight. ONLY WHEN THE TARGET IS IN SIGHT, otherwise use turn and move." 
                "Provide a spatial_indicator that MUST be one of: 'Right of the', 'Left of the', 'front of the', 'towards the', 'under the' "
                "and a target object description.\n\n"
                "The spatial_indicator specifies the spatial relationship you want with the target object. "
                "For example, if we want to pick up an object, we want to be 'front of the' it, but not 'towards the' it.\n\n"
                "If the goal is to interact with the target, make sure you use 'front of the' as the spatial indicator.\n\n"
                "After using it, you can use it again to get closer or pursue navigation "
                "in sight if you deem it necessary. Can be very helpful to follow paths or "
                "navigate to a target that is far. When using navigate_in_sight, to explore and navigate, do not use it to navigate to a precise target or you will end up stuck if its too close."
                "It is also naturally used in conjunction with the 'turn_and_move' "
                "primitive to turn and potentially look around if the target is not "
                "immediately visible but you know it might be around. NEVER USE VAGUE TARGETS LIKE: further into the corridor"
                "**IMPORTANT** Always remember that when using this primitive, you are describing where you want to move to another agent that does not have your context and thought, and that can only see the current image, so be as descriptive and clear as possible about the target object based on the image, or it might make the wrong decision."        
            )

    def point_selection_few_shot_examples(self):
        """
        Provide few-shot examples for point selection within navigate_in_sight.
        """
        return [
            {
                "image_path": "nav_in_sight_front.jpeg",
                "target_description": "front of the red bike",
                "selected_point": 7,
                "reasoning": "Point 7 is the best choice because it gets the robot on a good trajectory towards the front of the red bike without getting too close. It provides a clear path that avoids obstacles while positioning the robot to approach the bike from the front as requested."
            }
        ]

    def update_current_vars(
        self,
        current_x: float,
        current_y: float,
        current_yaw: float,
        image_b64: str,
        depth_payload: dict,
        horizontal_fov: float,
        vertical_fov: float,
        camera_info: dict,
    ):
        self.current_x = current_x
        self.current_y = current_y
        self.current_yaw = current_yaw
        self.image_b64 = image_b64
        self.depth_payload = depth_payload
        self.horizontal_fov = horizontal_fov
        self.vertical_fov = vertical_fov

        # Camera info is now always required from the payload
        self.pitch_deg = camera_info["pitch_deg"]
        self.x_cam = camera_info["x_cam"]
        self.height_cam = camera_info["height_cam"]

    def _decode_and_prepare_image(self):
        """
        Decode the base64 image and prepare it for processing.

        Returns:
            tuple: (cv_image, image_width, image_height, error_message)
        """
        try:
            image_bytes = base64.b64decode(self.image_b64)
            image_array = np.frombuffer(image_bytes, np.uint8)
            cv_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if cv_image is None:
                print("Failed to decode image into cv_image.")
                return None, None, None, "Failed to decode image"

            image_height, image_width = cv_image.shape[:2]
            return cv_image, image_width, image_height, None

        except Exception as e:
            print(f"Exception decoding image: {e}")
            return None, None, None, "Image decode error"

    def _sample_navigation_points_with_angles(self, map_array, map_info, angles):
        """
        Sample navigation points using the given angles.

        Args:
            map_array: The map array
            map_info: Map information
            angles: List of angles to sample

        Returns:
            tuple: (valid_points_absolute, valid_points_angle_distance, invalid_points_absolute, invalid_points_angle_distance)
        """
        return sample_valid_navigation_points(
            self.current_x,
            self.current_y,
            self.current_yaw,
            map_array,
            map_info,
            self.horizontal_fov,
            min_obstacle_distance=MIN_OBSTACLE_DISTANCE * 2,
            distances=[0.5, 0.8, 1.5],
            angles_deg=angles,
        )

    def _create_point_mapping(
        self,
        valid_navigation_points_absolute,
        valid_navigation_points_angle_distance,
        map_info,
    ):
        """
        Create point mapping and visualization points from navigation data.

        Returns:
            tuple: (point_mapping, grid_valid_points, camera_valid_points)
        """
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

        return (
            point_mapping,
            grid_valid_navigation_points,
            camera_valid_navigation_points,
        )

    def _prepare_invalid_points_for_visualization(
        self,
        invalid_navigation_points_absolute,
        invalid_navigation_points_angle_distance,
        map_info,
    ):
        """
        Prepare invalid points for visualization.

        Returns:
            tuple: (grid_invalid_points, camera_invalid_points)
        """
        grid_invalid_navigation_points = []
        camera_invalid_navigation_points = []

        for i, ((point_x, point_y, point_theta), (angle, distance)) in enumerate(
            zip(
                invalid_navigation_points_absolute,
                invalid_navigation_points_angle_distance,
            )
        ):
            # Using negative IDs for invalid points to distinguish them if needed
            point_id = -(i + 1)
            pixel_x, pixel_y = world_to_grid_coordinates(point_x, point_y, map_info)
            grid_invalid_navigation_points.append(
                (pixel_x, pixel_y, point_theta, point_id)
            )
            camera_invalid_navigation_points.append((angle, distance, point_id))

        return grid_invalid_navigation_points, camera_invalid_navigation_points

    def _create_visualizations(
        self,
        cv_image,
        image_width,
        image_height,
        map_array,
        map_info,
        grid_valid_points,
        grid_invalid_points,
        camera_valid_points,
        timestamp,
    ):
        """
        Create and save visualizations for navigation points.
        """

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

        # Always create annotated image (needed for Gemini point selection)
        annotated_image = annotate_camera_view(
            cv_image,
            camera_valid_points,
            convert_to_image_coords,
        )

        # Only create and save additional visualizations if enabled
        if ENABLE_VISUALIZATIONS:
            # Convert robot position from world coordinates to grid coordinates
            robot_pixel_x, robot_pixel_y = world_to_grid_coordinates(
                self.current_x, self.current_y, map_info
            )
            robot_pos = (robot_pixel_x, robot_pixel_y, self.current_yaw)

            # Create map visualization with grid coordinates
            map_vis = create_map_visualization(
                map_array,
                robot_pos,
                grid_valid_points,
                map_info,
                invalid_points=grid_invalid_points,
            )

            # Save visualizations
            save_navigation_visualizations(
                annotated_image, map_vis, timestamp, prefix="nav_in_sight"
            )

        return annotated_image

    def _prepare_point_selection_few_shot_examples(self):
        """
        Prepare few-shot examples content for point selection.
        
        Returns:
            str: Formatted few-shot examples text for the prompt
        """
        examples = self.point_selection_few_shot_examples()
        if not examples:
            return ""
            
        few_shot_text = "\n**Examples of good point selection:**\n\n"
        
        for example in examples:
            few_shot_text += f"Target: \"{example['target_description']}\"\n"
            few_shot_text += f"Best point chosen: {example['selected_point']}\n"
            few_shot_text += f"Why: {example['reasoning']}\n\n"
            
        return few_shot_text

    def _call_gemini_for_point_selection(
        self, target_description, stop_in_front_of_target, annotated_image
    ):
        """
        Call Gemini API to select a navigation point.

        Returns:
            tuple: (selected_point_id, gemini_response)
        """
        if stop_in_front_of_target:
            additional_prompt = (
                "Make sure you stop in front of the target, not after it or on the side of it! "
                "If there is no point between you and the target, return that you found no point, "
                "set the point_id to 0, and give the reason you didn't pick a point is because you're already close enough."
                "Do not pick a point too close to the target, as it might be too close to obstacles."
            )
        else:
            additional_prompt = ""

        # Prepare few-shot examples text
        few_shot_text = self._prepare_point_selection_few_shot_examples()

        # Create prompt for Gemini to select a navigation point
        user_prompt = f"""
I need to navigate to: {target_description}

The image shows several numbered green circles. Each circle represents a safe 
location I can navigate to.
Which numbered point should I navigate to based on the description?

{few_shot_text}

Please respond with the number of the best point (1, 2, 3, etc) if you found one.

- If no point is found and it might be because the numbers are occluding part of the image,
return NO_POINT_AVAILABLE.

- If you think you're already close enough, return ALREADY_CLOSE_ENOUGH, set the point_id to 0,
 and explain a little more the reason you didn't pick a point.

If there's a need for clarification, explain in the explanation field.

{additional_prompt}
"""

        try:
            # Check if API key is available and client was initialized
            if not self.genai_client:
                print("Warning: Gemini client not available. Using default point 1.")
                return "1", None
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

                # Call Gemini model using the client
                response = self.genai_client.models.generate_content(
                    contents=message_parts,
                    model=GEMINI_MODEL_NAME,
                    config=types.GenerateContentConfig(
                        temperature=GEMINI_TEMPERATURE,
                        top_p=GEMINI_TOP_P,
                        top_k=GEMINI_TOP_K,
                        max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=THINKING_BUDGET
                        ),
                        response_mime_type="application/json",
                        response_schema=ResponseSchema,
                    ),
                )
                response_parsed = response.parsed

                print(f"Gemini response: {response_parsed}")

                if response_parsed.found_a_point:
                    selected_point_id = str(response_parsed.point_id)
                    print(f"Extracted selected point ID: {selected_point_id}")
                    return selected_point_id, response_parsed
                else:
                    selected_point_id = "0"
                    print(
                        f"No point found in response, defaulting to point {selected_point_id}"
                    )
                    return selected_point_id, response_parsed

        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            # Default to the first point
            selected_point_id = "1"
            print(f"Error with Gemini API, defaulting to point {selected_point_id}")
            return selected_point_id, None

    def _handle_point_selection_response(
        self, selected_point_id, gemini_response, point_mapping, target_description
    ):
        """
        Handle the point selection response and create navigation command.

        Returns:
            tuple: (message, success, navigation_command)
        """
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
        elif selected_point_id == "0":
            # Handle special cases from Gemini response
            if (
                gemini_response
                and gemini_response.reason == PointSelectionReason.ALREADY_CLOSE_ENOUGH
            ):
                return (
                    f"Already close enough to target: {gemini_response.explanation}",
                    True,
                    None,  # No navigation needed
                )
            elif (
                gemini_response
                and gemini_response.reason == PointSelectionReason.NO_POINT_AVAILABLE
            ):
                return None  # Signal to retry with different angles
            else:
                return (
                    "No valid navigation points found. Use a different primitive to navigate.",
                    False,
                    None,
                )
        else:
            # If the selected point is not valid
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

    def _execute_with_angles(
        self,
        target_description,
        stop_in_front_of_target,
        map_array,
        map_info,
        cv_image,
        image_width,
        image_height,
        angles,
        attempt,
    ):
        """
        Execute navigation with a specific set of angles.

        Returns:
            tuple: (message, success, navigation_command) or None to continue with retry
        """
        if attempt > 0:
            print(f"Retrying with different angles: {angles}")

        # Sample valid navigation points using the utility function
        result = self._sample_navigation_points_with_angles(map_array, map_info, angles)

        # Unpack the tuple of absolute points and angle-distance points
        (
            valid_navigation_points_absolute,
            valid_navigation_points_angle_distance,
            invalid_navigation_points_absolute,
            invalid_navigation_points_angle_distance,
        ) = result

        # Create visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if attempt > 0:
            timestamp += "_retry"

        # Create point mapping and visualization points
        point_mapping, grid_valid_points, camera_valid_points = (
            self._create_point_mapping(
                valid_navigation_points_absolute,
                valid_navigation_points_angle_distance,
                map_info,
            )
        )

        print(f"Generated {len(point_mapping)} valid navigation points")

        # Prepare invalid points for visualization
        grid_invalid_points, camera_invalid_points = (
            self._prepare_invalid_points_for_visualization(
                invalid_navigation_points_absolute,
                invalid_navigation_points_angle_distance,
                map_info,
            )
        )

        print(f"Generated {len(grid_invalid_points)} invalid navigation points")

        # Create visualizations
        annotated_image = self._create_visualizations(
            cv_image,
            image_width,
            image_height,
            map_array,
            map_info,
            grid_valid_points,
            grid_invalid_points,
            camera_valid_points,
            timestamp,
        )

        # If no valid points are found, try the next set of angles (if not already retried)
        if not point_mapping:
            if attempt == 0:
                print(
                    "No valid navigation points found with initial angles, will retry with different angles..."
                )
                return None  # Signal to continue with retry
            else:
                print(
                    "No valid navigation points found even after retry. Visualizations saved for debugging."
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
                f"Only one valid point found, automatically selecting point {selected_point_id}"
            )
            return (
                f"Navigation to point {selected_point_id} initiated",
                True,
                navigation_command,
            )

        # Call Gemini for point selection
        selected_point_id, gemini_response = self._call_gemini_for_point_selection(
            target_description, stop_in_front_of_target, annotated_image
        )

        # Check if this is NO_POINT_AVAILABLE and we haven't retried yet
        if (
            gemini_response
            and gemini_response.reason == PointSelectionReason.NO_POINT_AVAILABLE
            and attempt == 0
        ):
            print(
                "Gemini returned NO_POINT_AVAILABLE, will retry with different angles..."
            )
            return None  # Signal to continue with retry
        elif (
            gemini_response
            and gemini_response.reason == PointSelectionReason.ALREADY_CLOSE_ENOUGH
        ):
            print("Gemini returned ALREADY_CLOSE_ENOUGH, we can just stay here.")
            return (
                f"Already close enough to target: {gemini_response.explanation}",
                True,
                None,
            )

        # Handle the point selection response
        result = self._handle_point_selection_response(
            selected_point_id, gemini_response, point_mapping, target_description
        )

        if result is None and attempt == 0:
            # This means NO_POINT_AVAILABLE was returned, try with different angles
            return None
        elif result is None and attempt > 0:
            # This means NO_POINT_AVAILABLE was returned even after retry
            return (
                f"No suitable navigation points found even after retry: {gemini_response.explanation}",
                False,
                None,
            )
        else:
            return result

    async def execute(
        self,
        spatial_indicator: str = None,
        target: str = None,
        map_payload: dict = None,
    ):
        """
        Execute the navigate_in_sight primitive using point selection.

        Args:
            spatial_indicator (str, optional): Spatial relationship (e.g., 'Right of the', 'Left of the', 'front of the', 'towards the', 'under the')
            target (str, optional): Description of the target object
            map_payload (dict, optional): Map payload from the robot

        Returns:
            tuple: (message, success, navigation_command)
        """
        
        # Validate spatial_indicator
        valid_spatial_indicators = ["Right of the", "Left of the", "front of the", "towards the", "under the"]
        if spatial_indicator not in valid_spatial_indicators:
            error_msg = f"Invalid spatial_indicator '{spatial_indicator}'. Must be one of: {', '.join(valid_spatial_indicators)}"
            return error_msg, False, None
            
        # Validate target
        if not target or not target.strip():
            error_msg = "Target description cannot be empty"
            return error_msg, False, None
            
        # Combine spatial_indicator and target into target_description
        target_description = f"{spatial_indicator} {target}"
        
        # Determine stop_in_front_of_target based on spatial_indicator
        stop_in_front_of_target = spatial_indicator == "front of the"
        
        print(
            f"NavigateInSight: Starting navigation from ({self.current_x}, {self.current_y})"
        )

        # Decode the map payload
        try:
            map_array, map_info = decode_map_payload(map_payload)
        except Exception as e:
            error_msg = f"Failed to decode map payload: {e}"
            print(error_msg)
            return error_msg, False, None

        # Decode and prepare the image
        cv_image, image_width, image_height, error_msg = (
            self._decode_and_prepare_image()
        )
        if error_msg:
            return error_msg, False, None

        # Initial angles for sampling navigation points
        initial_angles = [-40, -20, 0, 20, 40]
        retry_angles = [-50, -30, -10, 10, 30, 50]  # Alternative angles for retry

        # Try with initial angles first, then retry with different angles if needed
        for attempt, angles in enumerate([initial_angles, retry_angles]):
            result = self._execute_with_angles(
                target_description,
                stop_in_front_of_target,
                map_array,
                map_info,
                cv_image,
                image_width,
                image_height,
                angles,
                attempt,
            )

            if result is not None:
                return result

        # This should not be reached, but just in case
        return (
            "Navigation failed after all attempts.",
            False,
            None,
        )
