from datetime import datetime
from src.primitives.types import Primitive
import asyncio
import base64
import cv2
import numpy as np
from google import genai
from google.genai import types
import os
from pydantic import BaseModel
from enum import Enum
from typing import Optional, Callable, Awaitable

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

from src.brain_utils.payload_decoders import decode_map_payload
from src.constants_robots import ROBOT_PARAMS_TO_USE

ROBOT_CAMERA_INFO = ROBOT_PARAMS_TO_USE["camera_info"]
MIN_OBSTACLE_DISTANCE = ROBOT_PARAMS_TO_USE["min_obstacle_distance"]
ENABLE_VISUALIZATIONS = ROBOT_PARAMS_TO_USE["enable_visualizations"]

GEMINI_MODEL_NAME = "gemini-3-flash-preview"
GEMINI_TEMPERATURE = 0
GEMINI_TOP_P = 0.95
GEMINI_TOP_K = 64
GEMINI_MAX_OUTPUT_TOKENS = 8192
THINKING_BUDGET = 0


class ContinuousNavigationStatus(Enum):
    CONTINUE = "CONTINUE"
    OBJECTIVE_REACHED = "OBJECTIVE_REACHED"
    CANNOT_PROCEED = "CANNOT_PROCEED"
    EXPLORING = "EXPLORING"


class ContinuousNavigationResponse(BaseModel):
    status: ContinuousNavigationStatus
    point_id: int
    explanation: str
    progress_description: str


class NavInsightContinuous(Primitive):
    """
    A continuous navigation primitive that keeps receiving images and making
    navigation decisions until the objective is achieved.

    When activated:
    1. Receives images continuously from the client (every ~1 second)
    2. Compares current image with previous image to track progress
    3. Makes navigation decisions without going through the normal VLM pipeline
    4. Stops when it determines the objective has been achieved or cannot proceed
    """

    def __init__(self):
        super().__init__()
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                self.genai_client = genai.Client(api_key=api_key)
                print("NavInsightContinuous: Gemini client initialized.")
            except Exception as e:
                self.genai_client = None
                print(f"NavInsightContinuous: Failed to initialize Gemini client: {e}")
        else:
            self.genai_client = None
            print("NavInsightContinuous: GEMINI_API_KEY not found.")

        # State for continuous navigation
        self._is_active = False
        self._objective: Optional[str] = None
        self._stop_in_front_of_target: bool = True
        self._previous_image: Optional[np.ndarray] = None
        self._previous_annotated_image: Optional[np.ndarray] = None
        self._previous_decision: Optional[ContinuousNavigationResponse] = None
        self._iteration_count: int = 0
        self._max_iterations: int = 50

        # Callbacks for communication with Brain
        self._send_navigation_callback: Optional[Callable] = None
        self._request_image_callback: Optional[Callable] = None
        self._on_complete_callback: Optional[Callable] = None

        # Current state variables (updated each iteration)
        self.current_x: float = 0.0
        self.current_y: float = 0.0
        self.current_yaw: float = 0.0
        self.image_b64: str = ""
        self.depth_payload: dict = {}
        self.horizontal_fov: float = 0.0
        self.vertical_fov: float = 0.0
        self.pitch_deg: float = 0.0
        self.x_cam: float = 0.0
        self.height_cam: float = 0.0

    @property
    def name(self):
        return "nav_insight_continuous"

    def guidelines(self):
        return (
            "AUTONOMOUS continuous navigation - use this instead of navigate_in_sight when "
            "the target is FAR AWAY or NOT YET VISIBLE and requires sustained navigation.\n\n"
            "KEY DIFFERENCE: navigate_in_sight makes ONE move then returns control to you. "
            "nav_insight_continuous keeps moving autonomously until it reaches the target "
            "or determines it cannot proceed.\n\n"
            "USE THIS WHEN:\n"
            "- Target is far (end of hallway, another room, across a large space)\n"
            "- You want the robot to autonomously navigate without your step-by-step guidance\n"
            "- Following a corridor or path to its end\n"
            "- Searching/exploring an area\n\n"
            "DO NOT USE when target is close or you need precise positioning - use navigate_in_sight instead.\n\n"
            "Provide target_description (e.g., 'the exit door at the end of the hallway')."
        )

    def guidelines_when_running(self):
        return (
            "The robot is currently in continuous navigation mode, actively processing "
            "images and making navigation decisions. Do not interrupt unless necessary. "
            "The primitive will automatically stop when the objective is reached or "
            "if it cannot proceed further."
        )

    @property
    def is_active(self) -> bool:
        return self._is_active

    def set_callbacks(
        self,
        send_navigation: Callable[[dict], Awaitable[None]],
        request_image: Callable[[], Awaitable[None]],
        on_complete: Callable[[str, bool], Awaitable[None]],
    ):
        """
        Set the callbacks for communication with the Brain.

        Args:
            send_navigation: Callback to send navigation commands
            request_image: Callback to request the next image from client
            on_complete: Callback when navigation is complete (message, success)
        """
        self._send_navigation_callback = send_navigation
        self._request_image_callback = request_image
        self._on_complete_callback = on_complete

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
        """Update the current state variables for this iteration."""
        self.current_x = current_x
        self.current_y = current_y
        self.current_yaw = current_yaw
        self.image_b64 = image_b64
        self.depth_payload = depth_payload
        self.horizontal_fov = horizontal_fov
        self.vertical_fov = vertical_fov
        self.pitch_deg = camera_info["pitch_deg"]
        self.x_cam = camera_info["x_cam"]
        self.height_cam = camera_info["height_cam"]

    def _decode_image(self) -> tuple[Optional[np.ndarray], int, int]:
        """Decode the base64 image."""
        try:
            image_bytes = base64.b64decode(self.image_b64)
            image_array = np.frombuffer(image_bytes, np.uint8)
            cv_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if cv_image is None:
                return None, 0, 0
            height, width = cv_image.shape[:2]
            return cv_image, width, height
        except Exception as e:
            print(f"NavInsightContinuous: Error decoding image: {e}")
            return None, 0, 0

    def _sample_navigation_points(self, map_array, map_info, angles):
        """Sample valid navigation points."""
        return sample_valid_navigation_points(
            self.current_x,
            self.current_y,
            self.current_yaw,
            map_array,
            map_info,
            self.horizontal_fov,
            min_obstacle_distance=MIN_OBSTACLE_DISTANCE * 2,
            distances=[0.5, 1.0, 2.0],
            angles_deg=angles,
        )

    def _create_point_mapping(
        self,
        valid_points_absolute,
        valid_points_angle_distance,
        map_info,
    ):
        """Create point mapping from navigation data."""
        point_mapping = {}
        grid_valid_points = []
        camera_valid_points = []

        for i, ((px, py, ptheta), (angle, distance)) in enumerate(
            zip(valid_points_absolute, valid_points_angle_distance)
        ):
            point_id = i + 1
            pixel_x, pixel_y = world_to_grid_coordinates(px, py, map_info)
            grid_valid_points.append((pixel_x, pixel_y, ptheta, point_id))
            camera_valid_points.append((angle, distance, point_id))
            point_mapping[str(point_id)] = {
                "angle_distance": (angle, distance),
                "x": px,
                "y": py,
                "theta": ptheta,
            }

        return point_mapping, grid_valid_points, camera_valid_points

    def _annotate_image(
        self,
        cv_image,
        image_width,
        image_height,
        camera_valid_points,
    ) -> np.ndarray:
        """Create annotated image with navigation points."""

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

        return annotate_camera_view(
            cv_image,
            camera_valid_points,
            convert_to_image_coords,
        )

    def _call_gemini_continuous(
        self,
        annotated_image: np.ndarray,
        previous_annotated_image: Optional[np.ndarray],
        previous_decision: Optional[ContinuousNavigationResponse],
    ) -> tuple[Optional[str], Optional[ContinuousNavigationResponse]]:
        """
        Call Gemini for continuous navigation decision, comparing current and previous images.
        """
        if not self.genai_client:
            print("NavInsightContinuous: Gemini client not available.")
            return "1", None

        # Build the prompt with context about previous decision
        previous_context = ""
        if previous_decision:
            previous_context = f"""
Previous Decision Context:
- Status: {previous_decision.status.value}
- Explanation: {previous_decision.explanation}
- Progress: {previous_decision.progress_description}
"""

        stop_instruction = ""
        if self._stop_in_front_of_target:
            stop_instruction = (
                "Make sure to stop IN FRONT of the target, not on top of it or past it. "
                "If you can see the target and there are no navigation points between you and it, "
                "return OBJECTIVE_REACHED."
            )

        user_prompt = f"""
You are a robot navigating continuously towards an objective.

OBJECTIVE: {self._objective}

{previous_context}

ITERATION: {self._iteration_count + 1} / {self._max_iterations}

The images show:
1. CURRENT VIEW - The current camera view with numbered green circles representing safe navigation points
{f"2. PREVIOUS VIEW - The view from the previous decision point (for comparison)" if previous_annotated_image is not None else ""}

Analyze the current situation and decide:
1. If the objective has been reached, return status OBJECTIVE_REACHED with point_id 0
2. If you cannot proceed (blocked, lost, etc.), return status CANNOT_PROCEED with point_id 0
3. If you're still exploring/searching, return status EXPLORING with the best point to continue
4. If you're making progress towards the objective, return status CONTINUE with the best point

{stop_instruction}

Select the numbered point (1, 2, 3, etc.) that best helps reach the objective.
Provide a brief explanation and describe any progress made since the last image.
"""

        try:
            # Prepare image parts
            _, curr_encoded = cv2.imencode(".jpg", annotated_image)
            curr_bytes = curr_encoded.tobytes()
            current_image_part = types.Part.from_bytes(
                data=curr_bytes, mime_type="image/jpeg"
            )

            message_parts = [user_prompt, current_image_part]

            # Add previous image if available
            if previous_annotated_image is not None:
                _, prev_encoded = cv2.imencode(".jpg", previous_annotated_image)
                prev_bytes = prev_encoded.tobytes()
                previous_image_part = types.Part.from_bytes(
                    data=prev_bytes, mime_type="image/jpeg"
                )
                message_parts.append(previous_image_part)

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
                    response_schema=ContinuousNavigationResponse,
                ),
            )

            response_parsed = response.parsed
            print(f"NavInsightContinuous: Gemini response: {response_parsed}")

            if response_parsed.status in [
                ContinuousNavigationStatus.OBJECTIVE_REACHED,
                ContinuousNavigationStatus.CANNOT_PROCEED,
            ]:
                return "0", response_parsed
            else:
                return str(response_parsed.point_id), response_parsed

        except Exception as e:
            print(f"NavInsightContinuous: Error calling Gemini: {e}")
            return "1", None

    async def activate(
        self,
        objective: str,
        stop_in_front_of_target: bool = True,
        max_iterations: int = 50,
    ):
        """
        Activate continuous navigation mode.

        Args:
            objective: Description of what to navigate towards
            stop_in_front_of_target: Whether to stop in front of the target
            max_iterations: Maximum number of navigation iterations
        """
        self._is_active = True
        self._objective = objective
        self._stop_in_front_of_target = stop_in_front_of_target
        self._max_iterations = max_iterations
        self._iteration_count = 0
        self._previous_image = None
        self._previous_annotated_image = None
        self._previous_decision = None

        print(f"NavInsightContinuous: Activated with objective: {objective}")
        self._send_feedback(f"Starting continuous navigation towards: {objective}")

    def deactivate(self):
        """Deactivate continuous navigation mode."""
        self._is_active = False
        self._objective = None
        self._previous_image = None
        self._previous_annotated_image = None
        self._previous_decision = None
        print("NavInsightContinuous: Deactivated")

    async def process_image(
        self, map_payload: dict
    ) -> tuple[str, bool, Optional[dict]]:
        """
        Process an incoming image during continuous navigation.

        This is called by the Brain when an image arrives while this primitive is active.

        Returns:
            tuple: (message, should_continue, navigation_command)
        """
        if not self._is_active:
            return "Continuous navigation not active", False, None

        self._iteration_count += 1

        if self._iteration_count > self._max_iterations:
            self.deactivate()
            return f"Maximum iterations ({self._max_iterations}) reached", False, None

        # Decode map
        try:
            map_array, map_info = decode_map_payload(map_payload)
        except Exception as e:
            return f"Failed to decode map: {e}", False, None

        # Decode current image
        cv_image, image_width, image_height = self._decode_image()
        if cv_image is None:
            return "Failed to decode image", False, None

        # Sample navigation points
        angles = [-40, -20, 0, 20, 40]
        (
            valid_points_absolute,
            valid_points_angle_distance,
            _,
            _,
        ) = self._sample_navigation_points(map_array, map_info, angles)

        # Create point mapping
        point_mapping, grid_valid_points, camera_valid_points = (
            self._create_point_mapping(
                valid_points_absolute, valid_points_angle_distance, map_info
            )
        )

        if not point_mapping:
            # Try alternative angles
            angles = [-50, -30, -10, 10, 30, 50]
            (
                valid_points_absolute,
                valid_points_angle_distance,
                _,
                _,
            ) = self._sample_navigation_points(map_array, map_info, angles)
            point_mapping, grid_valid_points, camera_valid_points = (
                self._create_point_mapping(
                    valid_points_absolute, valid_points_angle_distance, map_info
                )
            )

        if not point_mapping:
            self._send_feedback("No valid navigation points found, stopping.")
            self.deactivate()
            return "No valid navigation points available", False, None

        # Create annotated image
        annotated_image = self._annotate_image(
            cv_image, image_width, image_height, camera_valid_points
        )

        # Save visualization if enabled
        if ENABLE_VISUALIZATIONS:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            robot_pixel_x, robot_pixel_y = world_to_grid_coordinates(
                self.current_x, self.current_y, map_info
            )
            robot_pos = (robot_pixel_x, robot_pixel_y, self.current_yaw)
            map_vis = create_map_visualization(
                map_array, robot_pos, grid_valid_points, map_info
            )
            save_navigation_visualizations(
                annotated_image, map_vis, timestamp, prefix="nav_continuous"
            )

        # Call Gemini with current and previous images
        selected_point_id, gemini_response = self._call_gemini_continuous(
            annotated_image,
            self._previous_annotated_image,
            self._previous_decision,
        )

        # Store current as previous for next iteration
        self._previous_image = cv_image.copy()
        self._previous_annotated_image = annotated_image.copy()
        self._previous_decision = gemini_response

        # Check if we should stop
        if gemini_response:
            if gemini_response.status == ContinuousNavigationStatus.OBJECTIVE_REACHED:
                self._send_feedback(f"Objective reached: {gemini_response.explanation}")
                self.deactivate()
                return f"Objective reached: {gemini_response.explanation}", False, None

            if gemini_response.status == ContinuousNavigationStatus.CANNOT_PROCEED:
                self._send_feedback(f"Cannot proceed: {gemini_response.explanation}")
                self.deactivate()
                return f"Cannot proceed: {gemini_response.explanation}", False, None

            # Send progress feedback
            self._send_feedback(
                f"[{self._iteration_count}/{self._max_iterations}] "
                f"{gemini_response.progress_description}"
            )

        # Get navigation command for selected point
        if selected_point_id in point_mapping:
            selected_point = point_mapping[selected_point_id]
            angle, distance = selected_point["angle_distance"]

            x = distance * np.cos(-angle)
            y = distance * np.sin(-angle)
            theta = -angle

            navigation_command = {
                "x": x,
                "y": y,
                "theta": theta,
                "local_frame": True,
            }

            return (
                f"Navigating to point {selected_point_id}",
                True,
                navigation_command,
            )
        else:
            # Fallback to first point
            first_point_id = list(point_mapping.keys())[0]
            selected_point = point_mapping[first_point_id]
            angle, distance = selected_point["angle_distance"]

            x = distance * np.cos(-angle)
            y = distance * np.sin(-angle)
            theta = -angle

            navigation_command = {
                "x": x,
                "y": y,
                "theta": theta,
                "local_frame": True,
            }

            return (
                f"Navigating to fallback point {first_point_id}",
                True,
                navigation_command,
            )

    async def execute(
        self,
        target_description: str = None,
        stop_in_front_of_target: bool = True,
        max_iterations: int = 50,
        map_payload: dict = None,
    ):
        """
        Execute the continuous navigation primitive.

        This method activates the continuous navigation mode. After this, the primitive
        expects to receive continuous image updates via process_image().

        Args:
            target_description: Description of where to navigate
            stop_in_front_of_target: Whether to stop in front of the target
            max_iterations: Maximum number of navigation iterations
            map_payload: Initial map payload (optional, for first iteration)

        Returns:
            tuple: (message, success, navigation_command or None)
        """
        print(
            f"NavInsightContinuous: Starting from ({self.current_x}, {self.current_y})"
        )

        # Activate continuous mode
        await self.activate(
            objective=target_description,
            stop_in_front_of_target=stop_in_front_of_target,
            max_iterations=max_iterations,
        )

        # Process the first image if map_payload is provided
        if map_payload:
            msg, should_continue, nav_command = await self.process_image(map_payload)
            if not should_continue:
                self.deactivate()
                return msg, not should_continue, None
            return msg, True, nav_command
        else:
            return (
                "Continuous navigation activated. Waiting for first image.",
                True,
                None,
            )
