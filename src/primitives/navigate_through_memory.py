from src.primitives.types import Primitive
import os
import pickle
import time
import networkx as nx
import numpy as np
from datetime import datetime
import base64
from typing import Optional, Tuple
from google import genai
from google.genai import types
import json
from PIL import Image
from io import BytesIO
from pydantic import BaseModel
from enum import Enum
from src.constants_robots import ROBOT_PARAMS_TO_USE

# Default values for PoseGraphMemory parameters
DEFAULT_MIN_DISTANCE = 0.5  # meters
DEFAULT_MIN_ANGLE_DEGREES = ROBOT_PARAMS_TO_USE["horizontal_fov"] * (
    100 / 120
)  # degrees
DEFAULT_EDGE_DISTANCE_THRESHOLD = 0.8  # meters
DEFAULT_EDGE_ANGLE_THRESHOLD_DEGREES = ROBOT_PARAMS_TO_USE["horizontal_fov"] * (
    100 / 120
)  # degrees

# File system constants
DATA_DIR_NAME = "data"
IMAGES_DIR_NAME = "images"
GRAPHS_DIR_NAME = "pose_graphs"
GRAPH_FILE_EXTENSION = ".pkl"
IMAGE_FILE_EXTENSION = ".jpg"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S_%f"

# Gemini API constants


GEMINI_MODEL_NAME = "gemini-flash-latest"
GEMINI_TEMPERATURE = 0
GEMINI_TOP_P = 0.95
GEMINI_TOP_K = 64
GEMINI_MAX_OUTPUT_TOKENS = 8192
GEMINI_RESPONSE_MIME_TYPE = "application/json"

# Image processing constants
MAX_IMAGE_DIMENSION = 800
IMAGE_MODE_RGB = "RGB"
IMAGE_FORMAT_JPEG = "JPEG"


class LocationSearchReason(Enum):
    FOUND_MATCHING_LOCATION = "FOUND_MATCHING_LOCATION"
    NO_MATCHING_LOCATION = "NO_MATCHING_LOCATION"
    MEMORY_EMPTY = "MEMORY_EMPTY"
    OTHER = "OTHER"


class LocationSearchResponse(BaseModel):
    found_location: bool
    frame_number: int  # 0 if no location found
    reason: LocationSearchReason
    explanation: str


class PoseGraphMemory:
    """
    A class to store and manage the robot's spatial memory as a pose graph.
    This is a singleton class that maintains the memory across different instances.
    """

    _instance = None
    _user_graphs = {}  # Map user tokens to their graphs

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PoseGraphMemory, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the pose graph memory system."""
        self.data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), DATA_DIR_NAME
        )
        self.images_dir = os.path.join(self.data_dir, IMAGES_DIR_NAME)
        self.graphs_dir = os.path.join(self.data_dir, GRAPHS_DIR_NAME)

        # Create directories if they don't exist
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.graphs_dir, exist_ok=True)

        # Parameters for node addition
        self.min_distance = (
            DEFAULT_MIN_DISTANCE  # Minimum distance between nodes (meters)
        )
        self.min_angle_diff = np.radians(
            DEFAULT_MIN_ANGLE_DEGREES
        )  # Minimum angle difference (radians)

        # Parameters for edge creation
        self.edge_distance_threshold = DEFAULT_EDGE_DISTANCE_THRESHOLD
        self.edge_angle_threshold = np.radians(
            DEFAULT_EDGE_ANGLE_THRESHOLD_DEGREES
        )  # Maximum angle for edge creation

        # Initialize Gemini API if API key is available
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            self.genai_client = genai.Client(api_key=api_key)

        else:
            self.genai_client = None
            print(
                "Warning: GEMINI_API_KEY not found in environment variables. "
                "VLM-based navigation will not be available."
            )

    def get_user_graph(self, user_token: str) -> nx.DiGraph:
        """Get the pose graph for a specific user, loading it if necessary."""
        if user_token not in self._user_graphs:
            self._user_graphs[user_token] = self._load_graph(user_token)
        return self._user_graphs[user_token]

    def should_add_node(
        self, user_token: str, x: float, y: float, theta: float
    ) -> bool:
        """
        Determine if a new node should be added to the graph based on distance and angle criteria.

        Args:
            user_token: The user/robot identifier
            x, y, theta: Current position and orientation

        Returns:
            Boolean indicating whether a new node should be added
        """
        graph = self.get_user_graph(user_token)

        # If the graph is empty, always add the first node
        if not graph.nodes:
            return True

        # Get the most recent node
        try:
            most_recent_node = max(
                graph.nodes, key=lambda n: graph.nodes[n].get("timestamp", 0)
            )
            last_node_data = graph.nodes[most_recent_node]
        except ValueError:
            # If there's an issue finding the most recent node, add a new one
            return True

        # Get the last position
        last_position = np.array(
            [
                last_node_data["position"]["x"],
                last_node_data["position"]["y"],
                0,  # Z coordinate (not used)
            ]
        )

        # Get the current position
        current_position = np.array([x, y, 0])

        # Check distance condition
        distance = np.linalg.norm(current_position - last_position)
        if distance > self.min_distance:
            return True

        # Check orientation condition
        last_theta = last_node_data["position"]["theta"]
        angle_diff = abs(theta - last_theta)
        angle_diff = min(angle_diff, 2 * np.pi - angle_diff)  # Handle wrap-around

        if angle_diff > self.min_angle_diff:
            return True

        # If neither condition is met, don't add a new node
        return False

    def add_image_to_graph(
        self, user_token: str, image_data: str, x: float, y: float, theta: float
    ) -> int:
        """
        Add an image with position data to the user's pose graph.

        Args:
            user_token: The user/robot identifier
            image_data: Base64 encoded image data
            x, y, theta: Position and orientation

        Returns:
            The ID of the newly added node
        """
        graph = self.get_user_graph(user_token)

        # Save the image to disk
        image_path = self._save_image(user_token, image_data)

        # Get the next node ID
        node_id = max(graph.nodes, default=0) + 1

        # Add the node to the graph
        graph.add_node(
            node_id,
            image_path=image_path,
            position={"x": x, "y": y, "theta": theta},
            timestamp=time.time(),
        )

        # Add edges to nearby nodes
        self._add_edges(graph, node_id)

        # Save the updated graph
        self._save_graph(user_token, graph)

        return node_id

    def find_location_by_description(
        self, user_token: str, description: str
    ) -> Optional[Tuple[float, float, float]]:
        """
        Find a location in the graph that matches the given description using VLM.

        Args:
            user_token: The user/robot identifier
            description: Text description of the location to find

        Returns:
            Tuple of (x, y, theta) if a match is found, None otherwise
        """
        graph = self.get_user_graph(user_token)

        if not graph.nodes:
            print(f"No locations found for user {user_token}")
            return None, None

        # If Gemini model is not available, fall back to most recent node
        if self.genai_client is None:
            raise ValueError(
                "Gemini model not available. Please set GEMINI_API_KEY in environment variables."
            )

        try:
            # Create a mapping from frame numbers to node IDs
            frame_to_node_id = {}
            message_parts = []

            # Base prompt - updated to use new response format
            base_assistant_text = (
                f"You are an AI assistant for a robot navigating through a space. "
                f"Your goal is to help the robot find specific locations based on "
                f"visual appearance. I will show you a series of images labeled as "
                f"Frame 1, Frame 2, etc. Each image shows a different view of the robot's memory. "
                f"Then I will ask you to identify which frame best matches a location "
                f"description. You must respond with a JSON object that indicates whether "
                f"you found a matching location, which frame number (if any), the reason "
                f"for your decision, and an explanation. If no matching location is found, "
                f"set found_location to false and frame_number to 0."
            )
            message_parts.append(base_assistant_text)

            # Add images to the prompt
            for idx, (node_id, node_data) in enumerate(graph.nodes(data=True)):
                frame_num = idx + 1
                frame_to_node_id[frame_num] = node_id

                # Load and encode the image
                image_path = node_data["image_path"]
                images_not_found = []

                if os.path.exists(image_path):
                    with Image.open(image_path) as img:
                        # Convert to RGB if needed
                        if img.mode != IMAGE_MODE_RGB:
                            img = img.convert(IMAGE_MODE_RGB)

                        # Resize if too large
                        if (
                            img.width > MAX_IMAGE_DIMENSION
                            or img.height > MAX_IMAGE_DIMENSION
                        ):
                            img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))

                        # Create a Gemini image part
                        # Save PIL image to bytes buffer
                        with BytesIO() as buf:
                            img.save(buf, format=IMAGE_FORMAT_JPEG)
                            image_bytes = buf.getvalue()

                        image_part = types.Part.from_bytes(
                            data=image_bytes,
                            mime_type="image/jpeg",
                        )
                        message_parts.append(image_part)
                        message_parts.append(f"Frame {frame_num}.")
                else:
                    images_not_found.append(image_path)

            if images_not_found:
                pass  # print(f"MVLA: Images not found: {images_not_found}")

            print(f"Calling MVLA with {len(message_parts) // 2} images")

            # Final question - updated to use new response format
            last_message = (
                f"Based on these images, does any frame match this description: "
                f'"{description}"? Look carefully at the visual elements in each frame. '
                f"If you find a matching location, set found_location to true and provide "
                f"the frame_number. If nothing matches the description, set found_location "
                f"to false, frame_number to 0, and explain why no location fits. "
                f"Be honest if nothing in the robot's memory matches what you're looking for."
            )
            message_parts.append(last_message)

            # Call Gemini model with structured response
            response = self.genai_client.models.generate_content(
                contents=message_parts,
                model=GEMINI_MODEL_NAME,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=1024),
                    response_mime_type="application/json",
                    response_schema=LocationSearchResponse,
                ),
            )

            # Parse the structured response
            try:
                response_parsed = response.parsed
                print(f"Location search response: {response_parsed}")

                if response_parsed.found_location and response_parsed.frame_number > 0:
                    frame_number = response_parsed.frame_number
                    if frame_number in frame_to_node_id:
                        node_id = frame_to_node_id[frame_number]
                        node_data = graph.nodes[node_id]
                        print(
                            f"Found matching location at frame {frame_number}: {response_parsed.explanation}"
                        )
                        return (
                            node_data["position"]["x"],
                            node_data["position"]["y"],
                            node_data["position"]["theta"],
                        ), response_parsed
                    else:
                        print(f"Invalid frame number in VLM response: {frame_number}")
                else:
                    print(f"No matching location found: {response_parsed.explanation}")
                    return None, response_parsed

            except Exception as e:
                print(
                    f"Error parsing structured VLM response: {e}. Response: {response}"
                )
                return None, None

        except Exception as e:
            print(f"Error using VLM for navigation: {e}")
            import traceback

            traceback.print_exc()
            return None, None

        return None, None

    def _save_image(self, user_token: str, image_data: str) -> str:
        """Save an image to disk and return the path."""
        user_dir = os.path.join(self.images_dir, user_token)
        os.makedirs(user_dir, exist_ok=True)

        timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
        filename = f"{timestamp}{IMAGE_FILE_EXTENSION}"
        filepath = os.path.join(user_dir, filename)

        # Decode base64 image and save to file
        try:
            image_bytes = base64.b64decode(image_data)
            with open(filepath, "wb") as f:
                f.write(image_bytes)
        except Exception as e:
            print(f"Error saving image: {e}")
            return ""

        return filepath

    def _add_edges(self, graph: nx.DiGraph, current_node_id: int):
        """Add edges between the current node and nearby nodes."""
        if current_node_id not in graph.nodes:
            return

        current_node = graph.nodes[current_node_id]
        current_position = np.array(
            [
                current_node["position"]["x"],
                current_node["position"]["y"],
                0,  # Z coordinate (not used)
            ]
        )
        current_theta = current_node["position"]["theta"]

        # Get the forward vector of the current node
        forward = np.array([np.cos(current_theta), np.sin(current_theta), 0])

        for node_id, node_data in graph.nodes(data=True):
            if node_id == current_node_id:
                continue

            node_position = np.array(
                [
                    node_data["position"]["x"],
                    node_data["position"]["y"],
                    0,  # Z coordinate (not used)
                ]
            )
            node_theta = node_data["position"]["theta"]

            # Calculate distance between nodes
            distance = np.linalg.norm(current_position - node_position)

            if distance <= self.edge_distance_threshold:
                # Check if the other node is in front of the current node
                direction = node_position - current_position
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)  # Normalize

                    # Calculate angle between forward vector and direction
                    dot_product = np.dot(forward, direction)
                    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

                    if angle <= self.edge_angle_threshold:
                        # Add a directed edge from current node to the visible node
                        graph.add_edge(current_node_id, node_id)

                # Check if the current node is in front of the other node
                node_forward = np.array([np.cos(node_theta), np.sin(node_theta), 0])
                reverse_direction = current_position - node_position

                if np.linalg.norm(reverse_direction) > 0:
                    reverse_direction = reverse_direction / np.linalg.norm(
                        reverse_direction
                    )

                    # Calculate angle between node's forward vector and direction
                    # to current node
                    reverse_dot_product = np.dot(node_forward, reverse_direction)
                    reverse_angle = np.arccos(np.clip(reverse_dot_product, -1.0, 1.0))

                    if reverse_angle <= self.edge_angle_threshold:
                        # Add a directed edge from the other node to the current node
                        graph.add_edge(node_id, current_node_id)

    def _save_graph(self, user_token: str, graph: nx.DiGraph):
        """Save the graph to persistent storage."""
        filepath = os.path.join(self.graphs_dir, f"{user_token}{GRAPH_FILE_EXTENSION}")

        try:
            with open(filepath, "wb") as f:
                pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Error saving graph: {e}")

    def _load_graph(self, user_token: str) -> nx.DiGraph:
        """Load the graph from persistent storage."""
        filepath = os.path.join(self.graphs_dir, f"{user_token}{GRAPH_FILE_EXTENSION}")

        if not os.path.exists(filepath):
            return nx.DiGraph()  # Return empty graph if no saved graph exists

        try:
            with open(filepath, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading graph: {e}")
            return nx.DiGraph()

    def get_all_positions(self, user_token: str) -> list:
        """
        Get all x, y, theta positions from the pose graph for a user.

        Args:
            user_token: The user/robot identifier

        Returns:
            List of dictionaries with 'x', 'y', and 'theta' keys
        """
        graph = self.get_user_graph(user_token)
        positions = []
        for node_id, node_data in graph.nodes(data=True):
            pos = node_data.get("position", {})
            positions.append(
                {
                    "x": pos.get("x", 0.0),
                    "y": pos.get("y", 0.0),
                    "theta": pos.get("theta", 0.0),
                }
            )
        return positions

    def reset_user_data(self, user_token: str):
        """Reset a user's pose graph and delete their image files."""
        # Create a new empty graph for the user
        self._user_graphs[user_token] = nx.DiGraph()

        # Save the empty graph to disk
        self._save_graph(user_token, self._user_graphs[user_token])

        # Delete all image files associated with this user
        try:
            user_images_dir = os.path.join(self.images_dir, user_token)
            if os.path.exists(user_images_dir):
                for filename in os.listdir(user_images_dir):
                    file_path = os.path.join(user_images_dir, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        print(f"Error deleting image file {file_path}: {e}")
                print(f"Cleared image directory for user {user_token}")
        except Exception as e:
            print(f"Error clearing images for user {user_token}: {e}")


class NavigateThroughMemory(Primitive):
    """
    A primitive that navigates to a location stored in the robot's memory.
    """

    def __init__(self):
        self.pose_graph_memory = PoseGraphMemory()

    @property
    def name(self):
        return "navigate_through_memory"

    def guidelines(self):
        return (
            "Use when you need to navigate to a location the robot has seen before. "
            "Provide a text description of the location, and the robot will search "
            "its memory for matching places and navigate there."
        )

    async def execute(self, description: str, user_token: str):
        """
        Find a location in the robot's memory that matches the description and
        return navigation parameters for a navigate_to_position command.

        Args:
            description: Text description of the location to find
            user_token: The user/robot identifier (connection_id from the Brain)

        Returns:
            Tuple of (result message, success boolean, navigation_command dict)
        """
        # Check if user has any memory first
        graph = self.pose_graph_memory.get_user_graph(user_token)
        if not graph.nodes:
            return (
                f"No locations stored in memory yet. The robot needs to explore and build up its memory first.",
                False,
                None,
            )

        # Find the location in the pose graph
        (location, response_parsed) = (
            self.pose_graph_memory.find_location_by_description(user_token, description)
        )

        if location is None:
            if response_parsed is not None:
                return (
                    (
                        f"No location matching '{description}' was selected. "
                        f"The reason is: {response_parsed.explanation}."
                    ),
                    False,
                    None,
                )
            else:
                return (
                    f"An unexpected error occurred while searching for the location. ",
                    False,
                    None,
                )

        x, y, theta = location

        # Create navigation command parameters
        navigation_command = {
            "x": x,
            "y": y,
            "theta": theta,
            "local_frame": False,
        }

        return (
            f"Found location matching '{description}' at "
            f"coordinates ({x:.2f}, {y:.2f}, {theta:.2f})",
            True,
            navigation_command,
        )
