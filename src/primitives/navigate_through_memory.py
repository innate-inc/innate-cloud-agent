from src.primitives.types import Primitive
from src.primitives.navigate_to_position import NavigateToPosition
import os
import pickle
import time
import networkx as nx
import numpy as np
from datetime import datetime
import base64
from typing import Optional, Tuple, Dict, List, Any
import google.generativeai as genai
import json
from PIL import Image
from io import BytesIO


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
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"
        )
        self.images_dir = os.path.join(self.data_dir, "images")
        self.graphs_dir = os.path.join(self.data_dir, "pose_graphs")

        # Create directories if they don't exist
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.graphs_dir, exist_ok=True)

        # Parameters for node addition
        self.min_distance = 0.2  # Minimum distance between nodes (meters)
        self.min_angle_diff = np.radians(45)  # Minimum angle difference (radians)

        # Parameters for edge creation
        self.edge_distance_threshold = 0.8  # Maximum distance for edge creation
        self.edge_angle_threshold = np.radians(90)  # Maximum angle for edge creation

        # Initialize Gemini API if API key is available
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel(
                model_name="gemini-1.5-pro",
                generation_config={
                    "temperature": 1,
                    "top_p": 0.95,
                    "top_k": 64,
                    "max_output_tokens": 8192,
                    "response_mime_type": "application/json",
                },
            )
        else:
            self.gemini_model = None
            print("Warning: GEMINI_API_KEY not found in environment variables. VLM-based navigation will not be available.")

    def get_user_graph(self, user_token: str) -> nx.DiGraph:
        """Get the pose graph for a specific user, loading it if necessary."""
        if user_token not in self._user_graphs:
            self._user_graphs[user_token] = self._load_graph(user_token)
        return self._user_graphs[user_token]

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
            return None

        # If Gemini model is not available, fall back to most recent node
        if self.gemini_model is None:
            print("Gemini model not available. Falling back to most recent node.")
            most_recent_node = max(
                graph.nodes, key=lambda n: graph.nodes[n].get("timestamp", 0)
            )
            node_data = graph.nodes[most_recent_node]
            return (
                node_data["position"]["x"],
                node_data["position"]["y"],
                node_data["position"]["theta"],
            )

        try:
            # Create a mapping from frame numbers to node IDs
            frame_to_node_id = {}
            message_parts = []

            # Base prompt - improved to be more explicit
            base_assistant_text = (
                f"You are an AI assistant for a robot navigating through a space. "
                f"Your goal is to help the robot find specific locations based on visual appearance. "
                f"I will show you a series of images labeled as Frame 1, Frame 2, etc. "
                f"Each image shows a different colored square. "
                f"Then I will ask you to identify which frame best matches a color description. "
                f"You must respond with a JSON object containing a 'frame_number' key with the "
                f"number of the best matching frame as an integer value. "
                f"For example: {{\"frame_number\": 2}} if Frame 2 is the best match. "
                f"This is critical for the robot's navigation."
            )
            message_parts.append(base_assistant_text)
            
            print(f"DEBUG: Base prompt: {base_assistant_text}")
            print(f"DEBUG: Number of nodes in graph: {len(graph.nodes)}")

            # Add images to the prompt
            for idx, (node_id, node_data) in enumerate(graph.nodes(data=True)):
                frame_num = idx + 1
                frame_to_node_id[frame_num] = node_id
                
                # Load and encode the image
                image_path = node_data["image_path"]
                print(f"DEBUG: Processing node {node_id}, frame {frame_num}, image path: {image_path}")
                
                if os.path.exists(image_path):
                    with Image.open(image_path) as img:
                        # Print image details
                        print(f"DEBUG: Image mode: {img.mode}, size: {img.size}")
                        
                        # Convert to RGB if needed
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        
                        # Resize if too large
                        if img.width > 800 or img.height > 800:
                            img.thumbnail((800, 800))
                            print(f"DEBUG: Resized image to {img.size}")
                            
                        # Create a Gemini image part
                        buff = BytesIO()
                        img.save(buff, format="JPEG")
                        img_bytes = buff.getvalue()
                        img_part = {"mime_type": "image/jpeg", "data": img_bytes}
                        message_parts.append(img_part)
                        message_parts.append(f"Frame {frame_num}.")
                        print(f"DEBUG: Added frame {frame_num} to message parts")
                else:
                    print(f"DEBUG: Image file not found: {image_path}")

            # Final question - improved to be more explicit
            last_message = (
                f"Based on these images, which frame best matches this description: \"{description}\"? "
                f"Look carefully at the colors and visual elements in each frame. "
                f"Pay special attention to the color of each square. "
                f"Respond ONLY with a JSON object containing a 'frame_number' key with the "
                f"number of the best matching frame as an integer value. "
                f"For example: {{\"frame_number\": 2}} if Frame 2 is the best match."
            )
            message_parts.append(last_message)
            print(f"DEBUG: Final question: {last_message}")
            print(f"DEBUG: Total message parts: {len(message_parts)}")
            print(f"DEBUG: Frame to node ID mapping: {frame_to_node_id}")

            # Call Gemini model
            print(f"DEBUG: Calling Gemini model with {len(message_parts)} message parts")
            response = self.gemini_model.generate_content(message_parts)
            print(f"DEBUG: Raw Gemini response: {response.text}")
            
            # Parse the response
            try:
                response_text = response.text
                response_json = json.loads(response_text)
                frame_number = response_json.get("frame_number")
                
                print(f"DEBUG: Parsed response JSON: {response_json}")
                print(f"DEBUG: Frame number from response: {frame_number}")
                
                if frame_number and frame_number in frame_to_node_id:
                    node_id = frame_to_node_id[frame_number]
                    node_data = graph.nodes[node_id]
                    print(f"VLM selected frame {frame_number} (node {node_id}) for description: {description}")
                    return (
                        node_data["position"]["x"],
                        node_data["position"]["y"],
                        node_data["position"]["theta"],
                    )
                else:
                    print(f"Invalid frame number in VLM response: {response_text}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing VLM response: {e}. Response: {response.text}")
        
        except Exception as e:
            print(f"Error using VLM for navigation: {e}")
            import traceback
            traceback.print_exc()
        
        # Fall back to most recent node if VLM fails
        print("Falling back to most recent node.")
        most_recent_node = max(
            graph.nodes, key=lambda n: graph.nodes[n].get("timestamp", 0)
        )
        node_data = graph.nodes[most_recent_node]
        return (
            node_data["position"]["x"],
            node_data["position"]["y"],
            node_data["position"]["theta"],
        )

    def _save_image(self, user_token: str, image_data: str) -> str:
        """Save an image to disk and return the path."""
        user_dir = os.path.join(self.images_dir, user_token)
        os.makedirs(user_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}.jpg"
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
        filepath = os.path.join(self.graphs_dir, f"{user_token}.pkl")

        try:
            with open(filepath, "wb") as f:
                pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Error saving graph: {e}")

    def _load_graph(self, user_token: str) -> nx.DiGraph:
        """Load the graph from persistent storage."""
        filepath = os.path.join(self.graphs_dir, f"{user_token}.pkl")

        if not os.path.exists(filepath):
            return nx.DiGraph()  # Return empty graph if no saved graph exists

        try:
            with open(filepath, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading graph: {e}")
            return nx.DiGraph()


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
        Navigate to a location in the robot's memory that matches the description.

        Args:
            description: Text description of the location to find
            user_token: The user/robot identifier (connection_id from the Brain)

        Returns:
            Tuple of (result message, success boolean)
        """
        # Find the location in the pose graph
        location = self.pose_graph_memory.find_location_by_description(
            user_token, description
        )

        if location is None:
            return (
                f"Could not find a location matching '{description}' in memory",
                False,
            )

        x, y, theta = location

        # Use navigate_to_position to go to the found location
        navigate = NavigateToPosition()
        result, success = await navigate.execute(x, y, theta)

        if success:
            return f"Successfully navigated to location matching '{description}'", True
        else:
            return f"Failed to navigate to location matching '{description}'", False
