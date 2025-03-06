from src.primitives.types import Primitive
from src.primitives.navigate_to_position import NavigateToPosition
import os
import pickle
import time
import networkx as nx
import numpy as np
from datetime import datetime
import base64
from typing import Optional, Tuple


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
        Find a location in the graph that matches the given description.

        Args:
            user_token: The user/robot identifier
            description: Text description of the location to find

        Returns:
            Tuple of (x, y, theta) if a match is found, None otherwise
        """
        graph = self.get_user_graph(user_token)

        if not graph.nodes:
            return None

        # For now, we'll use a simple approach - just return the most recent node
        # In a real implementation, this would use semantic search
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
            user_token: The user/robot identifier

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
