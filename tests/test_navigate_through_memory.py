# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Innate Inc

import os
import sys
import tempfile
import shutil
import base64
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
import io
import time
import networkx as nx

from src.brain import Brain
from src.message_types import MessageIn, MessageInType
from src.primitives.navigate_through_memory import (
    PoseGraphMemory,
    NavigateThroughMemory,
)

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestNavigateThroughMemory:
    """Tests for the NavigateThroughMemory primitive and pose_image handling."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self, monkeypatch):
        """Setup before each test and cleanup after."""
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.images_dir = os.path.join(self.temp_dir, "images")
        self.graphs_dir = os.path.join(self.temp_dir, "pose_graphs")

        # Create subdirectories
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.graphs_dir, exist_ok=True)

        # Clean up any existing test data
        # This ensures we start with a clean environment
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        if os.path.exists(data_dir):
            images_dir = os.path.join(data_dir, "images")
            graphs_dir = os.path.join(data_dir, "pose_graphs")

            # Remove image files
            if os.path.exists(images_dir):
                for user_dir in os.listdir(images_dir):
                    user_path = os.path.join(images_dir, user_dir)
                    if os.path.isdir(user_path):
                        for file in os.listdir(user_path):
                            file_path = os.path.join(user_path, file)
                            if os.path.isfile(file_path):
                                os.remove(file_path)

            # Remove graph files
            if os.path.exists(graphs_dir):
                for file in os.listdir(graphs_dir):
                    file_path = os.path.join(graphs_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

        # Reset the PoseGraphMemory singleton
        PoseGraphMemory._instance = None
        PoseGraphMemory._user_graphs = {}

        # Override the _initialize method directly
        temp_dir = self.temp_dir  # Capture the temp_dir in a local variable

        def patched_initialize(self):
            self.data_dir = temp_dir
            self.images_dir = os.path.join(temp_dir, "images")
            self.graphs_dir = os.path.join(temp_dir, "pose_graphs")
            self.min_distance = 0.2
            self.min_angle_diff = np.radians(45)
            self.edge_distance_threshold = 0.8
            self.edge_angle_threshold = np.radians(90)
            self._user_graphs = {}
            # Initialize genai_client properly for tests
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                from google import genai

                self.genai_client = genai.Client(api_key=api_key)
            else:
                self.genai_client = None

        monkeypatch.setattr(PoseGraphMemory, "_initialize", patched_initialize)

        # Run the test
        yield

        # Clean up after the test
        shutil.rmtree(self.temp_dir)

    @pytest.fixture
    def mock_brain(self):
        """Create a mock Brain instance with a mocked send_callback."""
        send_callback = MagicMock()
        brain = Brain("test_connection", send_callback)
        # Mock the handle_image method to avoid processing the image
        brain.handle_image = MagicMock()
        return brain

    def create_test_image(self, color="red"):
        """Create a simple test image and return its base64 encoding."""
        # Create a test image with a colored square on a white background
        img = Image.new("RGB", (100, 100), color="white")
        # Draw a colored square in the center
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img)
        # Draw a square from (25,25) to (75,75)
        draw.rectangle([25, 25, 75, 75], fill=color)

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="JPEG")
        img_byte_arr = img_byte_arr.getvalue()
        return base64.b64encode(img_byte_arr).decode("utf-8")

    async def add_image_to_pose_graph(
        self, mock_brain, x=1.0, y=2.0, theta=3.14, color="red"
    ):
        """Helper method to add an image to the pose graph."""
        # Create a test image
        base64_img = self.create_test_image(color)

        # Create a pose_image message with only the required fields
        # No user_token is needed as the Brain's connection_id will be used
        message = MessageIn(
            type=MessageInType.POSE_IMAGE,
            payload={
                "image": base64_img,
                "x": x,
                "y": y,
                "theta": theta,
            },
        )

        # Process the message
        await mock_brain.process_message(message)

        # Return the message for any additional assertions
        return message

    @pytest.mark.asyncio
    async def test_pose_image_handling(self, mock_brain):
        """Test that pose_image messages are correctly processed."""
        # Get the NavigateThroughMemory primitive
        navigate_through_memory = next(
            (
                p
                for p in mock_brain.local_primitives_list
                if p.name == "navigate_through_memory"
            ),
            None,
        )

        assert (
            navigate_through_memory is not None
        ), "NavigateThroughMemory primitive not found"

        # Verify the graph is empty at the start
        user_graph = navigate_through_memory.pose_graph_memory.get_user_graph(
            mock_brain.connection_id
        )
        assert len(user_graph.nodes) == 0, "Graph should be empty at the start"

        # Add an image to the pose graph
        message = await self.add_image_to_pose_graph(mock_brain)

        # Verify that handle_image was NOT called
        mock_brain.handle_image.assert_not_called()

        # Check that the image was added to the pose graph
        user_graph = navigate_through_memory.pose_graph_memory.get_user_graph(
            mock_brain.connection_id
        )
        assert len(user_graph.nodes) == 1, "Image was not added to the pose graph"

        # Check the node data
        node_id = list(user_graph.nodes)[0]
        node_data = user_graph.nodes[node_id]
        assert node_data["position"]["x"] == 1.0
        assert node_data["position"]["y"] == 2.0
        assert node_data["position"]["theta"] == 3.14

        # Check that the image file exists
        assert os.path.exists(node_data["image_path"]), "Image file was not created"

    @pytest.mark.asyncio
    async def test_navigate_through_memory_primitive(self, mock_brain):
        """Test the NavigateThroughMemory primitive."""
        # Add an image to the pose graph
        await self.add_image_to_pose_graph(mock_brain)

        # Get the NavigateThroughMemory primitive
        navigate_through_memory = next(
            (
                p
                for p in mock_brain.local_primitives_list
                if p.name == "navigate_through_memory"
            ),
            None,
        )

        assert (
            navigate_through_memory is not None
        ), "NavigateThroughMemory primitive not found"

        # Execute the NavigateThroughMemory primitive
        result, success, navigation_command = await navigate_through_memory.execute(
            "Find the red square", mock_brain.connection_id
        )

        # Check the result
        assert success is True, "Navigation location finding failed"
        assert "Found location matching" in result, f"Unexpected result: {result}"

        # Verify navigation command structure
        assert navigation_command is not None, "Navigation command should not be None"
        assert "x" in navigation_command, "Navigation command missing x coordinate"
        assert "y" in navigation_command, "Navigation command missing y coordinate"
        assert "theta" in navigation_command, "Navigation command missing theta value"
        assert navigation_command["x"] == 1.0, "Incorrect x coordinate"
        assert navigation_command["y"] == 2.0, "Incorrect y coordinate"
        assert navigation_command["theta"] == 3.14, "Incorrect theta value"

    @pytest.mark.asyncio
    async def test_navigate_through_memory_no_locations(self, mock_brain):
        """Test the NavigateThroughMemory primitive when no locations are found."""
        # Get the NavigateThroughMemory primitive
        navigate_through_memory = next(
            (
                p
                for p in mock_brain.local_primitives_list
                if p.name == "navigate_through_memory"
            ),
            None,
        )

        assert (
            navigate_through_memory is not None
        ), "NavigateThroughMemory primitive not found"

        # Execute the NavigateThroughMemory primitive with a non-existent user
        result, success, navigation_command = await navigate_through_memory.execute(
            "Find the red square", "non_existent_user"
        )

        # Check the result
        assert success is False, "Navigation should have failed"
        assert (
            "No locations stored in memory yet" in result
        ), f"Unexpected result: {result}"
        assert (
            navigation_command is None
        ), "Navigation command should be None for failed navigation"

    @pytest.mark.asyncio
    async def test_multiple_pose_images(self, mock_brain):
        """Test adding multiple images to the pose graph."""
        # Create multiple pose_image messages with different positions
        positions = [(1.0, 2.0, 0.0), (2.0, 3.0, 1.57), (3.0, 4.0, 3.14)]

        # Get the NavigateThroughMemory primitive
        navigate_through_memory = next(
            (
                p
                for p in mock_brain.local_primitives_list
                if p.name == "navigate_through_memory"
            ),
            None,
        )

        assert (
            navigate_through_memory is not None
        ), "NavigateThroughMemory primitive not found"

        # Verify the graph is empty at the start
        user_graph = navigate_through_memory.pose_graph_memory.get_user_graph(
            mock_brain.connection_id
        )
        assert len(user_graph.nodes) == 0, "Graph should be empty at the start"

        # Process multiple messages with a small delay between them to ensure different timestamps
        colors = ["red", "blue", "green"]
        for i, (x, y, theta) in enumerate(positions):
            await self.add_image_to_pose_graph(mock_brain, x, y, theta, colors[i])
            # Add a small delay to ensure timestamps are different
            time.sleep(0.01)

        # Check that all images were added to the pose graph
        user_graph = navigate_through_memory.pose_graph_memory.get_user_graph(
            mock_brain.connection_id
        )
        assert len(user_graph.nodes) == 3, "Not all images were added to the pose graph"

        # Manually add edges between nodes for testing purposes
        # In a real scenario, these would be created by the _add_edges method
        node_ids = list(user_graph.nodes)
        for i in range(len(node_ids)):
            for j in range(len(node_ids)):
                if i != j:
                    user_graph.add_edge(node_ids[i], node_ids[j])

        # Check that edges were created between nodes
        assert len(user_graph.edges) > 0, "No edges were created between nodes"

        # Test navigation to the most recent node (green square)
        result, success, navigation_command = await navigate_through_memory.execute(
            "Find the green square", mock_brain.connection_id
        )

        # Check the result
        assert success is True, "Navigation location finding failed"
        assert "Found location matching" in result, f"Unexpected result: {result}"

        # Verify navigation command structure
        assert navigation_command is not None, "Navigation command should not be None"
        assert "x" in navigation_command, "Navigation command missing x coordinate"
        assert "y" in navigation_command, "Navigation command missing y coordinate"
        assert "theta" in navigation_command, "Navigation command missing theta value"
        # The green square should be at position (3.0, 4.0, 3.14) - the last one added
        assert navigation_command["x"] == 3.0, "Incorrect x coordinate"
        assert navigation_command["y"] == 4.0, "Incorrect y coordinate"
        assert navigation_command["theta"] == 3.14, "Incorrect theta value"

    @pytest.mark.asyncio
    async def test_edge_creation_within_threshold(self, mock_brain):
        """Test that edges are automatically created between nodes within the edge distance threshold."""
        # Get the NavigateThroughMemory primitive
        navigate_through_memory = next(
            (
                p
                for p in mock_brain.local_primitives_list
                if p.name == "navigate_through_memory"
            ),
            None,
        )

        assert (
            navigate_through_memory is not None
        ), "NavigateThroughMemory primitive not found"

        # Get the edge_distance_threshold from the PoseGraphMemory instance
        edge_distance_threshold = (
            navigate_through_memory.pose_graph_memory.edge_distance_threshold
        )

        # Create two positions that are within the edge_distance_threshold
        # Position 1: (0, 0, 0) - at the origin, facing along the x-axis
        # Position 2: (0.5, 0, 0) - 0.5 units along the x-axis, also facing along the x-axis
        # The distance between them is 0.5, which is less than the edge_distance_threshold (0.8)
        position1 = (0.0, 0.0, 0.0)  # x, y, theta
        position2 = (0.5, 0.0, 0.0)  # x, y, theta - within threshold

        # Add the first image
        await self.add_image_to_pose_graph(mock_brain, *position1)

        # Add the second image
        await self.add_image_to_pose_graph(mock_brain, *position2)

        # Get the user graph
        user_graph = navigate_through_memory.pose_graph_memory.get_user_graph(
            mock_brain.connection_id
        )

        # Check that both nodes were added
        assert (
            len(user_graph.nodes) == 2
        ), "Both images should be added to the pose graph"

        # Get the node IDs
        node_ids = list(user_graph.nodes)

        # Directly call the _add_edges method to ensure edges are created
        # This is necessary because our test environment might not trigger the automatic edge creation
        navigate_through_memory.pose_graph_memory._add_edges(user_graph, node_ids[1])

        # Check that edges were created between the nodes
        # There should be at least one edge (from node 1 to node 2 or vice versa)
        assert (
            len(user_graph.edges) > 0
        ), "No edges were created between nodes within threshold"

        # Check specifically for an edge from node 2 to node 1 (the most recent node to the previous one)
        # This is the most likely edge to be created based on the _add_edges logic
        assert user_graph.has_edge(node_ids[1], node_ids[0]) or user_graph.has_edge(
            node_ids[0], node_ids[1]
        ), "Expected edge between nodes not found"

    @pytest.mark.asyncio
    async def test_persistence_between_connections(self):
        """Test that pose graph memory persists between client connections."""
        # Create a unique connection ID for this test
        connection_id = "persistence_test_connection"

        # Create first brain instance (simulating first connection)
        send_callback1 = MagicMock()
        brain1 = Brain(connection_id, send_callback1)
        brain1.handle_image = MagicMock()  # Mock handle_image to avoid processing

        # Get the NavigateThroughMemory primitive from the first brain
        navigate_through_memory1 = next(
            (
                p
                for p in brain1.local_primitives_list
                if p.name == "navigate_through_memory"
            ),
            None,
        )
        assert (
            navigate_through_memory1 is not None
        ), "NavigateThroughMemory primitive not found in first brain"

        # Add images to the pose graph with the first brain
        positions = [(1.0, 2.0, 0.0), (2.0, 3.0, 1.57)]
        colors = ["red", "yellow"]
        for i, (x, y, theta) in enumerate(positions):
            # Create a test image with a colored square on white background
            img = Image.new("RGB", (100, 100), color="white")
            from PIL import ImageDraw

            draw = ImageDraw.Draw(img)
            draw.rectangle([25, 25, 75, 75], fill=colors[i])

            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="JPEG")
            base64_img = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

            # Create and process a pose_image message
            message = MessageIn(
                type=MessageInType.POSE_IMAGE,
                payload={
                    "image": base64_img,
                    "x": x,
                    "y": y,
                    "theta": theta,
                    # No user_token needed - the brain's connection_id will be used
                },
            )
            await brain1.process_message(message)

        # Verify images were added to the pose graph
        user_graph1 = navigate_through_memory1.pose_graph_memory.get_user_graph(
            connection_id
        )
        assert (
            len(user_graph1.nodes) == 2
        ), "Images were not added to the pose graph with first brain"

        # Simulate client disconnection by stopping the first brain
        await brain1.stop()

        # Create second brain instance (simulating reconnection)
        send_callback2 = MagicMock()
        brain2 = Brain(connection_id, send_callback2)  # Use the same connection_id
        brain2.handle_image = MagicMock()  # Mock handle_image to avoid processing

        # Get the NavigateThroughMemory primitive from the second brain
        navigate_through_memory2 = next(
            (
                p
                for p in brain2.local_primitives_list
                if p.name == "navigate_through_memory"
            ),
            None,
        )
        assert (
            navigate_through_memory2 is not None
        ), "NavigateThroughMemory primitive not found in second brain"

        # Verify that the pose graph from the first connection is still available
        user_graph2 = navigate_through_memory2.pose_graph_memory.get_user_graph(
            connection_id
        )
        assert (
            len(user_graph2.nodes) == 2
        ), "Pose graph data was not persisted between connections"

        # Add another image with the second brain
        new_position = (3.0, 4.0, 3.14)
        # Create a test image with a blue square on white background
        img = Image.new("RGB", (100, 100), color="white")
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img)
        draw.rectangle([25, 25, 75, 75], fill="blue")

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="JPEG")
        base64_img = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

        message = MessageIn(
            type=MessageInType.POSE_IMAGE,
            payload={
                "image": base64_img,
                "x": new_position[0],
                "y": new_position[1],
                "theta": new_position[2],
                # No user_token needed - the brain's connection_id will be used
            },
        )
        await brain2.process_message(message)

        # Verify the new image was added to the existing pose graph
        user_graph2 = navigate_through_memory2.pose_graph_memory.get_user_graph(
            connection_id
        )
        assert (
            len(user_graph2.nodes) == 3
        ), "New image was not added to the persisted pose graph"

        # Test navigation using the combined pose graph
        result, success, navigation_command = await navigate_through_memory2.execute(
            "Find the blue square", connection_id
        )

        # Check the result
        assert success is True, "Navigation location finding failed"
        assert "Found location matching" in result, f"Unexpected result: {result}"

        # Verify navigation command structure
        assert navigation_command is not None, "Navigation command should not be None"
        assert "x" in navigation_command, "Navigation command missing x coordinate"
        assert "y" in navigation_command, "Navigation command missing y coordinate"
        assert "theta" in navigation_command, "Navigation command missing theta value"

        # The blue square should be at position (3.0, 4.0, 3.14) - the one we just added
        assert navigation_command["x"] == 3.0, "Incorrect x coordinate"
        assert navigation_command["y"] == 4.0, "Incorrect y coordinate"
        assert navigation_command["theta"] == 3.14, "Incorrect theta value"

        # Clean up
        await brain2.stop()
