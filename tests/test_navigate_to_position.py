import os
import sys
import pytest
import asyncio
import json
import websockets
import base64
import numpy as np
from PIL import Image
import io
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load the environment variables first if needed for subsequent imports
load_dotenv()

# Import connection_handler from run_server (local application import)
from run_server import connection_handler


async def common_setup(test_name):
    """
    Common setup that starts the server, connects the client,
    sends the auth message, and waits for the "ready_for_image" response.
    """
    port = 8767  # Use a different port than other tests
    # Start the temporary WebSocket server.
    server = await websockets.serve(
        connection_handler, "localhost", port, max_size=10 * 1024 * 1024
    )
    await asyncio.sleep(0.1)  # Allow time for the server to start.

    uri = f"ws://localhost:{port}"
    websocket = await websockets.connect(uri)

    # Send authentication message.
    auth_message = {
        "type": "auth",
        "payload": {"token": f"MY_HARDCODED_TOKEN_{test_name}"},
    }
    await websocket.send(json.dumps(auth_message))

    # Wait for the server to respond with "ready_for_image".
    raw_msg = await websocket.recv()
    msg = json.loads(raw_msg)
    assert (
        msg["type"] == "ready_for_image"
    ), "Authentication did not result in readiness"

    return server, websocket


async def basic_image_handling(
    websocket, image_path="tests/test_navigate.png", image_type="PNG"
):
    """
    Opens an image from a local file, reduces its dimensions by half,
    encodes it in base64, and sends it over the provided websocket.
    Also includes mock depth payload, mock map payload, and robot coordinates.
    """
    with open(image_path, "rb") as img_file:
        image_bytes = img_file.read()

    # Open the image and compute resized dimensions.
    original_image = Image.open(io.BytesIO(image_bytes))
    new_size = (original_image.width // 2, original_image.height // 2)
    # Resize using a high-quality resampling filter.
    resized_image = original_image.resize(new_size, Image.Resampling.LANCZOS)

    # Save the resized image to an in-memory bytes buffer.
    buffer = io.BytesIO()
    resized_image.save(buffer, format=image_type)
    resized_image_bytes = buffer.getvalue()

    # Encode the image.
    encoded_image = base64.b64encode(resized_image_bytes).decode("utf-8")

    # Create mock depth data (all pixels with value 1.0)
    # Using the same dimensions as the resized image
    width, height = new_size
    depth_value = 1.0  # 1 meter depth uniformly

    # Create a uniform depth array and encode to base64
    depth_data = np.ones((height, width), dtype=np.float32) * depth_value
    depth_bytes = depth_data.tobytes()
    depth_b64 = base64.b64encode(depth_bytes).decode("utf-8")

    # Create mock map data (e.g., a 50x50 grid, all free space '0')
    map_width = 50
    map_height = 50
    map_resolution = 0.1  # 0.1 meters per pixel
    map_origin_x = -2.5  # Map origin x, adjusted for larger map size
    map_origin_y = -2.5  # Map origin y, adjusted for larger map size
    map_origin_z = 0.0  # Map origin z in world coordinates (assuming flat)
    map_origin_yaw = 0.0  # Map origin yaw in world coordinates (assuming no rotation)
    map_frame_id = "map"  # Coordinate frame ID
    map_data = np.zeros((map_height, map_width), dtype=np.int8)  # 0 for free space
    map_bytes = map_data.tobytes()
    map_b64 = base64.b64encode(map_bytes).decode("utf-8")

    # Create mock robot coordinates
    robot_coords = {
        "x": 0.0,
        "y": 0.0,
        "theta": 0.0,  # Robot is facing east (0 radians)
    }

    # Send the image message with required depth, map, and robot_coords fields
    image_message = {
        "type": "image",
        "payload": {
            "image_b64": encoded_image,
            "depth": {
                "height": height,
                "width": width,
                "encoding": "32FC1",  # Using 32-bit float encoding
                "data": depth_b64,
            },
            "map": {
                "height": map_height,
                "width": map_width,
                "resolution": map_resolution,
                "origin_x": map_origin_x,
                "origin_y": map_origin_y,
                "origin_z": map_origin_z,
                "origin_yaw": map_origin_yaw,
                "frame_id": map_frame_id,
                "encoding": "8UC1",  # Using 8-bit unsigned char for occupancy grid
                "data": map_b64,
            },
            "robot_coords": robot_coords,
        },
    }
    await websocket.send(json.dumps(image_message))


@pytest.mark.asyncio
async def test_absolute_navigation():
    """
    Test that when the user asks to go to a specific position,
    the robot navigates to the absolute coordinates.
    """
    server, websocket = await common_setup("test_absolute_navigation")

    # First, register the navigate_to_position primitive and a directive
    navigate_guideline = (
        "Use when you need to navigate the robot to the specified position "
        "using provided x, y coordinates, and theta (yaw) angle IN RADIANS."
    )
    directive_text = (
        "You are a helpful robot assistant that can navigate to locations when asked."
    )
    register_message = {
        "type": "register_primitives_and_directive",
        "payload": {
            "primitives": [
                {
                    "name": "navigate_to_position",
                    "guideline": navigate_guideline,
                    "inputs": {
                        "x": "float",
                        "y": "float",
                        "theta": "float",
                    },
                }
            ],
            "directive": directive_text,
        },
    }
    await websocket.send(json.dumps(register_message))

    # Wait for acknowledgment of registration
    raw_msg = await websocket.recv()
    reg_response = json.loads(raw_msg)
    assert (
        reg_response["type"] == "primitives_and_directive_registered"
    ), "Expected primitives_and_directive_registered acknowledgment"

    # Send the chat message for absolute navigation
    chat_message = {
        "type": "chat_in",
        "payload": {"text": "Go to position x=5, y=10"},
    }
    await websocket.send(json.dumps(chat_message))

    # Send the image
    await basic_image_handling(websocket)

    # Expect a vision output response
    raw_msg = await websocket.recv()
    vision_output = json.loads(raw_msg)
    assert (
        vision_output["type"] == "vision_agent_output"
    ), "Expected vision_agent_output message"

    # Check that the next task is the 'navigate_to_position' primitive
    next_task = vision_output["payload"].get("next_task", {})
    assert (
        next_task.get("name") == "navigate_to_position"
    ), "Expected navigate_to_position primitive"

    # Check that the coordinates are absolute (not using delta mode)
    inputs = next_task.get("inputs", {})
    assert (
        abs(inputs.get("x", 0) - 5.0) < 0.1
    ), "Expected x coordinate to be close to 5.0"
    assert (
        abs(inputs.get("y", 0) - 10.0) < 0.1
    ), "Expected y coordinate to be close to 10.0"

    # Again, confirm that we activate the primitive
    activate_message = {
        "type": "primitive_activated",
        "payload": {
            "primitive_id": next_task.get("primitive_id"),
            "primitive_name": next_task.get("name"),
        },
    }
    await websocket.send(json.dumps(activate_message))

    # Next, the server should send a "ready_for_image" message
    raw_msg = await websocket.recv()
    msg = json.loads(raw_msg)
    assert (
        msg["type"] == "ready_for_image"
    ), "Expected ready_for_image after vision output"

    # Clean up: close the server
    server.close()
    await server.wait_closed()


@pytest.mark.asyncio
async def test_relative_navigation_turn_around():
    """
    Test that when the user asks to turn around,
    the robot uses a 180-degree rotation.
    """
    server, websocket = await common_setup("test_relative_navigation_turn_around")

    # First, register the navigate_to_position primitive and a directive
    navigate_guideline = (
        "Use when you need to navigate the robot to the specified position "
        "using provided x, y coordinates, and theta (yaw) angle IN RADIANS."
    )
    directive_text = (
        "You are a helpful robot assistant that can navigate to locations when asked."
    )
    register_message = {
        "type": "register_primitives_and_directive",
        "payload": {
            "primitives": [
                {
                    "name": "navigate_to_position",
                    "guideline": navigate_guideline,
                    "inputs": {
                        "x": "float",
                        "y": "float",
                        "theta": "float",
                    },
                }
            ],
            "directive": directive_text,
        },
    }
    await websocket.send(json.dumps(register_message))

    # Wait for acknowledgment of registration
    raw_msg = await websocket.recv()
    reg_response = json.loads(raw_msg)
    assert (
        reg_response["type"] == "primitives_and_directive_registered"
    ), "Expected primitives_and_directive_registered acknowledgment"

    # Send the chat message for turning around
    chat_message = {
        "type": "chat_in",
        "payload": {"text": "Turn around"},
    }
    await websocket.send(json.dumps(chat_message))

    # Send the image
    await basic_image_handling(websocket)

    # Expect a vision output response
    raw_msg = await websocket.recv()
    vision_output = json.loads(raw_msg)
    assert (
        vision_output["type"] == "vision_agent_output"
    ), "Expected vision_agent_output message"

    # Check that the next task is the 'navigate_to_position' primitive
    next_task = vision_output["payload"].get("next_task", {})
    assert (
        next_task.get("name") == "navigate_to_position"
    ), "Expected navigate_to_position primitive"

    # Check that the theta value is close to π (3.14) radians for a 180-degree turn
    inputs = next_task.get("inputs", {})
    theta = inputs.get("theta", 0)
    assert (
        abs(theta - np.pi) < 0.1 or abs(theta + np.pi) < 0.1
    ), "Expected theta to be close to π radians (180 degrees)"

    # For a turn in place, x and y should be close to 0
    assert (
        abs(inputs.get("x", 999)) < 0.1
    ), "Expected x to be close to 0 for turning in place"
    assert (
        abs(inputs.get("y", 999)) < 0.1
    ), "Expected y to be close to 0 for turning in place"

    # Again, confirm that we activate the primitive
    activate_message = {
        "type": "primitive_activated",
        "payload": {
            "primitive_id": next_task.get("primitive_id"),
            "primitive_name": next_task.get("name"),
        },
    }
    await websocket.send(json.dumps(activate_message))

    # Next, the server should send a "ready_for_image" message
    raw_msg = await websocket.recv()
    msg = json.loads(raw_msg)
    assert (
        msg["type"] == "ready_for_image"
    ), "Expected ready_for_image after vision output"

    # Clean up: close the server
    server.close()
    await server.wait_closed()


@pytest.mark.asyncio
async def test_relative_navigation_move_forward():
    """
    Test that when the user asks to move forward,
    the robot uses a positive x value.
    """
    server, websocket = await common_setup("test_relative_navigation_move_forward")

    # First, register the navigate_to_position primitive and a directive
    navigate_guideline = (
        "Use when you need to navigate the robot to the specified position "
        "using provided x, y coordinates, and theta (yaw) angle IN RADIANS."
    )
    directive_text = (
        "You are a helpful robot assistant that can navigate to locations when asked."
    )
    register_message = {
        "type": "register_primitives_and_directive",
        "payload": {
            "primitives": [
                {
                    "name": "navigate_to_position",
                    "guideline": navigate_guideline,
                    "inputs": {
                        "x": "float",
                        "y": "float",
                        "theta": "float",
                    },
                }
            ],
            "directive": directive_text,
        },
    }
    await websocket.send(json.dumps(register_message))

    # Wait for acknowledgment of registration
    raw_msg = await websocket.recv()
    reg_response = json.loads(raw_msg)
    assert (
        reg_response["type"] == "primitives_and_directive_registered"
    ), "Expected primitives_and_directive_registered acknowledgment"

    # Send the chat message for moving forward
    chat_message = {
        "type": "chat_in",
        "payload": {"text": "Move forward 2 meters"},
    }
    await websocket.send(json.dumps(chat_message))

    # Send the image
    await basic_image_handling(websocket)

    # Expect a vision output response
    raw_msg = await websocket.recv()
    vision_output = json.loads(raw_msg)
    assert (
        vision_output["type"] == "vision_agent_output"
    ), "Expected vision_agent_output message"

    # Check that the next task is the 'navigate_to_position' primitive
    next_task = vision_output["payload"].get("next_task", {})
    assert (
        next_task.get("name") == "navigate_to_position"
    ), "Expected navigate_to_position primitive"

    # The x value should be positive for moving forward
    inputs = next_task.get("inputs", {})
    x = inputs.get("x", 0)
    assert x > 1.5, "Expected x to be positive and close to 2.0 for moving forward"

    # For moving straight forward, y should be close to 0 and theta should be close to 0
    assert (
        abs(inputs.get("y", 999)) < 0.1
    ), "Expected y to be close to 0 for moving straight"
    assert (
        abs(inputs.get("theta", 999)) < 0.1
    ), "Expected theta to be close to 0 for moving straight"

    # Again, confirm that we activate the primitive
    activate_message = {
        "type": "primitive_activated",
        "payload": {
            "primitive_id": next_task.get("primitive_id"),
            "primitive_name": next_task.get("name"),
        },
    }
    await websocket.send(json.dumps(activate_message))

    # Next, the server should send a "ready_for_image" message
    raw_msg = await websocket.recv()
    msg = json.loads(raw_msg)
    assert (
        msg["type"] == "ready_for_image"
    ), "Expected ready_for_image after vision output"

    # Clean up: close the server
    server.close()
    await server.wait_closed()
