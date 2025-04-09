import os

# os.environ["BAML_LOG"] = "off"

import asyncio
import json
import pytest
import websockets
import base64  # <-- Import base64 for encoding the image
import numpy as np

import sys
from PIL import Image  # <-- Import Pillow for image processing.
import io  # <-- Import io for in-memory byte streams.
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load the environment variables
load_dotenv()

from run_server import connection_handler


async def common_setup(test_name):
    """
    Common setup that starts the server, connects the client,
    sends the auth message, and waits for the "ready_for_image" response.
    """
    port = 8766  # Ensure this port is free during testing.
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


async def basic_image_handling(websocket, image_path, image_type="JPEG"):
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


@pytest.mark.skip(reason="Temporarily deactivated the receipt test")
@pytest.mark.asyncio
async def test_chat_ask_receipt():
    """
    Test that uses a chat message and an image, then verifies the vision output.
    """
    server, websocket = await common_setup("test_chat_ask_receipt")

    # Send the chat message.
    chat_text = (
        "Hello agent. Can you save this receipt and confirm it by telling me "
        "what you did?"
    )
    chat_message = {
        "type": "chat_in",
        "payload": {"text": chat_text},
    }
    await websocket.send(json.dumps(chat_message))

    # Send the image using the helper.
    await basic_image_handling(websocket, "tests/test_receipt.jpg")

    # Expect a vision output response.
    raw_msg = await websocket.recv()
    vision_output = json.loads(raw_msg)
    assert (
        vision_output["type"] == "vision_agent_output"
    ), "Expected vision_agent_output message"

    # Check that the next task is the 'save_receipt' primitive.
    next_task_type = vision_output["payload"].get("next_task", {}).get("type", "")
    assert (
        next_task_type == "save_receipt"
    ), "Expected the save_receipt primitive to be called"

    # Next, the server should send a "ready_for_image" message.
    raw_msg = await websocket.recv()
    msg = json.loads(raw_msg)
    assert (
        msg["type"] == "ready_for_image"
    ), "Expected ready_for_image after vision output"

    # Clean up: close the server.
    server.close()
    await server.wait_closed()


@pytest.mark.asyncio
async def test_chat_ask_to_navigate():
    """
    Test that uses a chat message and an image, then verifies the vision output.
    """
    server, websocket = await common_setup("test_chat_ask_to_navigate")

    # First, register the navigate_to_position primitive and a directive
    navigate_guideline = (
        "Use when you need to navigate the robot to the specified position "
        "using provided x, y coordinates, and theta (yaw) angle IN RADIANS. "
        "Set is_delta=True to use delta mode for relative movement."
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
                        "is_delta": "bool",
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

    # Send the chat message.
    chat_message = {
        "type": "chat_in",
        "payload": {"text": "Hello agent. Can you navigate to x=100, y=100?"},
    }
    await websocket.send(json.dumps(chat_message))

    # Send the image using the helper.
    await basic_image_handling(websocket, "tests/test_navigate.png", "PNG")

    # Expect a vision output response.
    raw_msg = await websocket.recv()
    vision_output = json.loads(raw_msg)
    assert (
        vision_output["type"] == "vision_agent_output"
    ), "Expected vision_agent_output message"

    # Check that the next task is the 'navigate_to_position' primitive.
    next_task = vision_output["payload"].get("next_task", {})
    next_task_type = next_task.get("name", "")
    assert (
        next_task_type == "navigate_to_position"
    ), "Expected the navigate_to_position primitive to be called"

    # Get the primitive_id from the next_task
    primitive_id = next_task.get("primitive_id", "")
    assert primitive_id, "Expected a primitive_id in the next_task"

    # Send primitive_activated message
    activate_message = {
        "type": "primitive_activated",
        "payload": {
            "primitive_id": primitive_id,
            "primitive_name": "navigate_to_position",
        },
    }
    await websocket.send(json.dumps(activate_message))

    # Now, the server should send a "ready_for_image" message.
    raw_msg = await websocket.recv()
    msg = json.loads(raw_msg)
    assert (
        msg["type"] == "ready_for_image"
    ), "Expected ready_for_image after primitive activation"

    # Clean up: close the server.
    server.close()
    await server.wait_closed()


@pytest.mark.asyncio
async def test_chat_ask_to_navigate_with_task_in_execution():
    """
    Test that uses a chat message and an image, verifies that the task started is
    navigate_to_position and then sends another image. That should not stop the
    task.
    """
    server, websocket = await common_setup(
        "test_chat_ask_to_navigate_with_task_in_execution"
    )

    # First, register the navigate_to_position primitive and a directive
    navigate_guideline = (
        "Use when you need to navigate the robot to the specified position "
        "using provided x, y coordinates, and theta (yaw) angle IN RADIANS. "
        "Set is_delta=True to use delta mode for relative movement."
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
                        "is_delta": "bool",
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

    # Send the first navigation command.
    chat_text_1 = "Hello agent. Can you navigate to x=100, y=100?"
    chat_message_1 = {
        "type": "chat_in",
        "payload": {"text": chat_text_1},
    }
    await websocket.send(json.dumps(chat_message_1))

    # Send the image for the first command.
    await basic_image_handling(websocket, "tests/test_navigate.png", "PNG")

    # Expect vision output response for the first command.
    raw_msg = await websocket.recv()
    vision_output_1 = json.loads(raw_msg)
    assert (
        vision_output_1["type"] == "vision_agent_output"
    ), "Expected vision_agent_output message for first navigation command"

    next_task_1 = vision_output_1["payload"].get("next_task", {})
    next_task_type_1 = next_task_1.get("name", "")
    assert (
        next_task_type_1 == "navigate_to_position"
    ), "Expected the navigate_to_position primitive to be called for first navigation"

    # Get the primitive_id from the next_task
    primitive_id_1 = next_task_1.get("primitive_id", "")
    assert primitive_id_1, "Expected a primitive_id in the next_task"

    # Send primitive_activated message for the first navigation
    activate_message_1 = {
        "type": "primitive_activated",
        "payload": {
            "primitive_id": primitive_id_1,
            "primitive_name": "navigate_to_position",
        },
    }
    await websocket.send(json.dumps(activate_message_1))

    # Expect the server to send a "ready_for_image" message after activation.
    raw_msg = await websocket.recv()
    ready_msg_1 = json.loads(raw_msg)
    assert (
        ready_msg_1["type"] == "ready_for_image"
    ), "Expected ready_for_image after activating first navigation primitive"

    # Send the image for the second navigation attempt.
    await basic_image_handling(websocket, "tests/test_navigate.png", "PNG")

    # Expect vision output response for the second navigation attempt.
    raw_msg = await websocket.recv()
    vision_output_2 = json.loads(raw_msg)
    assert (
        vision_output_2["type"] == "vision_agent_output"
    ), "Expected vision_agent_output message for second navigation attempt"
    # Since a task is already in execution, no new task should be created.
    next_task_2 = vision_output_2["payload"].get("next_task", None)
    assert (
        next_task_2 is None
    ), "Expected no new task to be created while a navigation task is already executing"

    # The server should indicate readiness for a new image.
    raw_msg = await websocket.recv()
    ready_msg_2 = json.loads(raw_msg)
    assert (
        ready_msg_2["type"] == "ready_for_image"
    ), "Expected ready_for_image after vision output for second navigation attempt"

    # Clean up: close the server.
    server.close()
    await server.wait_closed()
