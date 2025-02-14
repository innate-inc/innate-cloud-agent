import os

# os.environ["BAML_LOG"] = "off"

import asyncio
import json
import pytest
import websockets
import base64  # <-- Import base64 for encoding the image

import sys
import os
from PIL import Image  # <-- Import Pillow for image processing.
import io  # <-- Import io for in-memory byte streams.

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from run_server import connection_handler


async def common_setup():
    """
    Common setup that starts the server, connects the client,
    sends the auth message, and waits for the "ready_for_image" response.
    """
    port = 8766  # Ensure this port is free during testing.
    # Start the temporary WebSocket server.
    server = await websockets.serve(connection_handler, "localhost", port)
    await asyncio.sleep(0.1)  # Allow time for the server to start.

    uri = f"ws://localhost:{port}"
    websocket = await websockets.connect(uri)

    # Send authentication message.
    auth_message = {
        "type": "auth",
        "payload": {"token": "MY_HARDCODED_TOKEN"},
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

    # Send the image message.
    image_message = {
        "type": "image",
        "payload": {"image_b64": encoded_image},
    }
    await websocket.send(json.dumps(image_message))


@pytest.mark.asyncio
async def test_chat_ask_receipt():
    """
    Test that uses a chat message and an image, then verifies the vision output.
    """
    server, websocket = await common_setup()

    # Send the chat message.
    chat_message = {
        "type": "chat_in",
        "payload": {
            "text": "Hello agent. Can you save this receipt and confirm it by telling me what you did?"
        },
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
    server, websocket = await common_setup()

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

    # Check that the next task is the 'save_receipt' primitive.
    next_task_type = vision_output["payload"].get("next_task", {}).get("type", "")
    assert (
        next_task_type == "navigate_to_position"
    ), "Expected the navigate_to_position primitive to be called"

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
async def test_chat_ask_to_navigate_with_task_in_execution():
    """
    Test that uses a chat message and an image, verifies that the task started is navigate_to_position
    and then sends another image. That should not stop the task.
    """
    server, websocket = await common_setup()

    # Send the first navigation command.
    chat_message_1 = {
        "type": "chat_in",
        "payload": {"text": "Hello agent. Can you navigate to x=100, y=100?"},
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
    next_task_type_1 = vision_output_1["payload"].get("next_task", {}).get("type", "")
    assert (
        next_task_type_1 == "navigate_to_position"
    ), "Expected the navigate_to_position primitive to be called for first navigation"

    # Expect the server to send a "ready_for_image" message.
    raw_msg = await websocket.recv()
    ready_msg_1 = json.loads(raw_msg)
    assert (
        ready_msg_1["type"] == "ready_for_image"
    ), "Expected ready_for_image after vision output for first navigation"

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

    # The server should again indicate readiness for a new image.
    raw_msg = await websocket.recv()
    ready_msg_2 = json.loads(raw_msg)
    assert (
        ready_msg_2["type"] == "ready_for_image"
    ), "Expected ready_for_image after vision output for second navigation attempt"

    # Clean up: close the server.
    server.close()
    await server.wait_closed()


@pytest.mark.asyncio
async def test_directive_workflow():
    """
    Test that sends a directive message and expects a directive acknowledgment.
    """
    server, websocket = await common_setup()

    directive_message = {
        "type": "directive",
        "payload": {"directive": "Test directive"},
    }
    await websocket.send(json.dumps(directive_message))

    # Verify that the server responds with a directive acknowledgment.
    raw_msg = await websocket.recv()
    msg = json.loads(raw_msg)
    assert msg["type"] == "directive_ack", "Expected a directive_ack message"

    # Clean up: close the server.
    server.close()
    await server.wait_closed()
