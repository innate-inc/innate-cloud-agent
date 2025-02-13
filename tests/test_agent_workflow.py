import os

os.environ["BAML_LOG"] = "off"

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


@pytest.mark.asyncio
async def test_basic_workflow():
    port = 8766  # Make sure this port is free during testing.
    # Start a temporary WebSocket server for the test.
    server = await websockets.serve(connection_handler, "localhost", port)
    # Give the server a short moment to start.
    await asyncio.sleep(0.1)

    uri = f"ws://localhost:{port}"
    async with websockets.connect(uri) as websocket:
        # ----- STEP 1: Authenticate -----
        # The authentication message should follow the MessageIn schema.
        print("Authenticating...")
        auth_message = {
            "type": "auth",
            "payload": {"token": "MY_HARDCODED_TOKEN"},
        }
        await websocket.send(json.dumps(auth_message))

        # The server should respond with a message indicating readiness.
        raw_msg = await websocket.recv()
        msg = json.loads(raw_msg)
        assert msg["type"] == "ready_for_image"

        # ----- STEP 2: Send a chat message -----
        # Note: Since the brain code calls message.get("text", ""),
        # we include an extra field "text" alongside the required keys.
        chat_message = {
            "type": "chat_in",
            "payload": {"text": "Hello agent. Can you respond to me later?"},
        }
        await websocket.send(json.dumps(chat_message))

        # ----- STEP 3: Send an image message with reduced dimensions -----
        # Open the image from ../baml_test/test.jpg, reduce its size by half, encode it in base64, and send it.
        image_path = os.path.join(os.path.dirname(__file__), "test_receipt.jpg")
        with open(image_path, "rb") as img_file:
            image_bytes = img_file.read()

        # Open the image via Pillow.
        original_image = Image.open(io.BytesIO(image_bytes))
        # Compute new dimensions: half the width and height.
        new_size = (original_image.width // 2, original_image.height // 2)
        # Use Image.Resampling.LANCZOS for high-quality downsampling.
        resized_image = original_image.resize(new_size, Image.Resampling.LANCZOS)
        # Save resized image to a bytes buffer.
        buffer = io.BytesIO()
        resized_image.save(buffer, format="JPEG")
        resized_image_bytes = buffer.getvalue()

        # Encode the resized image.
        encoded_image = base64.b64encode(resized_image_bytes).decode("utf-8")

        image_message = {
            "type": "image",
            "payload": {"image_b64": encoded_image},
        }
        await websocket.send(json.dumps(image_message))

        # The brain simulates processing by delaying for 1 second and then responding.
        raw_msg = await websocket.recv()
        vision_agent_output_msg = json.loads(raw_msg)
        assert vision_agent_output_msg["type"] == "vision_agent_output"

        # Assert it is the right primitive that is called
        assert (
            vision_agent_output_msg["payload"]["next_task"]["type"] == "save_receipt"
        ), "Expected 'save_receipt' primitive to be called"

        # Now we expect the brain to send a "ready_for_image" message.
        raw_msg = await websocket.recv()
        msg = json.loads(raw_msg)
        assert msg["type"] == "ready_for_image"

        # ----- STEP 4: Verify we received a chat message -----
        print(f"Received chat message: {vision_agent_output_msg}")
        assert (
            vision_agent_output_msg["payload"]["to_tell_user"] != ""
        ), "Expected 'to_tell_user' in vision_agent_output_msg['payload'] to be a non-empty string"

        # ----- STEP 5: Send a directive message -----
        directive_message = {
            "type": "directive",
            "payload": {"directive": "Test directive"},
        }
        await websocket.send(json.dumps(directive_message))

        raw_msg = await websocket.recv()
        msg = json.loads(raw_msg)
        assert msg["type"] == "directive_ack"
        # The directive acknowledgment should mention the directive.

    # Shutdown the server.
    server.close()
    await server.wait_closed()
