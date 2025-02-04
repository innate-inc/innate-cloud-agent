import asyncio
import json
import pytest
import websockets

# Import the connection handler from your server code.
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
        auth_message = {
            "type": "auth",
            "payload": {"token": "MY_HARDCODED_TOKEN"},
        }
        await websocket.send(json.dumps(auth_message))

        # The server should respond with a message indicating readiness.
        raw_msg = await websocket.recv()
        msg = json.loads(raw_msg)
        assert msg["type"] == "ready_for_image"

        # ----- STEP 2: Send an image message -----
        # For this test, we simply send a dummy string.
        image_message = {
            "type": "image",
            "payload": {"image_data": "dummy_image_data"},
        }
        await websocket.send(json.dumps(image_message))

        # The brain simulates processing by delaying for 1 second and then responding.
        raw_msg = await websocket.recv()
        msg = json.loads(raw_msg)
        assert msg["type"] == "vision_agent_output"

        # ----- STEP 3: Send a chat message -----
        # Note: Since the brain code calls message.get("text", ""),
        # we include an extra field "text" alongside the required keys.
        chat_message = {
            "type": "chat_in",
            "payload": {"text": "Hello agent"},
        }
        await websocket.send(json.dumps(chat_message))

        raw_msg = await websocket.recv()
        msg = json.loads(raw_msg)
        assert msg["type"] == "chat_out"

        # ----- STEP 4: Send a directive message -----
        directive_message = {
            "type": "directive",
            "payload": {"directive": "Test directive"},
        }
        await websocket.send(json.dumps(directive_message))

        raw_msg = await websocket.recv()
        msg = json.loads(raw_msg)
        assert msg["type"] == "directive_ack"
        # The directive acknowledgment should mention the directive.
        assert "Test directive" in msg.get("payload", "")

    # Shutdown the server.
    server.close()
    await server.wait_closed()
