# agent_connection.py

import os
import asyncio
import json
import base64
import time
import datetime
from websockets.exceptions import ConnectionClosed
from typing import Optional

from message_types import MessageType, VisionAgentOutput


def get_user_from_token(token: str):
    """
    Example token check. Replace with real logic (DB lookup, etc.) if needed.
    """
    if token == "MY_HARDCODED_TOKEN":
        return "user123"
    return None


class WebSocketAgentConnection:
    """
    Represents a single client connection. Handles:
      - Authentication
      - Incoming messages (image, directive, etc.)
      - Outgoing messages (ready_for_image, action_to_do, well_received)
      - Logging images to a per-session directory
    """

    def __init__(self, websocket):
        self.websocket = websocket
        self.user_id: Optional[str] = None
        self.recording_dir: str = ""
        self._processing_image = False

    async def handle_connection(self):
        """
        Main entrypoint after the server accepts a connection.
        Performs authentication, then enters a receive-loop for messages.
        """
        # Step 1: Authenticate
        if not await self._authenticate():
            await self.websocket.close()
            return

        # Step 2: Create a folder to store session images
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_dir = f"./recordings/session_{self.user_id}_{timestamp}"
        os.makedirs(self.recording_dir, exist_ok=True)
        print(f"[INFO] Created recording dir: {self.recording_dir}")

        try:
            # Start by telling the client we are ready for an image
            await self.send_message({"type": MessageType.READY_FOR_IMAGE.value})
            print("[DEBUG] Sent 'ready_for_image' to client")

            # Now enter the main listening loop
            while True:
                try:
                    raw_msg = await self.websocket.recv()
                except ConnectionClosed:
                    print(f"[INFO] Connection closed for user: {self.user_id}")
                    break

                try:
                    data = json.loads(raw_msg)
                except json.JSONDecodeError:
                    print("[WARN] Received non-JSON data from client. Ignoring.")
                    continue

                # Dispatch to the appropriate handler
                msg_type = data.get("type")
                if not msg_type:
                    print("[WARN] No 'type' field in message; ignoring.")
                    continue

                if msg_type == MessageType.IMAGE.value:
                    await self.handle_image_message(data)
                elif msg_type == MessageType.DIRECTIVE.value:
                    await self.handle_directive_message(data)
                else:
                    print(f"[WARN] Unknown message type: {msg_type}")

                # (Optional) Sleep a tiny bit to avoid busy looping
                await asyncio.sleep(0.01)

        except ConnectionClosed:
            print(f"[INFO] Client disconnected: {self.user_id}")
        except Exception as e:
            print(f"[ERROR] Exception in connection loop: {e}")

    async def _authenticate(self) -> bool:
        """
        Wait for the client's first message (the token).
        If valid, store self.user_id. Otherwise, reject.
        """
        try:
            token_msg = await self.websocket.recv()
        except ConnectionClosed:
            print("[ERROR] Connection closed before token was received.")
            return False

        # If the token message itself might be JSON, parse it,
        # but here we assume it's just a string token
        token = token_msg
        user_id = get_user_from_token(token)
        if user_id is None:
            print("[WARN] Invalid token, closing connection.")
            return False

        self.user_id = user_id
        print(f"[INFO] Authenticated user: {self.user_id}")
        return True

    async def handle_image_message(self, data):
        """
        Called when the client sends an 'image' message.
        We'll decode it, store it, respond with "well_received",
        then maybe do some 'AI' process and send an "action_to_do".
        """
        if self._processing_image:
            # In case you don't want parallel image handling
            print("[DEBUG] Still processing previous image. Ignoring new image.")
            return

        self._processing_image = True

        image_b64 = data.get("image_b64")
        if not image_b64:
            print("[WARN] No 'image_b64' field in image message.")
            self._processing_image = False
            return

        # 1) Decode and save image
        await self._save_incoming_image(image_b64)

        # 2) Send "well_received"
        await self.send_message({"type": MessageType.WELL_RECEIVED.value})
        print("[DEBUG] Sent 'well_received' to client")

        # 3) Simulate some “processing” (call your BrainCore or Orchestrator logic)
        #    Let’s pretend we got back a VisionAgentOutput-like object
        #    In reality, you’d call self.orchestrator_node.brain.process_vision_observation(...)
        mock_output = VisionAgentOutput(
            stop_current_task=False,
            observation="Analyzed image successfully",
            thoughts="Some internal thoughts here",
            new_goal=None,
            next_task=None,
            users_implicated=[],
            anticipation=None,
            to_tell_user="We see an object, everything is fine!",
        )

        time.sleep(1)  # Simulate some processing time

        # 4) Send “vision_agent_output” back to the client
        await self.send_message(
            {
                "type": MessageType.VISION_AGENT_OUTPUT.value,
                "payload": mock_output.dict(),  # or .json()
            }
        )
        print("[DEBUG] Sent 'vision_agent_output' to client")

        await self.send_message(
            {
                "type": MessageType.ACTION_TO_DO.value,
                "cmd": "set_velocity",
                "values": [5.0, 5.0],
            }
        )
        print("[DEBUG] Sent 'action_to_do'")

        # Then re-allow new images in next loop
        self._processing_image = False

        await self.send_message(
            {
                "type": MessageType.READY_FOR_IMAGE.value,
            }
        )
        print("[DEBUG] Sent 'ready_for_image'")

    async def handle_directive_message(self, data):
        """
        Called when the client sends a 'directive' message, e.g. to set some new directive.
        In your real code, you might call OrchestratorNode/BrainCore to update the directive.
        """
        directive = data.get("directive")
        if directive:
            print(f"[INFO] Received new directive: {directive}")

            await self.send_message(
                {
                    "type": "directive_ack",
                    "message": f"Directive '{directive}' updated successfully.",
                }
            )
        else:
            print("[WARN] 'directive' field missing in the message.")

    async def _save_incoming_image(self, image_b64: str):
        """
        Decodes the base64 image and saves it in self.recording_dir.
        """
        try:
            img_data = base64.b64decode(image_b64)
            filename = (
                f"image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S.%f')}.jpg"
            )
            filepath = os.path.join(self.recording_dir, filename)
            with open(filepath, "wb") as f:
                f.write(img_data)
            print(f"[INFO] Saved image to {filepath}")
        except Exception as e:
            print(f"[ERROR] Failed to decode/write image: {e}")

    async def send_message(self, msg: dict):
        """
        Utility to send JSON-encoded messages to the client.
        """
        try:
            await self.websocket.send(json.dumps(msg))
        except ConnectionClosed:
            print("[WARN] Attempted to send message after connection was closed.")
