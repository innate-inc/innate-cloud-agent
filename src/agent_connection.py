import os
import asyncio
import json
import base64
import datetime
import traceback
from websockets.exceptions import ConnectionClosed
from typing import Optional

from src.message_types import MessageInType, MessageOut, MessageOutType, MessageIn
from src.brain import Brain
from src.debug_panel import register_brain_for_debug, unregister_brain_for_debug


def get_user_from_token(token: str):
    """
    Example token check. Replace with real logic (DB lookup, etc.) if needed.
    """
    if token == "MY_HARDCODED_TOKEN":
        return "user123"
    return None


class WebSocketAgentConnection:
    """
    Represents a single client connection. Now, as soon as the connection is authenticated,
    we create a dedicated Brain instance that will handle all incoming messages.
    """

    def __init__(self, websocket):
        self.websocket = websocket
        self.user_token: Optional[str] = None
        self.recording_dir: str = ""
        self.brain: Optional[Brain] = None
        self.brain_task = None  # Keep a reference to the brain task

        # Check if memory commands are enabled from environment variable
        default_value = "false"
        env_key = "ENABLE_MEMORY_COMMANDS"
        env_value = os.environ.get(env_key, default_value).lower()
        self.enable_memory_commands = env_value == "true"
        if self.enable_memory_commands:
            print("[INFO] Memory state commands are enabled")
        else:
            print("[INFO] Memory state commands are disabled")
        self.enable_debug_panel = False
        # # Check if debug panel is enabled from environment variable
        # self.enable_debug_panel = (
        #     os.environ.get("ENABLE_DEBUG_PANEL", "false").lower() == "true"
        # )

    async def handle_connection(self):
        """
        Main entrypoint after the server accepts a connection.
        Performs authentication, creates a session folder, sets up the Brain,
        and then simply forwards future messages to this brain.
        """
        try:
            # Step 1: Authenticate
            if not await self._authenticate():
                await self.websocket.close()
                return

            # Step 2: Create a folder to store session images
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.recording_dir = f"./recordings/session_{self.user_token}_{timestamp}"
            os.makedirs(self.recording_dir, exist_ok=True)
            print(f"[INFO] Created recording dir: {self.recording_dir}")

            # Step 3: Create a per-connection Brain that will process all messages.
            # Pass memory commands configuration
            brain_args = {
                "connection_id": self.user_token,
                "send_callback": self.send_message,
                "enable_memory_commands": self.enable_memory_commands,
            }
            self.brain = Brain(**brain_args)
            self.brain_task = asyncio.create_task(self.brain.run())
            if self.enable_debug_panel:
                register_brain_for_debug(self.brain)
            print(f"[INFO] Brain instance started for user: {self.user_token}")
            if self.enable_memory_commands:
                print("[INFO] Memory state commands are enabled for this brain")

            # Notify client that the server is ready for an image (or other messages)
            await self.send_message(
                MessageOut(
                    type=MessageOutType.READY_FOR_IMAGE,
                    payload={},
                )
            )
            print("[DEBUG] Sent 'ready_for_image' to client")

            # Main listening loop --
            # For every message received on the websocket,
            # simply forward it to the brain for processing.
            while True:
                try:
                    raw_msg = await self.websocket.recv()
                except ConnectionClosed:
                    print(
                        f"[INFO] WebSocket closed for user: {self.user_token}. Traceback: {traceback.format_exc()}"
                    )
                    break

                try:
                    data_raw = json.loads(raw_msg)
                except json.JSONDecodeError:
                    print("[WARN] Received non-JSON data from client. Ignoring.")
                    continue

                # Parse the incoming data as a MessageIn. This ensures the message is validated.
                try:
                    message_in = MessageIn.model_validate(data_raw)
                except Exception as e:
                    print(f"[WARN] Received invalid message format: {e}")
                    continue

                if message_in.type == MessageInType.IMAGE:
                    await self._save_incoming_image(message_in.payload["image_b64"])

                await self.brain.enqueue_message(message_in)

                # Sleep a tiny bit to avoid busy looping
                await asyncio.sleep(0.01)
        except Exception as e:
            print(f"[ERROR] Exception in handle_connection: {e}")
        finally:
            # Shutdown the brain task gracefully
            if self.brain is not None:
                if self.enable_debug_panel:
                    unregister_brain_for_debug(self.brain.connection_id)
                await self.brain.stop()
            if self.brain_task is not None:
                self.brain_task.cancel()
                try:
                    await self.brain_task
                except asyncio.CancelledError:
                    print("[DEBUG] Brain task cancelled.")
            # Ensure the websocket is closed
            await self.websocket.close()
            print("[INFO] WebSocket connection closed cleanly.")

    async def _authenticate(self) -> bool:
        """
        Wait for the client's first message (the token).
        If valid, store self.user_token. Otherwise, reject.
        """
        try:
            auth_msg = await self.websocket.recv()
        except ConnectionClosed:
            print("[ERROR] Connection closed before token was received.")
            return False

        # The token message itself might be JSON, but here we assume its a string token.
        try:
            auth_msg_json = json.loads(auth_msg)
        except json.JSONDecodeError:
            print("[WARN] Received non-JSON data from client. Ignoring.")
            return False

        # Now we can use the token to get the user_token
        try:
            auth_msg_payload = MessageIn.model_validate(auth_msg_json).payload
            token = auth_msg_payload["token"]
        except Exception as e:
            print(f"[ERROR] Failed to get user_token from token: {e}")
            return False

        if token is None:
            print("[WARN] Invalid token, closing connection.")
            return False

        self.user_token = token
        print(f"[INFO] Authenticated user with token: {self.user_token}")
        return True

    async def _save_incoming_image(self, image_b64: str):
        """
        Optionally, you may want to decode and save the image in self.recording_dir.
        This helper remains available, should you choose to let the brain trigger it.
        """
        try:
            img_data = base64.b64decode(image_b64)
            filename = (
                f"image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S.%f')}.jpg"
            )
            filepath = os.path.join(self.recording_dir, filename)
            with open(filepath, "wb") as f:
                f.write(img_data)
            return filepath
        except Exception as e:
            print(f"[ERROR] Failed to decode/write image: {e}")
            return None

    async def send_message(self, msg: MessageOut):
        """
        Utility to send JSON-encoded messages to the client.
        This method is provided as the callback to the Brain.
        """
        json_msg = msg.model_dump_json()  # or msg.json() in Pydantic v1
        await self.websocket.send(json_msg)
