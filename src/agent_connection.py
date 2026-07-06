# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Innate Inc

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
from src.auth.token_auth import (
    get_authenticator,
    AuthContext,
    compare_versions,
    should_speak_version_warning,
)
from src.constants_robots import MIN_CLIENT_VERSION


def get_user_from_token(token: str) -> Optional[AuthContext]:
    """
    Validate token against BigQuery user_management table.

    Args:
        token: The robot_special_token to validate

    Returns:
        AuthContext if valid, None otherwise
    """
    authenticator = get_authenticator()
    return authenticator.validate_service_key(token)


class WebSocketAgentConnection:
    """
    Represents a single client connection. Now, as soon as the connection is authenticated,
    we create a dedicated Brain instance that will handle all incoming messages.
    """

    def __init__(self, websocket):
        self.websocket = websocket
        self.user_token: Optional[str] = None
        self.auth_context: Optional[AuthContext] = None
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
        # Check if debug panel is enabled from environment variable
        self.enable_debug_panel = (
            os.environ.get("ENABLE_DEBUG_PANEL", "false").lower() == "true"
        )

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
            print(f"[Auth] Sending ready_for_image to {self.user_token[:10]}...")
            await self.send_message(
                MessageOut(
                    type=MessageOutType.READY_FOR_IMAGE,
                    payload={},
                )
            )
            print(f"[Auth] ready_for_image sent successfully to {self.user_token[:10]}...")

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
            print("[WARN] No token provided, closing connection.")
            return False

        # Validate client version. Old, missing, or malformed versions are
        # rejected. The robot reconnects ~once per second while rejected, so the
        # spoken warning is rate-limited (see should_speak_version_warning) to
        # avoid flooding the robot's text-to-speech queue.
        client_version = auth_msg_payload.get("client_version")
        if not client_version:
            print(
                "[WARN] Rejecting connection: robot did not report its OS version "
                "(client_version missing — likely on 0.2.4 or older, or missing a "
                "release tag)."
            )
            if should_speak_version_warning(token):
                await self.send_message(
                    MessageOut(
                        type=MessageOutType.BRAIN_CHAT_OUT,
                        payload={
                            "text": (
                                "Warning. Your robot could not report its operating "
                                "system version. It may be on version 0 point 2 point 4 "
                                "or older, or missing a release tag. Please update your "
                                "robot to the latest version to connect to the cloud."
                            ),
                        },
                    )
                )
            return False

        is_valid, version_msg = compare_versions(client_version)
        print(f"[INFO] Client version check: {version_msg}")
        if not is_valid:
            print(
                f"[WARN] Rejecting connection: robot OS version {client_version!r} "
                f"is not supported (minimum {MIN_CLIENT_VERSION}). {version_msg}"
            )
            if should_speak_version_warning(token):
                spoken_version = client_version.replace(".", " point ")
                await self.send_message(
                    MessageOut(
                        type=MessageOutType.ERROR,
                        payload={
                            "error": "version_mismatch",
                            "message": (
                                f"Warning. Your robot's operating system, version "
                                f"{spoken_version}, is out of date and no longer "
                                f"supported. Please update your robot to the latest "
                                f"version to connect to the cloud."
                            ),
                            "min_version": MIN_CLIENT_VERSION,
                        },
                    )
                )
            return False
        # Validate token against BigQuery user_management table
        auth_context = get_user_from_token(token)
        if auth_context is None:
            print(f"[WARN] Invalid token, not found in database: {token[:10]}...")
            return False

        self.auth_context = auth_context
        self.user_token = auth_context.robot_special_token
        print(
            f"[INFO] Authenticated user_id: {auth_context.user_id} with token: {self.user_token[:10]}..."
        )
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
