"""
   +-------------------------------------------+
   |                 SERVER                   |
   +-------------------------------------------+
        ^                        |
        | (Client connects)      |
        |                        |
        |  1) send {"type": "ready_for_image"}|
        |                        |
        v                        |
   +-------------------------------------------+
   | 2) receive {"type": "image"} from client |
   +-------------------------------------------+
        |                        ^
        |  3) send {"type": "well_received"}  |
        |  4) (Server does processing)        |
        |  5) send {"type": "action_to_do"}   |
        |                        |
        +-----------> (repeat) <--------------+
"""

import asyncio
import json
import time
import random

import websockets
from websockets.exceptions import ConnectionClosed
from websockets.frames import CloseCode


# Example hard-coded token check
def get_user_from_token(token: str):
    """
    Return user identifier if token is valid; return None if invalid.
    Replace this with real logic (DB lookup, etc.) if needed.
    """
    if token == "MY_HARDCODED_TOKEN":
        return "user123"
    return None


async def echo_handler(websocket):
    """
    This handler:
    1) Receives the FIRST MESSAGE from the client (as the token).
    2) Authenticates or rejects the connection.
    3) If successful, proceeds with the new handshake-based flow:
       - send "ready_for_image"
       - wait for "image"
       - respond "well_received"
       - do processing (simulate with sleep)
       - send "action_to_do"
       - repeat
    """

    # --- AUTHENTICATION STEP ---
    try:
        # Wait for the FIRST message: it should be the token
        token = await websocket.recv()
        print(f"Received token: {token}")
    except ConnectionClosed:
        print("Connection closed before a token was received.")
        return

    user_id = get_user_from_token(token)
    if user_id is None:
        print("Invalid token. Closing connection.")
        await websocket.close(
            code=CloseCode.INTERNAL_ERROR, reason="authentication failed"
        )
        return

    print(f"Client authenticated as: {user_id}")

    # --- HANDSHAKE-BASED FLOW ---
    try:
        while True:
            # 1) Send "ready_for_image"
            msg_ready = {"type": "ready_for_image"}
            await websocket.send(json.dumps(msg_ready))
            print("Server -> Client: 'ready_for_image'")

            # 2) Wait for the next message (should be an 'image')
            try:
                incoming_msg = await websocket.recv()
            except ConnectionClosed:
                print("Connection closed while waiting for an image.")
                return

            try:
                data = json.loads(incoming_msg)
            except json.JSONDecodeError:
                print("Received non-JSON data. Ignoring.")
                continue

            if data.get("type") == "image":
                print("Server received an image (base64).")
                # 3) Acknowledge receipt
                msg_received = {"type": "well_received"}
                await websocket.send(json.dumps(msg_received))
                print("Server -> Client: 'well_received'")

                # 4) Do some "processing"
                processing_time = random.uniform(
                    0.5, 2.0
                )  # Random time between 0.5 and 2 seconds
                time.sleep(processing_time)  # simulate processing
                print(f"Processed for {processing_time:.2f} seconds")

                # 5) Send an action command (e.g., set_velocity)
                #    Here we just alternate left vs right
                turn_left = int(time.time()) % 2 == 0
                if turn_left:
                    vel_left = 10.0
                    vel_right = 0.0
                else:
                    vel_left = 0.0
                    vel_right = 10.0

                msg_action = {
                    "type": "action_to_do",
                    "cmd": "set_velocity",
                    "values": [vel_left, vel_right],
                }
                await websocket.send(json.dumps(msg_action))
                print(
                    f"Server -> Client: 'action_to_do' with velocity {msg_action['values']}"
                )
            else:
                print(f"Unexpected message type from client: {data.get('type')}")

    except ConnectionClosed:
        print("Client disconnected.")


async def main():
    # Listen on port 8765
    server = await websockets.serve(echo_handler, "0.0.0.0", 8765)
    print("Server started at ws://0.0.0.0:8765")
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
