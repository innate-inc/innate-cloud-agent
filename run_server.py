# scalable_server.py

import asyncio
import websockets

import os

# os.environ["BAML_LOG"] = "off"

from src.agent_connection import WebSocketAgentConnection

# Import dotenv
from dotenv import load_dotenv

load_dotenv()

# Optionally read port from environment or config
WEBSOCKET_PORT = 8765


async def connection_handler(websocket):
    """
    This coroutine is called for every new client connection.
    We create a new WebSocketAgentConnection and let it handle messages.
    """
    connection = WebSocketAgentConnection(websocket)
    await connection.handle_connection()


async def main():
    """
    Main entrypoint to start the server.
    """
    print(f"Starting WebSocket server on port {WEBSOCKET_PORT}...")
    async with websockets.serve(connection_handler, "0.0.0.0", WEBSOCKET_PORT):
        print(f"Server started. Listening at ws://0.0.0.0:{WEBSOCKET_PORT}")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server shutting down.")
