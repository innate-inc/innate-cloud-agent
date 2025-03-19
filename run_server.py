# scalable_server.py

import asyncio
import websockets
import argparse
import os

# import os

# os.environ["BAML_LOG"] = "off"

from src.agent_connection import WebSocketAgentConnection

# Import dotenv
from dotenv import load_dotenv

load_dotenv()

# Optionally read port from environment or config
WEBSOCKET_PORT = 8765


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Start the WebSocket server")
    parser.add_argument(
        "--enable-memory-commands",
        action="store_true",
        help="Enable memory state management commands",
    )
    return parser.parse_args()


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
    # Parse arguments
    args = parse_arguments()

    # Set environment variable based on arguments
    if args.enable_memory_commands:
        os.environ["ENABLE_MEMORY_COMMANDS"] = "true"
        print("Memory state commands are ENABLED")
    else:
        os.environ["ENABLE_MEMORY_COMMANDS"] = "false"
        print("Memory state commands are DISABLED")

    print(f"Starting WebSocket server on port {WEBSOCKET_PORT}...")
    async with websockets.serve(
        connection_handler, "0.0.0.0", WEBSOCKET_PORT, max_size=10 * 1024 * 1024
    ):
        print(f"Server started. Listening at ws://0.0.0.0:{WEBSOCKET_PORT}")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server shutting down.")
