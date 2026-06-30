# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Innate Inc

# scalable_server.py

import asyncio
import websockets
import argparse
import os

# import os

# os.environ["BAML_LOG"] = "off"

from src.agent_connection import WebSocketAgentConnection
from src.debug_panel import start_debug_panel

# Import dotenv
from dotenv import load_dotenv

load_dotenv()

# Optionally read port from environment or config
WEBSOCKET_PORT = int(os.environ.get("PORT", "8080"))
DEBUG_PANEL_PORT = 8081


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Start the WebSocket server")
    parser.add_argument(
        "--enable-memory-commands",
        action="store_true",
        help="Enable memory state management commands",
    )
    parser.add_argument(
        "--debug-panel",
        action="store_true",
        help="Enable the debug panel web UI",
        default=False,
    )
    parser.add_argument(
        "--debug-panel-port",
        type=int,
        default=DEBUG_PANEL_PORT,
        help=f"Port for the debug panel (default: {DEBUG_PANEL_PORT})",
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

    # Start debug panel if enabled
    debug_server = None
    if args.debug_panel:
        os.environ["ENABLE_DEBUG_PANEL"] = "true"
        debug_server = start_debug_panel(port=args.debug_panel_port)
        print(f"Debug panel available at http://localhost:{args.debug_panel_port}")
    else:
        os.environ["ENABLE_DEBUG_PANEL"] = "false"

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
