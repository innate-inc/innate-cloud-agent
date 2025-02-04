import asyncio
import websockets
import argparse


async def test_websocket(local=False):
    # Choose URI based on local flag
    uri = (
        "ws://localhost:8765"
        if local
        else "wss://innate-agent-websocket-service-533276562345.us-central1.run.app"
    )

    print(f"Connecting to {uri}...")

    try:
        # 1) Connect to the WebSocket server
        async with websockets.connect(uri) as websocket:
            print("Connected!")

            # 2) Immediately send your hard-coded token
            token = "MY_HARDCODED_TOKEN"
            print(f"Sending token: {token}")
            await websocket.send(token)

            # 3) Listen for messages from the server
            #    We'll just do a simple loop reading whatever the server sends
            while True:
                message = await websocket.recv()
                print("Received from server:", message)
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Connection closed with error: {e}")
    except Exception as e:
        print(f"Failed to connect or other error: {e}")
        # Give the full error message
        print(f"Full error message: {e}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="WebSocket client test script")
    parser.add_argument(
        "--local", action="store_true", help="Connect to local WebSocket server"
    )
    args = parser.parse_args()

    # Run the async function with the local flag
    asyncio.run(test_websocket(args.local))


if __name__ == "__main__":
    main()
