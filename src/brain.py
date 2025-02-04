import asyncio

from src.message_types import MessageIn, MessageOut


class Brain:
    def __init__(self, connection_id: str, send_callback):
        """
        connection_id: an identifier for this brain instance (for logging/debugging)
        send_callback: an async function to send a response back to the client.
        """
        self.connection_id = connection_id
        self.send_callback = send_callback
        self.message_queue = asyncio.Queue()
        self.running = True

    async def enqueue_message(self, message: MessageIn):
        """
        Called externally (by your websocket connection handler) to push
        messages into the brain for processing.
        """
        await self.message_queue.put(message)

    async def run(self):
        """
        The brain's main loop. It runs on a single thread (event loop), processes
        one message at a time, and sends back results with send_callback.
        """
        while self.running:
            message = await self.message_queue.get()
            if message is None:
                break  # Allow graceful shutdown when a None message is pushed
            await self.process_message(message)

    async def process_message(self, message: MessageIn):
        """
        Process a standardized message and dispatch it to the appropriate handler
        based on the message type.
        """
        message_type = message.type
        print(f"[Brain {self.connection_id}] Processing message: {message_type}")

        if message_type == "image":
            await self.handle_image(message)
        elif message_type == "chat_in":
            await self.handle_chat_in(message)
        elif message_type == "directive":
            await self.handle_directive(message)
        else:
            await self.handle_unknown(message)

    async def handle_image(self, message: MessageIn):
        """
        Handle messages of type 'image'.
        Simulates image processing (e.g., running ML inference) with a delay.
        """
        await asyncio.sleep(1)
        response = MessageOut(
            type="vision_agent_output",
            payload={
                "stop_current_task": False,
                "observation": "Analyzed image successfully",
                "thoughts": "Brain processing logic applied",
                "new_goal": None,
                "next_task": None,
                "users_implicated": [],
                "anticipation": None,
                "to_tell_user": "Image processed.",
            },
        )
        await self.send_callback(response)
        await self.send_callback(MessageOut(type="ready_for_image", payload={}))

    async def handle_chat_in(self, message: MessageIn):
        """
        Handle messages of type 'chat_in'.
        Echoes back the text received.
        """
        text = message.payload["text"]
        response = MessageOut(
            type="chat_out",
            payload={"text": f"Echo: {text}"},
        )
        await self.send_callback(response)

    async def handle_directive(self, message: MessageIn):
        """
        Handle messages of type 'directive'.
        Processes the directive and sends an acknowledgment.
        """
        directive = message.payload["directive"]
        response = MessageOut(
            type="directive_ack",
            payload={"text": f"Directive '{directive}' processed."},
        )
        await self.send_callback(response)

    async def handle_unknown(self, message: MessageIn):
        """
        Handle any unrecognized message types.
        """
        response = MessageOut(
            type="error",
            payload={"text": f"Unhandled message type: {message.type}"},
        )
        await self.send_callback(response)

    async def stop(self):
        """
        Stop the brain by flagging running=False and enqueueing a None message to exit the loop.
        """
        self.running = False
        await self.message_queue.put(None)
