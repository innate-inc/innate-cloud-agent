import asyncio
import json
import time
import openai  # Ensure the OpenAI SDK is installed

from src.message_types import (
    MessageIn,
    MessageInType,
    MessageOut,
    Task,
    TaskType,
    VisionAgentOutput,
)


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
        # Flag to override the next vision output (set via a chat_in command)
        self.forward_command_active = False
        # NEW: Store the latest user message that should be consumed once by the vision language model.
        self.latest_user_message = None

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
        time_start = time.time()

        if message_type == MessageInType.IMAGE:
            await self.handle_image(message)
        elif message_type == MessageInType.CHAT_IN:
            await self.handle_chat_in(message)
        elif message_type == MessageInType.DIRECTIVE:
            await self.handle_directive(message)
        else:
            await self.handle_unknown(message)

        print(
            f"[Brain {self.connection_id}] Processed message in {time.time() - time_start} seconds"
        )

    async def call_visual_language_model(self, user_prompt: str) -> VisionAgentOutput:
        """
        Calls the external visual language model (GPT-4-O 2024-11-20) with the given prompt.
        Expects the model to return a JSON structure adhering to the VisionAgentOutput schema.
        """
        try:
            # Call the OpenAI chat completion API asynchronously using the new parsing format.
            completion = openai.beta.chat.completions.parse(
                model="gpt-4o-2024-11-20",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a vision agent that processes images. Analyze the image and provide observations,"
                            "thoughts, and other insights. Unless the user explicitly instructs you to perform an action,"
                            "do not set a next task—in such cases, 'next_task' should be null. Provide a JSON response that"
                            "strictly adheres to the VisionAgentOutput schema. The JSON should include the keys: stop_current_task,"
                            "observation, thoughts, new_goal, next_task, users_implicated, anticipation, and to_tell_user."
                        ),
                    },
                    {"role": "user", "content": user_prompt},
                ],
                response_format=VisionAgentOutput,
            )
            # Extract the parsed content from the first choice.
            vision_output = completion.choices[0].message.parsed
            print(
                f"[Brain {self.connection_id}] Vision output: {vision_output.model_dump()}"
            )
            return vision_output
        except Exception as e:
            print(
                f"[Brain {self.connection_id}] Error calling visual language model: {e}"
            )
            # Fallback logic: provide a default action if the model call fails.
            fallback_task = Task(
                type=TaskType.VELOCITY_CONTROL,
                description=json.dumps({"forward": 0.0, "angle": 0.0}),
            )
            return VisionAgentOutput(
                stop_current_task=False,
                observation="Image processed using fallback logic.",
                thoughts=f"Fallback due to error: {str(e)}",
                new_goal=None,
                next_task=fallback_task,
                users_implicated=[],
                anticipation=None,
                to_tell_user="Fallback: Image processed.",
            )

    async def handle_image(self, message: MessageIn):
        """
        Handle messages of type 'image'.
        Processes the image and uses a visual language model to decide the next action,
        sending back a structured vision agent output.

        This updated version makes sure to include any image provided in the message payload.
        It checks for either an 'image_url' or an 'image_b64' field.
        """
        # Simulate image processing delay
        await asyncio.sleep(1)

        # Start with the latest stored user message if available.
        if self.latest_user_message:
            user_prompt_text = f"The user said: {self.latest_user_message}"
            self.latest_user_message = None
        else:
            user_prompt_text = "The user said nothing recently."

        # Check if the incoming message contains an image URL.
        base64_img = message.payload["image_b64"]
        # Construct a Data URL for a JPEG image.
        user_prompt_image = f"data:image/jpeg;base64,{base64_img}"

        user_prompt = [
            {"type": "text", "text": user_prompt_text},
            {"type": "image_url", "image_url": {"url": user_prompt_image}},
        ]

        print(f"[Brain {self.connection_id}] Sending request to visual language model.")

        # Call the visual language model with the combined prompt.
        vision_output = await self.call_visual_language_model(user_prompt)

        next_task_type = (
            vision_output.next_task.type if vision_output.next_task else "None"
        )

        print(
            f"[Brain {self.connection_id}] Vision output has determined task to do next to be: {next_task_type}"
        )

        # Build the response message using the structured output.
        response = MessageOut(
            type="vision_agent_output",
            payload=vision_output.model_dump(),
        )
        print(
            f"[Brain {self.connection_id}] Sending response to client with type: {response.type}"
        )
        await self.send_callback(response)
        await self.send_callback(MessageOut(type="ready_for_image", payload={}))

    async def handle_chat_in(self, message: MessageIn):
        """
        Handle messages of type 'chat_in'.
        Echoes back the text received and, if a special command is detected,
        sets a flag to modify the next vision output.
        """
        text = message.payload["text"]

        # If the message is like "Go to XXX", send back a chat_out with "Going to XXX"
        if text.startswith("Go to "):
            response = MessageOut(
                type="chat_out",
                payload={"text": f"Going to {text[5:]}"},
            )
            await self.send_callback(response)

        # Save the latest user message
        self.latest_user_message = text

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
