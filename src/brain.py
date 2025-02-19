import asyncio
import json
import time
import traceback

from src.message_types import (
    MessageIn,
    MessageInType,
    MessageOut,
)
from src.agents.baml_agent import vision_agent
from src.baml_client.types import VisionAgentOutput
from src.primitives.navigate_to_position import NavigateToPosition
from src.primitives.transforms import primitive_to_dict
from src.history import History, HistoryEntryType
from src.agents.types import VisionAgentInput, PrimitiveDefinition


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
        # Store the latest user message that should be consumed once by the visual language model.
        self.latest_user_message = None
        self.primitives_list = [
            primitive_to_dict(NavigateToPosition()),
            # primitive_to_dict(SaveReceipt()),
        ]
        self.primitive_in_execution = None

        # Initialize history to record chat messages and vision agent outputs.
        self.history = History()

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
        elif message_type == MessageInType.PRIMITIVE_COMPLETED:
            await self.handle_primitive_completed(message)
        elif message_type == MessageInType.PRIMITIVE_ACTIVATED:
            await self.handle_primitive_activated(message)
        else:
            await self.handle_unknown(message)

        print(
            f"[Brain {self.connection_id}] Processed message in {time.time() - time_start} seconds"
        )

    async def call_visual_language_model(self, vlm_inputs: dict) -> VisionAgentOutput:
        """
        Calls the external visual language model (GPT-4-O 2024-11-20) with the given prompt.
        Expects the model to return a JSON structure adhering to the VisionAgentOutput schema.
        """
        try:
            print(
                f"[Brain {self.connection_id}] Calling visual language model while current primitive is {self.primitive_in_execution['name'] if self.primitive_in_execution else 'None'}"
            )
            if self.latest_user_message:
                print(
                    f"[Brain {self.connection_id}] Sending user message to vision agent: {vlm_inputs['user_prompt_text']}"
                )
            completion = await vision_agent(vlm_inputs)
            if completion.next_task:  # Keep the current task if appropriate
                self.primitive_in_execution = completion.next_task
            return completion
        except Exception as e:
            print(
                f"[Brain {self.connection_id}] Error calling visual language model: {e}"
            )
            # Fallback logic: provide a default action if the model call fails.
            return VisionAgentOutput(
                stop_current_task=True,
                observation="The brain failed, so it stopped the current task.",
                thoughts=f"Fallback due to error: {str(e)}\nTraceback: {traceback.format_exc()}",
                new_goal=None,
                next_task=None,
                anticipation=None,
                to_tell_user="BEEP BOOP BEEP BOOP, the brain failed. Stopping the current task.",
            )

    async def handle_image(self, message: MessageIn):
        """
        Handle messages of type 'image'.
        Processes the image and uses a visual language model to decide the next action,
        sending back a structured vision agent output.
        """
        # Simulate image processing delay
        await asyncio.sleep(1)

        # Start with the latest stored user message if available.
        if self.latest_user_message:
            user_prompt_text = self.latest_user_message
            self.latest_user_message = None
        else:
            user_prompt_text = None

        base64_img = message.payload["image_b64"]

        # Convert the current primitive in execution (if any) into a PrimitiveDefinition instance.
        primitive_in_execution = None
        if self.primitive_in_execution:
            primitive_in_execution = PrimitiveDefinition.model_validate(
                self.primitive_in_execution
            )

        # Convert the list of primitives into a list of PrimitiveDefinition instances.
        primitives_list = [
            PrimitiveDefinition.model_validate(prim) for prim in self.primitives_list
        ]

        # Create a VisionAgentInput instance with validated data.
        vlm_inputs = VisionAgentInput(
            base64_img=base64_img,
            user_prompt_text=user_prompt_text,
            primitive_in_execution=primitive_in_execution,
            primitives_list=primitives_list,
            history_as_string=self.history.get_as_string(),
        )

        # Call the visual language model with the validated inputs.
        vision_output = await self.call_visual_language_model(vlm_inputs)

        next_task_type = (
            vision_output.next_task["type"] if vision_output.next_task else "None"
        )

        print(
            f"[Brain {self.connection_id}] Agent decided next task to be: {next_task_type}"
        )

        # Record the vision agent output in the history.
        self.history.add(
            HistoryEntryType.VISION_AGENT_OUTPUT,
            description=json.dumps(vision_output.model_dump()),
        )

        response = MessageOut(
            type="vision_agent_output",
            payload=vision_output.model_dump(),
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

        # Save the latest user message for processing.
        self.latest_user_message = text

        # Record this chat message in the history.
        self.history.add(HistoryEntryType.CHAT_MESSAGE, description=text)

    async def handle_primitive_completed(self, message: MessageIn):
        """
        Handle messages of type 'primitive_completed'.
        Processes the primitive completion and sends an acknowledgment.
        """
        primitive_name = message.payload["primitive_name"]
        print(f"[Brain {self.connection_id}] Primitive '{primitive_name}' completed.")
        if primitive_name == self.primitive_in_execution["name"]:
            self.primitive_in_execution = None
        else:
            raise ValueError(
                f"[Brain {self.connection_id}] Primitive '{primitive_name}' is not the current primitive in execution. That's a weird bug."
            )

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

    async def handle_primitive_activated(self, message: MessageIn):
        """
        Handle messages of type 'primitive_activated'.
        Processes the primitive activation and sends an acknowledgment.
        """
        primitive_name = message.payload["primitive_name"]
        print(f"[Brain {self.connection_id}] Primitive '{primitive_name}' activated.")
        self.primitive_in_execution = next(
            (prim for prim in self.primitives_list if prim["name"] == primitive_name),
            None,
        )
        await self.send_callback(MessageOut(type="ready_for_image", payload={}))

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
