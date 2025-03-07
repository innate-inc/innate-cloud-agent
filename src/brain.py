import asyncio
import json
import time


from src.message_types import (
    MessageIn,
    MessageInType,
    MessageOut,
    MessageOutType,
)
from src.primitives.transforms import primitive_to_object
from src.history import History, HistoryEntryType
from src.agents.types import PrimitiveDefinition
from src.primitives.navigate_in_sight import NavigateInSight
from src.primitives.navigate_through_memory import NavigateThroughMemory

from src.brain_utils.logger import BrainLogger
from src.brain_utils.image_processor import ImageProcessor
from src.brain_utils.vision_service import VisionService
from src.brain_utils.navigation_handler import NavigationHandler


def prim_list_to_prim_obj_list(prim_list):
    return [primitive_to_object(prim) for prim in prim_list]


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
        self.primitives_list = []

        self.local_primitives_list = [
            NavigateInSight(),
            NavigateThroughMemory(),
        ]  # These are the ones defined in the brain here, not registered with the server by the user
        self.primitive_in_execution = None
        self.directive = None  # Store the directive that will steer the VLM

        # Initialize history to record chat messages and vision agent outputs.
        self.history = History()

        # Initialize logger and helper modules
        self.logger = BrainLogger(connection_id)
        self.image_processor = ImageProcessor(self.logger)
        self.vision_service = VisionService(self.logger)
        self.navigation_handler = NavigationHandler(
            self.logger,
            self.local_primitives_list,
        )

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
        try:
            message_type = message.type

            self.logger.info(f"Processing message: {message_type}")
            time_start = time.time()

            if message_type == MessageInType.IMAGE:
                await self.handle_image(message)
            elif message_type == MessageInType.POSE_IMAGE:
                await self.handle_pose_image(message)
            elif message_type == MessageInType.CHAT_IN:
                await self.handle_chat_in(message)
            elif message_type == MessageInType.PRIMITIVE_COMPLETED:
                await self.handle_primitive_completed(message)
            elif message_type == MessageInType.PRIMITIVE_ACTIVATED:
                await self.handle_primitive_activated(message)
            elif message_type == MessageInType.REGISTER_PRIMITIVES_AND_DIRECTIVE:
                await self.handle_register_primitives_and_directive(message)
            else:
                await self.handle_unknown(message)

            self.logger.info(f"Processed message in {time.time() - time_start} seconds")
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")

    async def handle_image(self, message: MessageIn):
        """Handle messages of type 'image'."""
        # Extract data from payload

        self.logger.debug(f"Received image message: {message.payload.keys()}")
        base64_img, depth_payload, robot_coords = (
            self.image_processor.extract_image_data(message.payload)
        )

        # Ensure image is in JPEG format
        try:
            base64_img = self.image_processor.ensure_jpeg_format(base64_img)
        except ValueError as e:
            self.logger.error(f"Image format error: {e}")
            # Send error response to client
            await self.send_callback(
                MessageOut(type="error", payload={"text": f"Image format error: {e}"})
            )
            # Request a new image
            await self.send_callback(MessageOut(type="ready_for_image", payload={}))
            return

        # Process depth map if available
        if depth_payload:
            self.image_processor.process_depth_map(depth_payload)

        # Convert the local primitives list to a list of PrimitiveDefinition instances
        local_primitives_list = prim_list_to_prim_obj_list(self.local_primitives_list)

        # Call VLM and get output
        vision_output = await self.vision_service.call_visual_language_model(
            base64_img=base64_img,
            user_prompt_text=self.latest_user_message,
            primitive_in_execution=self.primitive_in_execution,
            primitives_list=local_primitives_list + self.primitives_list,
            history_as_string=self.history.get_as_string(),
            robot_coords=robot_coords,
            directive=self.directive,
        )

        # Clear the user message as it's been consumed
        self.latest_user_message = None

        # Validate the next task
        vision_output.next_task = (
            PrimitiveDefinition.model_validate(vision_output.next_task)
            if vision_output.next_task
            else None
        )

        # Update primitive_in_execution if needed
        if vision_output.next_task:
            self.primitive_in_execution = PrimitiveDefinition.model_validate(
                vision_output.next_task
            )

        # Handle special case for navigate_in_sight
        if (
            vision_output.next_task
            and vision_output.next_task.name == "navigate_in_sight"
        ):
            vision_output = await self.navigation_handler.handle_navigate_in_sight(
                vision_output, robot_coords, base64_img, depth_payload
            )
            # Make sure to update our primitive_in_execution to match what was created in navigate_in_sight
            if vision_output.next_task:
                self.primitive_in_execution = vision_output.next_task

        # Handle special case for navigate_through_memory
        if (
            vision_output.next_task
            and vision_output.next_task.name == "navigate_through_memory"
        ):
            vision_output = await self.navigation_handler.handle_navigate_through_memory(
                vision_output, self.connection_id
            )
            # Make sure to update our primitive_in_execution to match what was created
            if vision_output.next_task:
                self.primitive_in_execution = vision_output.next_task

        # Send response and prepare for next image
        await self._send_vision_output(vision_output)

    async def handle_pose_image(self, message: MessageIn):
        """Handle messages of type 'pose_image'."""
        # Extract data from payload
        base64_img = message.payload.get("image", "")
        x = message.payload.get("x", 0.0)
        y = message.payload.get("y", 0.0)
        theta = message.payload.get("theta", 0.0)

        # Always use the connection_id as the user token for pose graph memory
        # Ignore any user_token in the payload
        user_token = self.connection_id

        # Find the NavigateThroughMemory primitive in the local_primitives_list
        navigate_through_memory = next(
            (
                p
                for p in self.local_primitives_list
                if p.name == "navigate_through_memory"
            ),
            None,
        )

        if navigate_through_memory:
            # Use the PoseGraphMemory instance from the primitive
            pose_graph_memory = navigate_through_memory.pose_graph_memory

            if not pose_graph_memory.should_add_node(user_token, x, y, theta):
                self.logger.debug(
                    f"Skipping image addition to pose graph because a close node already exists"
                )
                return

            # Add the image to the pose graph
            self.logger.debug(
                f"Adding image to pose graph with user_token: {user_token}"
            )
            node_id = pose_graph_memory.add_image_to_graph(
                user_token, base64_img, x, y, theta
            )

            self.logger.info(f"Added image to pose graph with node ID: {node_id}")
        else:
            self.logger.error("NavigateThroughMemory primitive not found")

    async def _send_vision_output(self, vision_output):
        # Record the vision agent output in the history.
        self.history.add(
            HistoryEntryType.VISION_AGENT_OUTPUT,
            description=json.dumps(vision_output.model_dump()),
        )

        # Send the vision output to the client.
        response = MessageOut(
            type="vision_agent_output", payload=vision_output.model_dump()
        )
        await self.send_callback(response)

        # Save the entire history to a file
        self.history.save()

        # Notify the client that the server is ready for the next image.
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
        self.logger.info(f"Primitive '{primitive_name}' completed.")
        if (
            self.primitive_in_execution
            and primitive_name == self.primitive_in_execution.name
        ):
            self.primitive_in_execution = None
        else:
            raise ValueError(
                f"[Brain {self.connection_id}] Primitive '{primitive_name}' is not the current primitive in execution. That's a weird bug."
            )

    async def handle_primitive_activated(self, message: MessageIn):
        """
        Handle messages of type 'primitive_activated'.
        Processes the primitive activation and sends an acknowledgment.
        """
        primitive_name = message.payload["primitive_name"]
        self.logger.info(
            f"\033[92m[Brain {self.connection_id}] Primitive '{primitive_name}' activated.\033[0m"
        )

        # Check if this is a navigate_to_position primitive that was derived from navigate_in_sight
        if (
            primitive_name == "navigate_to_position"
            and self.primitive_in_execution
            and self.primitive_in_execution.name == "navigate_in_sight"
        ):
            # This is a special case - we're actually executing a navigate_to_position that was
            # derived from a navigate_in_sight request, so we don't need to do anything
            pass
        else:
            # Normal case - just set the primitive_in_execution based on the primitive_name
            matched_prim = next(
                (prim for prim in self.primitives_list if prim.name == primitive_name),
                None,
            )
            if matched_prim is not None:
                # Convert the dict to a PrimitiveDefinition instance
                self.primitive_in_execution = primitive_to_object(matched_prim)
            else:
                self.primitive_in_execution = None

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

    async def handle_register_primitives_and_directive(self, message: MessageIn):
        """
        Handle messages of type 'register_primitives_and_directive'.
        Registers new primitives and directive provided by the client.
        """
        primitives_data = message.payload.get("primitives", [])
        new_directive = message.payload.get("directive")
        registered_count = 0
        directive_registered = False

        # Process primitives
        for primitive_data in primitives_data:
            try:
                name = primitive_data.get("name")
                guideline = primitive_data.get("guideline")
                inputs = primitive_data.get("inputs", {})

                # Validate required fields
                if not name:
                    self.logger.error(
                        f"Primitive registration missing required 'name' field: {primitive_data}"
                    )
                    continue

                # Check if a primitive with this name already exists in the local list
                existing_primitive = next(
                    (p for p in self.local_primitives_list if p.name == name), None
                )
                if existing_primitive:
                    self.logger.info(f"Primitive '{name}' already registered, skipping")
                    continue

                new_primitive = PrimitiveDefinition(
                    name=name, guideline=guideline, inputs=inputs
                )
                self.primitives_list.append(new_primitive)
                registered_count += 1
                self.logger.info(f"Registered new primitive: {name}")

            except Exception as e:
                self.logger.error(f"Error registering primitive: {e}")

        # Process directive if provided
        if new_directive is not None:
            try:
                old_directive = self.directive
                self.directive = new_directive
                directive_registered = True
                self.logger.info(f"Registered directive: {new_directive}")

                # Record the directive change in history
                if old_directive is None:
                    history_message = f"Directive set to '{new_directive}'"
                else:
                    history_message = (
                        f"Directive changed from '{old_directive}' to '{new_directive}'"
                    )

                self.history.add(
                    HistoryEntryType.SYSTEM_MESSAGE, description=history_message
                )
            except Exception as e:
                self.logger.error(f"Error registering directive: {e}")

        # Acknowledge the registration
        response = MessageOut(
            type=MessageOutType.PRIMITIVES_AND_DIRECTIVE_REGISTERED,
            payload={
                "success": True,
                "count": registered_count,
                "directive_registered": directive_registered,
                "message": f"Successfully registered {registered_count} primitives and {'a' if directive_registered else 'no'} directive.",
            },
        )
        await self.send_callback(response)

        self.logger.info(
            f"Registered {registered_count} primitives and {'a' if directive_registered else 'no'} directive."
        )

    async def stop(self):
        """
        Stop the brain by flagging running=False and enqueueing a None message to exit the loop.
        """
        self.running = False
        await self.message_queue.put(None)
