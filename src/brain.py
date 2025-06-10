import asyncio
import json
import time
from typing import Dict, Optional
import uuid


from src.baml_client.partial_types import VisionAgentOutput
from src.message_types import (
    MessageIn,
    MessageInType,
    MessageOut,
    MessageOutType,
)
from src.primitives.transforms import primitive_to_object
from src.history.history import History, HistoryEntryType
from src.agents.types import PrimitiveDefinition
from src.primitives.navigate_in_sight import NavigateInSight
from src.primitives.navigate_through_memory import NavigateThroughMemory
from src.primitives.turn_and_move import TurnAndMove
from src.primitives.check_distance_and_orientation import CheckDistanceAndOrientation

from src.brain_utils.logger import BrainLogger
from src.brain_utils.image_processor import ImageProcessor
from src.brain_utils.vision_service import VisionAgentType, VisionService
from src.brain_utils.navigation_handler import NavigationHandler
from src.brain_utils.memory_state_manager import MemoryStateManager


from src.constants_robots import ROBOT_PARAMS_TO_USE
from src.primitives.types import Primitive


AVERAGE_POS_COV_THRESHOLD = ROBOT_PARAMS_TO_USE["average_pos_cov_threshold"]
AVERAGE_YAW_COV_THRESHOLD = ROBOT_PARAMS_TO_USE["average_yaw_cov_threshold"]


def prim_list_to_prim_obj_list(prim_list):
    return [primitive_to_object(prim) for prim in prim_list]


class Brain:
    def __init__(
        self,
        connection_id: str,
        send_callback,
        enable_memory_commands: bool = False,
        max_recent_generic_images: int = 3,
        max_recent_pre_action_images: int = 3,
    ):
        """
        connection_id: an identifier for this brain instance (for logging/debugging)
        send_callback: an async function to send a response back to the client.
        enable_memory_commands: whether to enable memory state save/load/list commands
        max_recent_generic_images: max generic images in multimodal history
        max_recent_pre_action_images: max pre-action images in multimodal history
        """
        self.connection_id = connection_id
        self.send_callback = send_callback
        self.message_queue = asyncio.Queue()
        self.running = True
        # Flag to override the next vision output (set via a chat_in command)
        self.forward_command_active = False
        # Store the latest user message that should be consumed once by the
        # visual language model.
        self.latest_user_message = None
        self.primitives_list = []
        # Whether memory state commands are enabled
        self.enable_memory_commands = enable_memory_commands
        # Current Gemini agent variant to use
        self.gemini_variant = "gemini1"

        self.local_primitives_list = [
            NavigateInSight(),
            NavigateThroughMemory(),
            TurnAndMove(),
            CheckDistanceAndOrientation(),
        ]  # These are the ones defined in the brain here, not registered with
        # the server by the user
        for p in self.local_primitives_list:
            p.set_feedback_callback(
                lambda msg: self._handle_primitive_feedback(p.name, msg)
            )
        self.primitive_in_execution = None
        self.primitive_ids_map: Dict[str, PrimitiveDefinition] = {}
        self.directive = None  # Store the directive that will steer the VLM

        # Initialize history to record chat messages and vision agent outputs.
        # The History class defaults to MAX_MULTIMODAL_IMAGES = 3.
        # To override, instantiate with: History(max_multimodal_images=N)
        self.history = History(
            max_recent_generic_images=max_recent_generic_images,
            max_recent_pre_action_images=max_recent_pre_action_images,
        )

        # Initialize logger and helper modules
        self.logger = BrainLogger(connection_id)
        self.image_processor = ImageProcessor(self.logger)
        self.vision_service = VisionService(self.logger)
        self.navigation_handler = NavigationHandler(
            self.logger,
            self.local_primitives_list,
        )

        # Initialize memory state manager if commands are enabled
        self.memory_state_manager = None
        if self.enable_memory_commands:
            self.memory_state_manager = MemoryStateManager(self.logger, connection_id)

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

            # If the message is not a pose_image, log the time it takes to process
            # the message
            FREQUENT_MESSAGES_TO_NOT_LOG = [
                MessageInType.POSE_IMAGE,
            ]

            if message_type not in FREQUENT_MESSAGES_TO_NOT_LOG:
                self.logger.info(f"Processing message: {message_type}")
                time_start = time.time()

            if message_type == MessageInType.IMAGE:
                vision_output = await self.handle_image(message)
                task_and_id = (
                    (
                        f"{vision_output.next_task.name} "
                        f"(id: {vision_output.next_task.primitive_id})"
                    )
                    if vision_output.next_task
                    else "None"
                )
                self.logger.info(
                    f"Processed image message in {time.time() - time_start} seconds, "
                    f"sent task: {task_and_id}\n"
                )
            elif message_type == MessageInType.POSE_IMAGE:
                await self.handle_pose_image(message)
            elif message_type == MessageInType.CHAT_IN:
                await self.handle_chat_in(message)
            elif message_type == MessageInType.PRIMITIVE_COMPLETED:
                await self.handle_primitive_completed(message)
            elif message_type == MessageInType.PRIMITIVE_ACTIVATED:
                await self.handle_primitive_activated(message)
            elif message_type == MessageInType.PRIMITIVE_FAILED:
                await self.handle_primitive_failed(message)
            elif message_type == MessageInType.PRIMITIVE_INTERRUPTED:
                await self.handle_primitive_interrupted(message)
            elif message_type == MessageInType.PRIMITIVE_FEEDBACK:
                await self.handle_primitive_feedback(message)
            elif message_type == MessageInType.REGISTER_PRIMITIVES_AND_DIRECTIVE:
                await self.handle_register_primitives_and_directive(message)
            elif message_type == MessageInType.RESET:
                await self.handle_reset(message)
            else:
                await self.handle_unknown(message)

        except Exception as e:
            import traceback

            self.logger.error(
                f"Error processing message: {e}\n{traceback.format_exc()}"
            )

    async def handle_image(self, message: MessageIn):
        """Handle messages of type 'image'."""
        # Extract data from payload

        self.logger.debug(f"Received image message: {message.payload.keys()}")
        (
            base64_img_extracted,
            depth_payload,
            robot_coords,
            map_payload,
            additional_image_data,
        ) = self.image_processor.extract_image_data(message.payload)

        current_image_for_vlm: str
        try:
            current_image_for_vlm = self.image_processor.ensure_jpeg_format(
                base64_img_extracted
            )
            if additional_image_data:
                additional_image_for_vlm = self.image_processor.ensure_jpeg_format(
                    additional_image_data["image_b64"]
                )
                additional_image_data["image_b64"] = additional_image_for_vlm
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
            self.image_processor.process_depth(depth_payload)
        if map_payload:
            self.image_processor.process_map_with_robot(map_payload, robot_coords)

        # The image is now stored in current_image_for_vlm
        # It will be added to history after the VLM call with the correct type.

        local_primitives_list = prim_list_to_prim_obj_list(self.local_primitives_list)

        vision_output = await self.vision_service.call_visual_language_model(
            base64_img=current_image_for_vlm,  # Use local variable
            user_prompt_text=self.latest_user_message,
            primitive_in_execution=self.primitive_in_execution,
            primitives_list=local_primitives_list + self.primitives_list,
            history=self.history.get_as_multimodal_list(),
            robot_coords=robot_coords,
            directive=self.directive,
            agent_type=VisionAgentType.GEMINI_FLASH_MULTI,
            gemini_variant=self.gemini_variant,
            additional_image_data=additional_image_data,
        )

        # Log the image with appropriate type AFTER VLM call
        if current_image_for_vlm:
            if vision_output and vision_output.next_task:
                self.history.add(
                    HistoryEntryType.IMAGE_PRE_ACTION,
                    description=current_image_for_vlm,
                )
            else:
                self.history.add(
                    HistoryEntryType.GENERIC_IMAGE,
                    description=current_image_for_vlm,
                )

        if not vision_output:
            self.logger.error(
                f"No vision output received for connection {self.connection_id}"
            )
            vision_output = VisionAgentOutput(
                stop_current_task=True,
                observation="The brain failed, so it stopped the current task.",
                thoughts="The brain failed, so it stopped the current task.",
                new_goal=None,
                next_task=None,
                anticipation=None,
                to_tell_user=(
                    "BEEP BOOP BEEP BOOP, the brain failed. "
                    "Stopping the current task."
                ),
            )

        # Validate the next task
        vision_output.next_task = (
            PrimitiveDefinition.model_validate(vision_output.next_task)
            if vision_output.next_task
            else None
        )

        # Look for discrepancies in the vision output
        # Could be a function later
        # Potential discrepancy 1: There's a primitive running, the VLM does not say
        # stop_current_task and yet it returns a next_task
        if (
            not vision_output.stop_current_task
            and vision_output.next_task is not None
            and self.primitive_in_execution is not None
        ):
            self.history.record_discrepancy(
                message=(
                    f"The VLM returned a next_task ({vision_output.next_task.name}) "
                    f"even though there is a task running "
                    f"({self.primitive_in_execution.name}) and it did not say to "
                    f"stop the current task."
                )
            )
            # For now, we force the next_task to be None if it's not strictly asked
            # to be stopped.
            vision_output.next_task = None

        # Clear the user message as it's been consumed
        self.latest_user_message = None

        # Update primitive_in_execution if needed
        if vision_output.next_task:
            # Make sure next_task has a primitive_id
            if not vision_output.next_task.primitive_id:
                vision_output.next_task.primitive_id = str(uuid.uuid4())

        # Before replacing what we send to the client, we store it locally
        # as it will be used to write to the history.
        # TODO: For benchmarking it might make sense to also send this one tothe client
        # so that the benchmark is aware of the navigation choices.
        vision_output_to_write_in_history = None

        # Handle special case for navigate_in_sight
        if (
            vision_output.next_task
            and vision_output.next_task.name == "navigate_in_sight"
        ):
            vision_output_to_write_in_history = vision_output.model_copy()
            vision_output, has_canceled_task = (
                await self.navigation_handler.handle_navigate_in_sight(
                    vision_output,
                    robot_coords,
                    base64_img_extracted,
                    depth_payload,
                    map_payload,
                )
            )
            if has_canceled_task:
                vision_output_to_write_in_history = vision_output.model_copy()

        # Handle special case for navigate_through_memory
        if (
            vision_output.next_task
            and vision_output.next_task.name == "navigate_through_memory"
        ):
            vision_output_to_write_in_history = vision_output.model_copy()
            vision_output, has_canceled_task = (
                await self.navigation_handler.handle_navigate_through_memory(
                    vision_output, self.connection_id, map_payload
                )
            )
            if has_canceled_task:
                vision_output_to_write_in_history = vision_output.model_copy()

        # Handle special case for turn_and_move
        if vision_output.next_task and vision_output.next_task.name == "turn_and_move":
            vision_output_to_write_in_history = vision_output.model_copy()
            vision_output, has_canceled_task = (
                await self.navigation_handler.handle_turn_and_move(
                    vision_output, robot_coords, map_payload
                )
            )
            if has_canceled_task:
                vision_output_to_write_in_history = vision_output.model_copy()

        # Handle special case for check_distance_and_orientation
        if (
            vision_output.next_task
            and vision_output.next_task.name == "check_distance_and_orientation"
        ):
            vision_output_to_write_in_history = vision_output.model_copy()
            vision_output, has_canceled_task = (
                await self.navigation_handler.handle_check_distance_and_orientation(
                    vision_output,
                    robot_coords,
                    base64_img_extracted,
                    depth_payload,
                    map_payload,
                )
            )
            # We should also mark this primitive as activated and then completed
            self.history.add(
                HistoryEntryType.TASK_ACTIVATED,
                description=f"Primitive {vision_output.next_task.name} activated",
            )
            self.history.add(
                HistoryEntryType.TASK_COMPLETED,
                description=f"Primitive {vision_output.next_task.name} completed",
            )
            vision_output.next_task = None

        # Send response and prepare for next image
        await self._send_vision_output(vision_output, vision_output_to_write_in_history)

        self.history.check_and_summarize()

        return vision_output

    async def handle_pose_image(self, message: MessageIn):
        """Handle messages of type 'pose_image'."""
        # Extract data from payload
        base64_img = message.payload.get("image", "")
        x = message.payload.get("x", 0.0)
        y = message.payload.get("y", 0.0)
        theta = message.payload.get("theta", 0.0)

        cov_x = message.payload.get("cov_x", 0.0)
        cov_y = message.payload.get("cov_y", 0.0)
        cov_yaw = message.payload.get("cov_yaw", 0.0)

        # Now we receive here cov_x, cov_y, cov_yaw
        # If they are above a certain threshold, we should not add the image to the pose graph
        # because it means we don't know where the robot is and we want to avoid
        # adding wrong nodes to the pose graph
        if (
            cov_x + cov_y
        ) / 2 > AVERAGE_POS_COV_THRESHOLD or cov_yaw > AVERAGE_YAW_COV_THRESHOLD:
            self.logger.debug(
                f"Skipping image addition to pose graph because cov_x, cov_y, cov_yaw are too high: {cov_x}, {cov_y}, {cov_yaw}"
            )
            return

        # Update current robot coordinates
        self.current_robot_coords = {"x": x, "y": y, "theta": theta}

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
                    "Skipping image addition to pose graph because a close node exists"
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

    async def _send_vision_output(
        self, vision_output, vision_output_to_write_in_history=None
    ):
        primitive_to_remember = (
            vision_output_to_write_in_history.next_task
            if vision_output_to_write_in_history
            else vision_output.next_task
        )
        if primitive_to_remember:
            self.primitive_ids_map[primitive_to_remember.primitive_id] = (
                primitive_to_remember
            )
        # Record the vision agent output in the history.
        self.history.add(
            HistoryEntryType.VISION_AGENT_OUTPUT,
            description=(
                json.dumps(vision_output_to_write_in_history.model_dump())
                if vision_output_to_write_in_history
                else json.dumps(vision_output.model_dump())
            ),
        )

        # Send the vision output to the client.
        response = MessageOut(
            type="vision_agent_output", payload=vision_output.model_dump()
        )
        await self.send_callback(response)

        self.history.save()

        # Only notify the client that the server is ready for the next image if
        # there's no next task.
        # If there is a next task, we'll wait for the primitive_activated message
        # before requesting next image.
        if not vision_output.next_task:
            await self.send_callback(MessageOut(type="ready_for_image", payload={}))

    async def handle_chat_in(self, message: MessageIn):
        """
        Handle messages of type 'chat_in'.
        Echoes back the text received and processes special commands.

        Special commands (if enabled):
        - !save_memory NAME: Saves the current memory state
        - !load_memory NAME: Loads a saved memory state
        - !list_memory: Lists available memory states

        Other commands (always enabled):
        - !gemini VERSION: Switches the Gemini version used by the vision agent
                          Valid versions: gemini1, gemini2, gemini3, gemini4
        """
        text = message.payload["text"]

        # Handle Gemini version switch command (always enabled)
        if text.startswith("!gemini"):
            parts = text.split(maxsplit=1)
            if len(parts) > 1:
                requested_variant = parts[1].strip().lower()
                valid_variants = ["gemini1", "gemini2", "gemini3", "gemini4"]

                if requested_variant in valid_variants:
                    old_variant = self.gemini_variant
                    self.gemini_variant = requested_variant

                    response_text = (
                        f"Gemini variant switched from '{old_variant}' "
                        f"to '{requested_variant}'"
                    )
                    self.logger.info(response_text)

                    # Add to history
                    self.history.add(
                        HistoryEntryType.SYSTEM_MESSAGE,
                        description=response_text,
                    )

                    await self.send_callback(
                        MessageOut(type="chat_out", payload={"text": response_text})
                    )
                else:
                    response_text = (
                        f"Invalid Gemini variant: '{requested_variant}'. "
                        f"Valid options are: {', '.join(valid_variants)}"
                    )
                    await self.send_callback(
                        MessageOut(type="chat_out", payload={"text": response_text})
                    )
                return
            else:
                response_text = (
                    f"Current Gemini variant: '{self.gemini_variant}'\n"
                    f"To change, use: !gemini VERSION\n"
                    f"Valid versions: gemini1, gemini2, gemini3, gemini4"
                )
                await self.send_callback(
                    MessageOut(type="chat_out", payload={"text": response_text})
                )
                return

        # Check for special commands if memory commands are enabled
        if self.enable_memory_commands and self.memory_state_manager is not None:
            if text.startswith("!save_memory"):
                parts = text.split(maxsplit=1)
                state_name = parts[1] if len(parts) > 1 else ""

                # Find the NavigateThroughMemory primitive
                navigate_through_memory = next(
                    (
                        p
                        for p in self.local_primitives_list
                        if p.name == "navigate_through_memory"
                    ),
                    None,
                )

                success = await self.memory_state_manager.save_memory_state(
                    state_name, self.history, navigate_through_memory
                )

                # Send response to user
                if success:
                    response_text = f"Memory state '{state_name}' saved successfully"
                else:
                    response_text = f"Failed to save memory state '{state_name}'"

                await self.send_callback(
                    MessageOut(type="chat_out", payload={"text": response_text})
                )
                return

            elif text.startswith("!load_memory"):
                parts = text.split(maxsplit=1)
                if len(parts) > 1:
                    state_name = parts[1]

                    # Reset state variables but preserve Gemini variant
                    self.latest_user_message = None
                    self.directive = None
                    self.primitive_in_execution = None
                    # We explicitly don't reset self.gemini_variant here to preserve it

                    # Find the NavigateThroughMemory primitive
                    navigate_through_memory = next(
                        (
                            p
                            for p in self.local_primitives_list
                            if p.name == "navigate_through_memory"
                        ),
                        None,
                    )

                    success = await self.memory_state_manager.load_memory_state(
                        state_name, self.history, navigate_through_memory
                    )

                    # Send response to user
                    if success:
                        response_text = (
                            f"Memory state '{state_name}' loaded successfully"
                        )
                    else:
                        response_text = f"Failed to load memory state '{state_name}'"

                    await self.send_callback(
                        MessageOut(type="chat_out", payload={"text": response_text})
                    )
                    return
                else:
                    await self.send_callback(
                        MessageOut(
                            type="chat_out",
                            payload={
                                "text": "Please specify a memory state name to load"
                            },
                        )
                    )
                    return

            elif text.startswith("!list_memory"):
                # Get list of available memory states
                states = self.memory_state_manager.get_available_states()

                if states:
                    states_list = "\n- " + "\n- ".join(states)
                    response_text = f"Available memory states:{states_list}"
                else:
                    response_text = "No memory states available"

                await self.send_callback(
                    MessageOut(type="chat_out", payload={"text": response_text})
                )
                return

        # Handle memory command attempt when disabled or manager is None
        if text.startswith("!"):
            memory_commands = ["!save_memory", "!load_memory", "!list_memory"]
            if any(text.startswith(cmd) for cmd in memory_commands):
                response_text = (
                    "Memory management commands are disabled. "
                    "They can be enabled when starting the brain."
                )
                await self.send_callback(
                    MessageOut(type="chat_out", payload={"text": response_text})
                )
                return

        # Save the latest user message for processing.
        self.latest_user_message = text

        # Record this chat message in the history.
        self.history.add(HistoryEntryType.AUDIO_IN, description=text)

    async def handle_primitive_completed(self, message: MessageIn):
        """
        Handle messages of type 'primitive_completed'.
        Processes the primitive completion and sends an acknowledgment.
        """
        primitive_id = message.payload["primitive_id"]
        primitive_name = message.payload["primitive_name"]

        # Check if we have a matching primitive in execution
        if (
            self.primitive_in_execution
            and primitive_id == self.primitive_in_execution.primitive_id
        ):
            self.logger.info(
                f"Task '{self.primitive_in_execution.name}' "
                f"(ID: {primitive_id}) completed."
            )
            # Use system message type for completion
            self.history.add(
                HistoryEntryType.TASK_COMPLETED,
                description=f"Task '{self.primitive_in_execution.name}' completed.",
            )
            self.primitive_in_execution = None
        else:
            task_id_msg = f"Task '{primitive_name}' (ID: {primitive_id})"
            raise ValueError(
                f"[Brain {self.connection_id}] {task_id_msg} is not the current "
                f"task in execution."
            )

    async def handle_primitive_failed(self, message: MessageIn):
        """
        Handle messages of type 'primitive_failed'.
        Processes the primitive failure and sends an acknowledgment.
        """
        primitive_id = message.payload["primitive_id"]
        primitive_name = message.payload["primitive_name"]
        if (
            self.primitive_in_execution
            and self.primitive_in_execution.primitive_id == primitive_id
        ):
            task_name = self.primitive_in_execution.name
            self.logger.info(f"Task '{task_name}' failed.")

            # Use task_cancelled type for failed tasks
            self.history.add(
                HistoryEntryType.TASK_CANCELLED,
                description=f"Task '{task_name}' failed.",
            )
            self.primitive_in_execution = None
        else:
            task_id_msg = f"Task '{primitive_name}' (ID: {primitive_id})"
            raise ValueError(
                f"[Brain {self.connection_id}] {task_id_msg} is not the current "
                f"task in execution."
            )

    async def handle_primitive_interrupted(self, message: MessageIn):
        """
        Handle messages of type 'primitive_interrupted'.
        Processes the primitive interruption and sends an acknowledgment.
        """
        primitive_id = message.payload["primitive_id"]
        primitive_name = message.payload["primitive_name"]

        if (
            self.primitive_in_execution
            and self.primitive_in_execution.primitive_id == primitive_id
        ):
            task_name = self.primitive_in_execution.name
            self.logger.info(f"Task '{task_name}' interrupted.")
            self.history.add(
                HistoryEntryType.TASK_INTERRUPTED,
                description=f"Task '{task_name}' interrupted.",
            )
            self.primitive_in_execution = None
        else:
            task_id_msg = f"Task '{primitive_name}' (ID: {primitive_id})"
            raise ValueError(
                f"[Brain {self.connection_id}] {task_id_msg} is not the current "
                f"task in execution."
            )

    async def handle_primitive_feedback(self, message: MessageIn):
        """
        Handle messages of type 'primitive_feedback'.
        Retrieves the feedback string and adds it to the history.
        """
        feedback_text = message.payload.get("feedback")
        if feedback_text:
            self.logger.info(f"Received primitive feedback: {feedback_text}")
            task_name = self.primitive_in_execution.name
            entry_text = f"'{task_name}': {feedback_text}"
            self.history.add(
                HistoryEntryType.TASK_FEEDBACK,
                description=entry_text,
            )
        else:
            self.logger.warning(
                "Received primitive_feedback message with no feedback text."
            )

    async def handle_primitive_activated(self, message: MessageIn):
        """
        Handle messages of type 'primitive_activated'.
        Processes primitive activation and sends acknowledgment.
        Requests next image after activation confirmation to prevent race conditions.
        """
        primitive_id = message.payload["primitive_id"]
        primitive_activated = self.primitive_ids_map[primitive_id]

        # Check if this is a navigate_to_position primitive from navigate_in_sight
        if self.primitive_in_execution:
            # The client has activated a primitive we didn't decide to activate.
            task_name = self.primitive_in_execution.name
            self.logger.warn(
                f"[Brain {self.connection_id}] Task '{task_name}' (ID: {primitive_id}) "
                f"was activated by the client, but we didn't activate it."
            )
        else:
            # Normal case - check it corresponds to a primitive in our list
            task_name = primitive_activated.name
            self.logger.info(
                f"\033[92m[Brain {self.connection_id}] Task '{task_name}' "
                f"(ID: {primitive_id}) activated.\033[0m"
            )
            matched_prim = next(
                (
                    prim
                    for prim in self.primitives_list + self.local_primitives_list
                    if prim.name == task_name
                ),
                None,
            )
            if matched_prim is not None:
                # Convert the dict to a PrimitiveDefinition instance with the ID
                prim_obj = (
                    primitive_to_object(matched_prim)
                    if isinstance(matched_prim, Primitive)
                    else matched_prim
                )
                # Override the ID if provided in the message
                if primitive_id:
                    prim_obj.primitive_id = primitive_id
                self.primitive_in_execution = prim_obj
                self.history.add(
                    HistoryEntryType.TASK_ACTIVATED,
                    description=f"Task '{task_name}' activated.",
                )
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

    async def handle_reset(self, message: MessageIn):
        """
        Handle messages of type 'reset'.
        Resets brain state: history, pose graph memory, and state variables.
        If a memory_state parameter is provided and memory commands are enabled,
        loads that state.
        """
        self.logger.info(f"Resetting brain state for connection {self.connection_id}")

        # Check if a memory state was provided in the payload
        memory_state = message.payload.get("memory_state")

        if (
            memory_state
            and self.enable_memory_commands
            and self.memory_state_manager is not None
        ):
            # Reset state variables but preserve Gemini variant
            self.latest_user_message = None
            self.directive = None
            self.primitive_in_execution = None
            # We explicitly don't reset self.gemini_variant here to preserve it

            # Find the NavigateThroughMemory primitive
            navigate_through_memory = next(
                (
                    p
                    for p in self.local_primitives_list
                    if p.name == "navigate_through_memory"
                ),
                None,
            )

            # Load the specified memory state
            success = await self.memory_state_manager.load_memory_state(
                memory_state, self.history, navigate_through_memory
            )

            if success:
                self.logger.info(
                    f"Loaded memory state '{memory_state}' for "
                    f"connection {self.connection_id}"
                )
                # Notify the client that the server is ready for the next image
                await self.send_callback(
                    MessageOut(type=MessageOutType.READY_FOR_IMAGE, payload={})
                )
                return
            else:
                self.logger.error(
                    f"Failed to load memory state '{memory_state}', "
                    f"performing standard reset"
                )
        elif memory_state and (
            not self.enable_memory_commands or self.memory_state_manager is None
        ):
            self.logger.warning(
                f"Memory state '{memory_state}' provided, "
                f"but memory commands are disabled"
            )

        # Perform standard reset if no memory state was provided or loading failed
        # Reset history
        self.history.reset()

        # Reset latest user message
        self.latest_user_message = None

        # Reset directive
        self.directive = None

        # Reset primitive in execution
        self.primitive_in_execution = None

        # Reset Gemini variant to default
        self.gemini_variant = "gemini1"
        self.logger.info("Reset Gemini variant to default (gemini1)")

        # Reset pose graph memory for this connection
        navigate_through_memory = next(
            (
                p
                for p in self.local_primitives_list
                if p.name == "navigate_through_memory"
            ),
            None,
        )

        if navigate_through_memory:
            pose_graph_memory = navigate_through_memory.pose_graph_memory
            pose_graph_memory.reset_user_data(self.connection_id)
            self.logger.info(
                f"Reset pose graph memory for connection {self.connection_id}"
            )
        else:
            self.logger.error(
                "NavigateThroughMemory primitive not found, "
                "couldn't reset pose graph memory"
            )

        # Notify the client that the server is ready for the next image
        await self.send_callback(
            MessageOut(type=MessageOutType.READY_FOR_IMAGE, payload={})
        )

    async def handle_register_primitives_and_directive(self, message: MessageIn):
        """
        Handle messages of type 'register_primitives_and_directive'.
        Registers new primitives and directive provided by the client.
        """
        primitives_data = message.payload.get("primitives", [])
        new_directive = message.payload.get("directive")
        registered_count = 0
        directive_registered = False

        # Clean up the primitives list
        self.primitives_list = []

        # Process primitives
        for primitive_data in primitives_data:
            try:
                name = primitive_data.get("name")
                guidelines = primitive_data.get("guidelines")
                guidelines_when_running = primitive_data.get("guidelines_when_running")
                inputs = primitive_data.get("inputs", {})

                # Validate required fields
                if not name:
                    self.logger.error(
                        f"Primitive registration missing required 'name' field: "
                        f"{primitive_data}"
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
                    name=name,
                    guidelines=guidelines,
                    guidelines_when_running=guidelines_when_running,
                    inputs=inputs,
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
                "message": (
                    f"Successfully registered {registered_count} primitives and "
                    f"{'a' if directive_registered else 'no'} directive."
                ),
            },
        )
        await self.send_callback(response)

        self.logger.info(
            f"Registered {registered_count} primitives and "
            f"{'a' if directive_registered else 'no'} directive."
        )

    async def stop(self):
        """
        Stop the brain by flagging running=False and enqueueing a None message to
        exit the loop.
        """
        self.running = False
        await self.message_queue.put(None)

    def _handle_primitive_feedback(self, primitive_name: str, feedback_message: str):
        """
        Handles feedback from a primitive, called directly from the primitive.
        """
        if feedback_message:
            self.logger.info(
                f"Received primitive feedback from '{primitive_name}': {feedback_message}"
            )
            entry_text = f"'{primitive_name}': {feedback_message}"
            self.history.add(
                HistoryEntryType.TASK_FEEDBACK,
                description=entry_text,
            )
        else:
            self.logger.warning(
                f"Received empty feedback message from primitive '{primitive_name}'."
            )
