# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Innate Inc

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Dict
import traceback

from src.logging.bigquery_logger import BigQueryLogger

from src.message_types import (
    MessageIn,
    MessageInType,
    MessageOut,
    MessageOutType,
)
from src.history.history import History, HistoryEntryType
from src.agents.types import PrimitiveDefinition
from src.primitives.navigate_in_sight import NavigateInSight
from src.primitives.nav_insight_continuous import NavInsightContinuous
from src.primitives.navigate_through_memory import NavigateThroughMemory
from src.primitives.turn_and_move import TurnAndMove

from src.brain_utils.constants import PrimitiveNames
from src.brain_utils.logger import BrainLogger
from src.brain_utils.image_processor import ImageProcessor
from src.brain_utils.vision_service import VisionService
from src.brain_utils.navigation_handler import NavigationHandler
from src.brain_utils.memory_state_manager import MemoryStateManager
from src.brain_utils.image_handler import ImageHandler
from src.brain_utils.chat_handler import ChatHandler
from src.brain_utils.primitive_handler import PrimitiveHandler
from src.brain_utils.pose_graph_handler import PoseGraphHandler

DEFAULT_GEMINI_VARIANT = "gemini-flash"


@dataclass
class BrainState:
    """Encapsulates the mutable state of the Brain."""

    latest_user_message: str | None = None
    primitive_in_execution: PrimitiveDefinition | None = None
    # Whether the robot confirmed (primitive_activated) the primitive that
    # primitive_in_execution tracks. The image loop is normally re-armed by
    # the activation handler; a primitive that reaches a terminal state
    # WITHOUT ever activating (the robot refused or failed to start it —
    # report_start_failure paths) must have its terminal handler re-arm the
    # loop instead, or no image request is left outstanding.
    primitive_activation_seen: bool = False
    directive: str | None = None
    gemini_variant: str = DEFAULT_GEMINI_VARIANT
    primitives_list: list = field(default_factory=list)
    primitive_ids_map: Dict[str, PrimitiveDefinition] = field(default_factory=dict)
    slow_agent_running: bool = False  # Skip image processing while slow agent is running

    def reset(self, preserve_gemini_variant: bool = False):
        """Reset state to initial values."""
        saved_variant = (
            self.gemini_variant if preserve_gemini_variant else DEFAULT_GEMINI_VARIANT
        )
        self.latest_user_message = None
        self.primitive_in_execution = None
        self.primitive_activation_seen = False
        self.directive = None
        self.gemini_variant = saved_variant
        self.slow_agent_running = False
        # Note: primitives_list and primitive_ids_map are not reset here
        # as they are managed separately


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
        self.enable_memory_commands = enable_memory_commands

        self.state = BrainState()
        self.current_robot_coords = {}

        # Initialize history to record chat messages and vision agent outputs.
        # The History class defaults to MAX_MULTIMODAL_IMAGES = 3.
        # To override, instantiate with: History(max_multimodal_images=N)
        self.history = History(
            max_recent_generic_images=max_recent_generic_images,
            max_recent_pre_action_images=max_recent_pre_action_images,
        )

        # Local primitives defined in the brain (not registered by user)
        self.local_primitives_list = [
            NavigateInSight(),
            NavInsightContinuous(),
            NavigateThroughMemory(),
            TurnAndMove(),
        ]
        # Set up feedback callbacks after primitive_handler is initialized (see below)
        self._pending_feedback_callbacks = [
            (p, p.name) for p in self.local_primitives_list
        ]

        # Initialize logger and helper modules
        self.logger = BrainLogger(connection_id)

        # Initialize BigQuery logger
        self.bq_logger = BigQueryLogger()

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

        # Initialize the image handler with chat callback for fast agent responses
        self.image_handler = ImageHandler(
            logger=self.logger,
            image_processor=self.image_processor,
            vision_service=self.vision_service,
            navigation_handler=self.navigation_handler,
            history=self.history,
            local_primitives_list=self.local_primitives_list,
            send_chat_callback=self._send_chat_out,
        )

        # Initialize chat handler
        self.chat_handler = ChatHandler(
            logger=self.logger,
            history=self.history,
            local_primitives_list=self.local_primitives_list,
            send_callback=self.send_callback,
            vision_service=self.vision_service,
            memory_state_manager=self.memory_state_manager,
        )

        # Initialize primitive handler
        self.primitive_handler = PrimitiveHandler(
            logger=self.logger,
            history=self.history,
            primitives_list=self.state.primitives_list,
            local_primitives_list=self.local_primitives_list,
            primitive_ids_map=self.state.primitive_ids_map,
            send_callback=self.send_callback,
            connection_id=self.connection_id,
        )

        # Now set up feedback callbacks (primitive_handler must exist)
        for p, name in self._pending_feedback_callbacks:
            p.set_feedback_callback(
                lambda msg, n=name: self.primitive_handler.handle_internal_feedback(
                    n, msg
                )
            )
        del self._pending_feedback_callbacks

        # Initialize pose graph handler
        self.pose_graph_handler = PoseGraphHandler(
            logger=self.logger,
            local_primitives_list=self.local_primitives_list,
            connection_id=self.connection_id,
        )

    async def enqueue_message(self, message: MessageIn):
        """
        Called externally (by your websocket connection handler) to push
        messages into the brain for processing.

        CHAT_IN messages are processed immediately in a separate task to avoid
        waiting behind slow IMAGE processing. This ensures fast response times
        for user questions.
        """
        self.logger.debug(f"[Brain] Enqueue message type={message.type}, queue_size={self.message_queue.qsize()}")
        # Process CHAT_IN immediately without waiting for the queue
        if message.type == MessageInType.CHAT_IN:
            # Fire and forget - process chat in parallel
            self.logger.debug(f"[Brain] CHAT_IN fast path - processing immediately")
            asyncio.create_task(self._process_chat_immediately(message))
        else:
            await self.message_queue.put(message)

    async def _process_chat_immediately(self, message: MessageIn):
        """
        Process a CHAT_IN message immediately, bypassing the queue.
        This ensures fast response times for user questions.
        """
        try:
            self.logger.info(f"Processing message: {message.type} (fast path)")
            await self.handle_chat_in(message)
        except Exception as e:
            self.logger.error(
                f"Error processing chat message: {e}\n{traceback.format_exc()}"
            )

    async def run(self):
        """
        The brain's main loop. It runs on a single thread (event loop), processes
        one message at a time, and sends back results with send_callback.
        """
        self.logger.info("[Brain] Main loop started")
        while self.running:
            message = await self.message_queue.get()
            if message is None:
                self.logger.info("[Brain] Received None message, shutting down")
                break  # Allow graceful shutdown when a None message is pushed
            self.logger.debug(f"[Brain] Processing from queue: type={message.type}, queue_size={self.message_queue.qsize()}")
            await self.process_message(message)

    async def process_message(self, message: MessageIn):
        """Dispatch message to appropriate handler using match statement."""
        try:
            # Log non-frequent messages
            if message.type != MessageInType.POSE_IMAGE:
                self.logger.info(f"Processing message: {message.type}")

            match message.type:
                case MessageInType.IMAGE:
                    await self._process_image(message)
                case MessageInType.POSE_IMAGE:
                    await self.handle_pose_image(message)
                case MessageInType.CHAT_IN:
                    await self.handle_chat_in(message)
                case MessageInType.PRIMITIVE_COMPLETED:
                    await self.handle_primitive_completed(message)
                case MessageInType.PRIMITIVE_ACTIVATED:
                    await self.handle_primitive_activated(message)
                case MessageInType.PRIMITIVE_FAILED:
                    await self.handle_primitive_failed(message)
                case MessageInType.PRIMITIVE_INTERRUPTED:
                    await self.handle_primitive_interrupted(message)
                case MessageInType.PRIMITIVE_FEEDBACK:
                    await self.handle_primitive_feedback(message)
                case MessageInType.REGISTER_PRIMITIVES_AND_DIRECTIVE:
                    await self.handle_register_primitives_and_directive(message)
                case MessageInType.RESET:
                    await self.handle_reset(message)
                case _:
                    await self.handle_unknown(message)

        except Exception as e:
            self.logger.error(
                f"Error processing message: {e}\n{traceback.format_exc()}"
            )

    async def _process_image(self, message: MessageIn):
        """
        Process an image message and log the results.
        Handles both success and failure cases cleanly.
        
        If continuous navigation is active, bypasses normal VLM processing
        and routes the image directly to the NavInsightContinuous primitive.
        """
        # Skip image processing while slow agent is running (prevents duplicate primitives)
        if self.state.slow_agent_running:
            self.logger.info("Skipping image processing while slow agent is running")
            return

        # Check if continuous navigation is active
        if await self._handle_continuous_navigation_image(message):
            return  # Handled by continuous navigation
        
        time_start = time.time()
        vision_output, vlm_processing_time = await self.handle_image(message)
        time_elapsed = time.time() - time_start

        # Handle failure case
        if vision_output is None:
            self.logger.warn(f"Image processing failed after {time_elapsed:.2f}s")
            self._log_image_metrics_to_bigquery(
                vlm_processing_time=None,
                token_metrics=None,
                time_elapsed=time_elapsed,
            )
            return

        # Calculate and log success metrics
        token_metrics = self.image_handler.calculate_token_metrics(
            vision_output, time_elapsed
        )
        self._log_image_processing_result(vision_output, token_metrics, time_elapsed)
        self._log_image_metrics_to_bigquery(
            vlm_processing_time, token_metrics, time_elapsed
        )

    def _log_image_processing_result(
        self,
        vision_output,
        token_metrics,
        time_elapsed: float,
    ):
        """Log the image processing result to console."""
        task_and_id = (
            f"{vision_output.next_task.name} "
            f"(id: {vision_output.next_task.primitive_id})"
            if vision_output.next_task
            else "None"
        )
        token_info = self.image_handler.format_token_info(token_metrics)
        stop_prefix = "stop and " if vision_output.stop_current_task else ""

        self.logger.info(
            f"Processed image message in {time_elapsed:.2f} seconds{token_info}, "
            f"sent {stop_prefix}task: {task_and_id}\n"
        )

    def _log_image_metrics_to_bigquery(
        self,
        vlm_processing_time: float | None,
        token_metrics,
        time_elapsed: float,
    ):
        """Log image processing metrics to BigQuery."""
        token_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": self.state.gemini_variant,
            "decision_making_time": vlm_processing_time,
            "input_tokens": token_metrics.input_tokens if token_metrics else None,
            "output_tokens": token_metrics.output_tokens if token_metrics else None,
            "total_tokens": token_metrics.total_tokens if token_metrics else None,
            "tokens_per_second": (
                token_metrics.tokens_per_second if token_metrics else None
            ),
            "total_processing_seconds": time_elapsed,
            "connection_id": self.connection_id,
        }
        self.bq_logger.log("token_metrics", token_data, self.logger)

    async def handle_image(self, message: MessageIn):
        """Handle messages of type 'image'."""
        self.logger.debug(f"Received image message: {message.payload.keys()}")

        # Define callbacks for error handling
        async def send_error(error_text: str):
            await self.send_callback(
                MessageOut(type="error", payload={"text": error_text})
            )

        async def send_ready():
            await self.send_callback(MessageOut(type="ready_for_image", payload={}))

        # Delegate to ImageHandler for processing
        result = await self.image_handler.process_image_message(
            payload=message.payload,
            latest_user_message=self.state.latest_user_message,
            primitive_in_execution=self.state.primitive_in_execution,
            primitives_list=self.state.primitives_list,
            directive=self.state.directive,
            gemini_variant=self.state.gemini_variant,
            connection_id=self.connection_id,
            send_error_callback=send_error,
            send_ready_callback=send_ready,
        )

        if result is None:
            return None, 0.0

        processing_result, vision_output_for_history = result
        vision_output = processing_result.vision_output
        vlm_processing_time = processing_result.vlm_processing_time

        # If there was no explicit user message for this image processing,
        # clear any to_tell_user that the VLM might have generated from
        # seeing user messages in history - those are handled by the chat handler.
        if self.state.latest_user_message is None and vision_output.to_tell_user:
            self.logger.debug(
                f"Clearing to_tell_user from image processing (no explicit user message): "
                f"{vision_output.to_tell_user[:50]}..."
            )
            vision_output.to_tell_user = None

        # Clear the user message as it's been consumed
        self.state.latest_user_message = None

        # Send response and prepare for next image
        await self._send_vision_output(vision_output, vision_output_for_history)

        self.history.check_and_summarize()

        return vision_output, vlm_processing_time

    async def handle_pose_image(self, message: MessageIn):
        """Handle messages of type 'pose_image'."""
        payload = message.payload
        robot_coords, positions = self.pose_graph_handler.handle_pose_image(
            image_b64=payload.get("image", ""),
            x=payload.get("x", 0.0),
            y=payload.get("y", 0.0),
            theta=payload.get("theta", 0.0),
            cov_x=payload.get("cov_x", 0.0),
            cov_y=payload.get("cov_y", 0.0),
            cov_yaw=payload.get("cov_yaw", 0.0),
        )
        if robot_coords:
            self.current_robot_coords = robot_coords

        # Send the full pose graph positions back to the client when a node is added
        if positions:
            await self.send_callback(
                MessageOut(
                    type=MessageOutType.MEMORY_POSITIONS,
                    payload={"positions": positions},
                )
            )

    async def _send_vision_output(
        self, vision_output, vision_output_to_write_in_history=None
    ):
        primitive_to_remember = (
            vision_output_to_write_in_history.next_task
            if vision_output_to_write_in_history
            else vision_output.next_task
        )
        if primitive_to_remember:
            self.state.primitive_ids_map[primitive_to_remember.primitive_id] = (
                primitive_to_remember
            )
            # Set primitive_in_execution immediately when sending to prevent race conditions
            # where multiple primitives could be sent before the first is acknowledged
            self.state.primitive_in_execution = primitive_to_remember
            self.state.primitive_activation_seen = False
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

        # Only request next image if there's no next task
        if not vision_output.next_task:
            await self.send_callback(MessageOut(type="ready_for_image", payload={}))

    async def handle_chat_in(self, message: MessageIn):
        """
        Handle messages of type 'chat_in'.

        Fast and slow agents run in parallel. If fast agent can answer,
        response is sent immediately. If fast defers, slow agent result
        is already available and processed here.
        """
        # Set flag to skip image processing while slow agent may be running
        self.state.slow_agent_running = True
        try:
            result = await self.chat_handler.handle_chat_in(
                text=message.payload["text"],
                image_b64=message.payload.get("image_b64"),
                current_gemini_variant=self.state.gemini_variant,
                enable_memory_commands=self.enable_memory_commands,
                directive=self.state.directive,
                primitive_in_execution=self.state.primitive_in_execution,
                primitives_list=self.state.primitives_list,
                robot_coords=self.current_robot_coords,
            )
        finally:
            self.state.slow_agent_running = False

        if result.new_gemini_variant is not None:
            self.state.gemini_variant = result.new_gemini_variant

        if result.vision_output is not None:
            # Fast agent deferred - process the slow agent's vision output
            vision_output = result.vision_output
            vision_output_for_history = result.vision_output

            # Handle navigation primitives if we have the required data
            if vision_output.next_task:
                (
                    vision_output,
                    vision_output_for_history,
                ) = await self.image_handler._handle_navigation_primitives(
                    vision_output,
                    robot_coords=message.payload.get("robot_coords", self.current_robot_coords),
                    base64_img=message.payload.get("image_b64"),
                    depth_payload=message.payload.get("depth_payload"),
                    map_payload=message.payload.get("map_payload"),
                    camera_info=message.payload.get("camera_info", {}),
                    connection_id=self.connection_id,
                )

            await self._send_vision_output(vision_output, vision_output_for_history)
            self.history.check_and_summarize()
        else:
            # Fast agent answered or no vision output - request next image
            await self.send_callback(MessageOut(type="ready_for_image", payload={}))

    async def _finish_terminal_lifecycle(self, cleared: bool):
        """Housekeeping after a terminal lifecycle handler resolved (or
        ignored) a primitive.

        The image loop is re-armed by primitive_activated; a primitive that
        reaches a terminal state WITHOUT ever activating (e.g. the robot
        refused/dropped the task and answered with a terminal "failed") would
        otherwise leave no image request outstanding and the vision loop would
        stall — exactly the never-started paths of INN-711.
        """
        if not cleared:
            return
        rearm = not self.state.primitive_activation_seen
        self.state.primitive_activation_seen = False
        if rearm:
            await self.send_callback(MessageOut(type="ready_for_image", payload={}))

    async def handle_primitive_completed(self, message: MessageIn):
        """Handle primitive completion."""
        had = self.state.primitive_in_execution is not None
        self.state.primitive_in_execution = (
            await self.primitive_handler.handle_primitive_completed(
                message.payload, self.state.primitive_in_execution
            )
        )
        await self._finish_terminal_lifecycle(had and self.state.primitive_in_execution is None)

    async def handle_primitive_failed(self, message: MessageIn):
        """Handle primitive failure."""
        had = self.state.primitive_in_execution is not None
        self.state.primitive_in_execution = (
            await self.primitive_handler.handle_primitive_failed(
                message.payload, self.state.primitive_in_execution
            )
        )
        await self._finish_terminal_lifecycle(had and self.state.primitive_in_execution is None)
        # Deactivate continuous navigation if active
        nav_continuous = self._get_nav_insight_continuous_primitive()
        if nav_continuous and nav_continuous.is_active:
            nav_continuous.deactivate()
            self.logger.info("Deactivated continuous navigation due to primitive failure")

    async def handle_primitive_interrupted(self, message: MessageIn):
        """Handle primitive interruption."""
        had = self.state.primitive_in_execution is not None
        self.state.primitive_in_execution = (
            await self.primitive_handler.handle_primitive_interrupted(
                message.payload, self.state.primitive_in_execution
            )
        )
        await self._finish_terminal_lifecycle(had and self.state.primitive_in_execution is None)
        # Deactivate continuous navigation if active
        nav_continuous = self._get_nav_insight_continuous_primitive()
        if nav_continuous and nav_continuous.is_active:
            nav_continuous.deactivate()
            self.logger.info("Deactivated continuous navigation due to interruption")

    async def handle_primitive_feedback(self, message: MessageIn):
        """Handle primitive feedback."""
        self.primitive_handler.handle_primitive_feedback(
            message.payload, self.state.primitive_in_execution
        )

    async def handle_primitive_activated(self, message: MessageIn):
        """Handle primitive activation."""
        self.state.primitive_in_execution = (
            await self.primitive_handler.handle_primitive_activated(
                message.payload, self.state.primitive_in_execution
            )
        )
        # Confirmation disarms the never-confirmed watchdog: from here on the
        # primitive is genuinely running and may take as long as it takes.
        if (
            self.state.primitive_in_execution is not None
            and self.state.primitive_in_execution.primitive_id == message.payload.get("primitive_id")
        ):
            self.state.primitive_activation_seen = True

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
            self.state.reset(preserve_gemini_variant=True)

            # Load the specified memory state
            navigate_through_memory = self._get_navigate_through_memory_primitive()
            success = await self.memory_state_manager.load_memory_state(
                memory_state, self.history, navigate_through_memory
            )

            if success:
                self.logger.info(
                    f"Loaded memory state '{memory_state}' for "
                    f"connection {self.connection_id}"
                )
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
            self.logger.warn(
                f"Memory state '{memory_state}' provided, "
                f"but memory commands are disabled"
            )

        # Perform standard reset
        self.history.reset()
        self.state.reset()
        self.pose_graph_handler.reset_pose_graph()
        self.logger.info(f"Reset Gemini variant to default ({DEFAULT_GEMINI_VARIANT})")

        await self.send_callback(
            MessageOut(type=MessageOutType.READY_FOR_IMAGE, payload={})
        )

    async def handle_register_primitives_and_directive(self, message: MessageIn):
        """Handle 'register_primitives_and_directive' messages."""
        primitives_data = message.payload.get("primitives", [])
        new_directive = message.payload.get("directive")

        # Reset and process primitives (use clear() to preserve reference held by primitive_handler)
        self.state.primitives_list.clear()
        local_names = {p.name for p in self.local_primitives_list}

        for p in primitives_data:
            name = p.get("name")
            if not name:
                self.logger.error(f"Primitive missing 'name': {p}")
                continue
            if name in local_names:
                self.logger.info(f"Primitive '{name}' already exists locally, skipping")
                continue

            self.state.primitives_list.append(
                PrimitiveDefinition(
                    name=name,
                    guidelines=p.get("guidelines"),
                    guidelines_when_running=p.get("guidelines_when_running"),
                    inputs=p.get("inputs", {}),
                )
            )

        registered_count = len(self.state.primitives_list)

        # Process directive
        directive_registered = new_directive is not None
        if directive_registered:
            old_directive = self.state.directive
            directive_changed = (
                old_directive is not None and old_directive != new_directive
            )

            # Reset history if directive actually changed (not just set for the first time)
            if directive_changed:
                self.logger.info(
                    f"Directive changed, resetting conversation history for connection {self.connection_id}"
                )
                self.history.reset()

            self.state.directive = new_directive

            self.bq_logger.log(
                "directive_changes",
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "directive_name": "",
                    "directive_text": new_directive,
                    "connection_id": self.connection_id,
                },
                self.logger,
            )

            self.history.add(
                HistoryEntryType.SYSTEM_MESSAGE,
                description=(
                    f"Directive set to '{new_directive}'"
                    if old_directive is None
                    else f"Directive changed from '{old_directive}' to '{new_directive}'"
                ),
            )

        await self.send_callback(
            MessageOut(
                type=MessageOutType.PRIMITIVES_AND_DIRECTIVE_REGISTERED,
                payload={
                    "success": True,
                    "count": registered_count,
                    "directive_registered": directive_registered,
                    "message": f"Registered {registered_count} primitives and {'a' if directive_registered else 'no'} directive.",
                },
            )
        )

    async def stop(self):
        """
        Stop the brain by flagging running=False and enqueueing a None message to
        exit the loop.
        """
        self.running = False
        await self.message_queue.put(None)

    async def _send_chat_out(self, text: str):
        """Send a chat message to the client."""
        await self.send_callback(
            MessageOut(type=MessageOutType.CHAT_OUT, payload={"text": text})
        )

    def _get_navigate_through_memory_primitive(self):
        """Get the NavigateThroughMemory primitive from local primitives."""
        return next(
            (
                p
                for p in self.local_primitives_list
                if p.name == PrimitiveNames.NAVIGATE_THROUGH_MEMORY
            ),
            None,
        )

    def _get_nav_insight_continuous_primitive(self):
        """Get the NavInsightContinuous primitive from local primitives."""
        return next(
            (
                p
                for p in self.local_primitives_list
                if p.name == PrimitiveNames.NAV_INSIGHT_CONTINUOUS
            ),
            None,
        )

    async def _handle_continuous_navigation_image(self, message: MessageIn) -> bool:
        """
        Handle an image message when continuous navigation is active.
        
        When NavInsightContinuous is active, images bypass normal VLM processing
        and are routed directly to the primitive for faster navigation decisions.
        
        Args:
            message: The image message
            
        Returns:
            True if the image was handled by continuous navigation, False otherwise
        """
        nav_continuous = self._get_nav_insight_continuous_primitive()
        
        if nav_continuous is None or not nav_continuous.is_active:
            return False
        
        self.logger.info("Processing image for continuous navigation (bypassing VLM)")
        
        payload = message.payload
        
        # Extract image data
        try:
            extracted_data = self.image_processor.extract_image_data(payload)
            if extracted_data is None:
                self.logger.error("Failed to extract image data for continuous nav")
                return False
            
            (
                base64_img,
                depth_payload,
                robot_coords,
                map_payload,
                _,  # additional_image_data
                camera_info,
            ) = extracted_data
        except Exception as e:
            self.logger.error(f"Error extracting image data: {e}")
            return False
        
        # Process through navigation handler
        should_continue, navigation_command, msg = (
            await self.navigation_handler.process_continuous_navigation_image(
                robot_coords=robot_coords,
                base64_img=base64_img,
                depth_payload=depth_payload,
                map_payload=map_payload,
                camera_info=camera_info,
            )
        )
        
        self.logger.info(f"Continuous nav result: {msg}, continue={should_continue}")
        
        if should_continue and navigation_command:
            # Send navigation command to client
            from src.agents.types import PrimitiveDefinition
            
            nav_task = PrimitiveDefinition(
                name=PrimitiveNames.NAVIGATE_TO_POSITION,
                inputs={
                    "x": navigation_command["x"],
                    "y": navigation_command["y"],
                    "theta": navigation_command["theta"],
                    "local_frame": True,
                },
                primitive_id=nav_continuous._primitive_id,
            )
            
            # Create minimal vision output for navigation
            from src.agents.types import VisionAgentOutput
            
            vision_output = VisionAgentOutput(
                stop_current_task=False,
                observation=msg,
                thoughts="Continuing navigation towards objective.",
                new_goal=None,
                next_task=nav_task,
                anticipation="Will continue navigating after this movement.",
                to_tell_user=None,
            )
            
            await self._send_vision_output(vision_output)
        else:
            # Navigation complete or failed
            nav_continuous.deactivate()
            self.state.primitive_in_execution = None
            
            # Request next image for normal processing
            await self.send_callback(
                MessageOut(type=MessageOutType.READY_FOR_IMAGE, payload={})
            )
            
            if msg:
                self.logger.info(f"Navigation complete: {msg}")
        
        return True

    def get_debug_state(self) -> dict:
        """
        Export the current state of the Brain for debugging purposes.
        Returns a dictionary with all relevant state information.
        """
        # Get history entries (limit to last 50 for performance)
        history_entries = []
        for entry in self.history.entries[-50:]:
            entry_data = {
                "timestamp": entry.timestamp.isoformat(),
                "type": entry.type.value,
                "description": (
                    entry.description[:500]
                    if len(entry.description) > 500
                    else entry.description
                ),
            }
            # Don't include full image data in debug output
            if entry.type.value in ["generic_image", "image_pre_action"]:
                entry_data["description"] = "[Image data omitted]"
            history_entries.append(entry_data)

        # Get primitive info
        primitive_in_execution = None
        if self.state.primitive_in_execution:
            primitive_in_execution = {
                "name": self.state.primitive_in_execution.name,
                "primitive_id": self.state.primitive_in_execution.primitive_id,
                "inputs": self.state.primitive_in_execution.inputs,
                "guidelines": self.state.primitive_in_execution.guidelines,
            }

        # Get registered primitives
        registered_primitives = []
        for p in self.state.primitives_list:
            registered_primitives.append(
                {
                    "name": p.name,
                    "guidelines": p.guidelines[:200] if p.guidelines else None,
                    "inputs": p.inputs,
                }
            )

        # Get local primitives
        local_primitives = [p.name for p in self.local_primitives_list]

        # Get primitive IDs map (recent ones)
        primitive_ids_map = {}
        for pid, pdef in list(self.state.primitive_ids_map.items())[-10:]:
            primitive_ids_map[pid] = {
                "name": pdef.name,
                "inputs": pdef.inputs,
            }

        # Get discrepancies
        discrepancies = []
        for d in self.history.discrepancies[-10:]:
            discrepancies.append(
                {
                    "timestamp": d["timestamp"].isoformat(),
                    "message": d["message"],
                }
            )

        return {
            "connection_id": self.connection_id,
            "running": self.running,
            "gemini_variant": self.state.gemini_variant,
            "directive": self.state.directive,
            "latest_user_message": self.state.latest_user_message,
            "primitive_in_execution": primitive_in_execution,
            "registered_primitives": registered_primitives,
            "local_primitives": local_primitives,
            "primitive_ids_map": primitive_ids_map,
            "message_queue_size": self.message_queue.qsize(),
            "history_entry_count": len(self.history.entries),
            "history_entries": history_entries,
            "discrepancies": discrepancies,
            "is_summarizing": self.history.is_summarizing,
            "enable_memory_commands": self.enable_memory_commands,
        }
