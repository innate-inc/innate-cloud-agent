"""
Image message handler for the Brain.
Breaks down the complex handle_image logic into smaller, focused methods.
"""

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from src.agents.types import PrimitiveDefinition, VisionAgentOutput
from src.brain_utils.constants import PrimitiveNames
from src.brain_utils.image_processor import ImageProcessor
from src.brain_utils.navigation_handler import NavigationHandler
from src.brain_utils.parallel_agents import (
    run_agents_in_parallel,
    validate_vision_output,
)
from src.brain_utils.vision_service import VisionAgentType, VisionService
from src.history.history import History, HistoryEntryType
from src.primitives.transforms import primitive_to_object
from src.primitives.types import Primitive


@dataclass
class ImageProcessingResult:
    """Result of processing an image message."""

    vision_output: VisionAgentOutput
    vlm_processing_time: float


@dataclass
class TokenMetrics:
    """Token usage metrics from VLM processing."""

    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]
    tokens_per_second: Optional[float]
    total_processing_seconds: float


class ImageHandler:
    """
    Handles image message processing for the Brain.
    Uses parallel fast/slow agent processing when a user message is present.
    """

    def __init__(
        self,
        logger,
        image_processor: ImageProcessor,
        vision_service: VisionService,
        navigation_handler: NavigationHandler,
        history: History,
        local_primitives_list: List[Primitive],
        send_chat_callback: Optional[Callable] = None,
    ):
        self.logger = logger
        self.image_processor = image_processor
        self.vision_service = vision_service
        self.navigation_handler = navigation_handler
        self.history = history
        self.local_primitives_list = local_primitives_list
        self.send_chat_callback = send_chat_callback

    async def process_image_message(
        self,
        payload: dict,
        latest_user_message: Optional[str],
        primitive_in_execution: Optional[PrimitiveDefinition],
        primitives_list: List[PrimitiveDefinition],
        directive: Optional[str],
        gemini_variant: str,
        connection_id: str,
        send_error_callback: Callable,
        send_ready_callback: Callable,
    ) -> Optional[ImageProcessingResult]:
        """
        Process an image message and return the vision output.

        Args:
            payload: The message payload containing image data
            latest_user_message: Optional user message to include
            primitive_in_execution: Currently executing primitive
            primitives_list: List of registered primitives
            directive: Directive to steer the VLM
            gemini_variant: Which Gemini variant to use
            connection_id: Connection identifier for logging
            send_error_callback: Callback to send error messages
            send_ready_callback: Callback to request next image

        Returns:
            ImageProcessingResult or None if processing failed
        """
        # Extract and validate image data
        extracted_data = self._extract_and_validate_image_data(payload)
        if extracted_data is None:
            await send_error_callback("Failed to extract image data from payload")
            await send_ready_callback()
            return None

        (
            base64_img_extracted,
            depth_payload,
            robot_coords,
            map_payload,
            additional_image_data,
            camera_info,
        ) = extracted_data

        # Ensure images are in JPEG format
        processed_images = await self._process_image_formats(
            base64_img_extracted,
            additional_image_data,
            send_error_callback,
            send_ready_callback,
        )
        if processed_images is None:
            return None

        current_image_for_vlm, additional_image_data = processed_images

        # Process depth and map data
        self._process_supplementary_data(depth_payload, map_payload, robot_coords)

        # Call the VLM (with parallel fast agent if user message present)
        local_primitives_list = [
            primitive_to_object(prim) for prim in self.local_primitives_list
        ]

        vlm_start_time = time.time()

        # Parallel processing: if there's a user message, start fast agent in parallel
        # with the vision agent. Fast agent may answer immediately, avoiding VLM latency.
        if latest_user_message:
            self.logger.info(
                f"Processing image with pending user message: '{latest_user_message[:50]}...'"
            )
            vision_output, fast_answered = await self._call_agents_in_parallel(
                current_image_for_vlm=current_image_for_vlm,
                latest_user_message=latest_user_message,
                primitive_in_execution=primitive_in_execution,
                primitives_list=local_primitives_list + primitives_list,
                robot_coords=robot_coords,
                directive=directive,
                gemini_variant=gemini_variant,
                additional_image_data=additional_image_data,
            )
            if fast_answered:
                # Fast agent handled the response, VLM's to_tell_user already cleared
                vlm_processing_time = time.time() - vlm_start_time
                self.logger.info(f"Fast agent answered in {vlm_processing_time:.2f}s")
            elif (
                vision_output and vision_output.to_tell_user and self.send_chat_callback
            ):
                # Fast agent deferred, send slow agent's response as chat
                await self.send_chat_callback(vision_output.to_tell_user)
                self.logger.info(
                    f"Slow agent response sent: {vision_output.to_tell_user[:50]}..."
                )
                # Record in history
                self.history.add(
                    HistoryEntryType.SYSTEM_MESSAGE,
                    description=f"Vision agent response: {vision_output.to_tell_user}",
                )
        else:
            # No user message, just call the vision agent
            vision_output = await self.vision_service.call_visual_language_model(
                base64_img=current_image_for_vlm,
                user_prompt_text=latest_user_message,
                primitive_in_execution=primitive_in_execution,
                primitives_list=local_primitives_list + primitives_list,
                history=self.history.get_as_multimodal_list(),
                robot_coords=robot_coords,
                directive=directive,
                agent_type=VisionAgentType.NATIVE_GEMINI_MULTI,
                gemini_variant=gemini_variant,
                additional_image_data=additional_image_data,
            )

        vlm_processing_time = time.time() - vlm_start_time

        # Add image to history with appropriate type
        self._add_image_to_history(current_image_for_vlm, vision_output)

        # Handle VLM failure
        if not vision_output:
            self.logger.error(
                f"No vision output received for connection {connection_id}"
            )
            vision_output = self._create_fallback_vision_output()

        # Validate and clean up vision output
        vision_output = validate_vision_output(
            vision_output, primitive_in_execution, self.history
        )

        # Handle navigation primitives
        (
            vision_output,
            vision_output_for_history,
        ) = await self._handle_navigation_primitives(
            vision_output,
            robot_coords,
            base64_img_extracted,
            depth_payload,
            map_payload,
            camera_info,
            connection_id,
        )

        return (
            ImageProcessingResult(
                vision_output=vision_output,
                vlm_processing_time=vlm_processing_time,
            ),
            vision_output_for_history,
        )

    def _extract_and_validate_image_data(self, payload: dict) -> Optional[Tuple]:
        """Extract and validate image data from payload."""
        try:
            return self.image_processor.extract_image_data(payload)
        except ValueError as e:
            self.logger.error(f"Image data extraction error: {e}")
            return None

    async def _process_image_formats(
        self,
        base64_img: str,
        additional_image_data: dict,
        send_error_callback: Callable,
        send_ready_callback: Callable,
    ) -> Optional[Tuple[str, dict]]:
        """
        Ensure images are in JPEG format.

        Returns:
            Tuple of (processed_image, additional_image_data) or None if failed
        """
        try:
            current_image_for_vlm = self.image_processor.ensure_jpeg_format(base64_img)
            if additional_image_data:
                additional_image_for_vlm = self.image_processor.ensure_jpeg_format(
                    additional_image_data["image_b64"]
                )
                additional_image_data["image_b64"] = additional_image_for_vlm
            return current_image_for_vlm, additional_image_data
        except ValueError as e:
            self.logger.error(f"Image format error: {e}")
            await send_error_callback(f"Image format error: {e}")
            await send_ready_callback()
            return None

    def _process_supplementary_data(
        self,
        depth_payload: Optional[dict],
        map_payload: Optional[dict],
        robot_coords: dict,
    ) -> None:
        """Process depth map and map data if available."""
        if depth_payload:
            self.image_processor.process_depth(depth_payload)
        if map_payload:
            self.image_processor.process_map_with_robot(map_payload, robot_coords)

    def _add_image_to_history(
        self,
        image: str,
        vision_output: Optional[VisionAgentOutput],
    ) -> None:
        """Add the current image to history with the appropriate type."""
        if not image:
            return

        if vision_output and vision_output.next_task:
            self.history.add(
                HistoryEntryType.IMAGE_PRE_ACTION,
                description=image,
            )
        else:
            self.history.add(
                HistoryEntryType.GENERIC_IMAGE,
                description=image,
            )

    def _create_fallback_vision_output(self) -> VisionAgentOutput:
        """Create a fallback vision output when VLM fails."""
        return VisionAgentOutput(
            stop_current_task=True,
            observation="The brain failed, so it stopped the current task.",
            thoughts="The brain failed, so it stopped the current task.",
            new_goal=None,
            next_task=None,
            anticipation=None,
            to_tell_user=(
                "BEEP BOOP BEEP BOOP, the brain failed. Stopping the current task."
            ),
        )

    async def _handle_navigation_primitives(
        self,
        vision_output: VisionAgentOutput,
        robot_coords: dict,
        base64_img: str,
        depth_payload: Optional[dict],
        map_payload: Optional[dict],
        camera_info: dict,
        connection_id: str,
    ) -> Tuple[VisionAgentOutput, Optional[VisionAgentOutput]]:
        """
        Handle special navigation primitives that need conversion.

        Returns:
            Tuple of (updated_vision_output, vision_output_for_history)
        """
        if not vision_output.next_task:
            return vision_output, None

        primitive_name = vision_output.next_task.name
        vision_output_for_history = None
        has_canceled = False

        if primitive_name == PrimitiveNames.NAVIGATE_IN_SIGHT:
            vision_output_for_history = vision_output.model_copy()
            (
                vision_output,
                has_canceled,
            ) = await self.navigation_handler.handle_navigate_in_sight(
                vision_output,
                robot_coords,
                base64_img,
                depth_payload,
                map_payload,
                camera_info,
            )

        elif primitive_name == PrimitiveNames.NAVIGATE_THROUGH_MEMORY:
            vision_output_for_history = vision_output.model_copy()
            (
                vision_output,
                has_canceled,
            ) = await self.navigation_handler.handle_navigate_through_memory(
                vision_output, connection_id, map_payload
            )

        elif primitive_name == PrimitiveNames.TURN_AND_MOVE:
            vision_output_for_history = vision_output.model_copy()
            (
                vision_output,
                has_canceled,
            ) = await self.navigation_handler.handle_turn_and_move(
                vision_output, robot_coords, map_payload
            )

        elif primitive_name == PrimitiveNames.NAV_INSIGHT_CONTINUOUS:
            vision_output_for_history = vision_output.model_copy()
            (
                vision_output,
                has_canceled,
            ) = await self.navigation_handler.handle_nav_insight_continuous(
                vision_output,
                robot_coords,
                base64_img,
                depth_payload,
                map_payload,
                camera_info,
            )

        elif primitive_name == PrimitiveNames.CHECK_DISTANCE_AND_ORIENTATION:
            vision_output_for_history = vision_output.model_copy()
            (
                vision_output,
                has_canceled,
            ) = await self.navigation_handler.handle_check_distance_and_orientation(
                vision_output,
                robot_coords,
                base64_img,
                depth_payload,
                map_payload,
                camera_info,
            )
            # Add history entries for this self-completing primitive
            self.history.add(
                HistoryEntryType.PRIMITIVE_ACTIVATED,
                description=f"Primitive {PrimitiveNames.CHECK_DISTANCE_AND_ORIENTATION} activated",
            )
            self.history.add(
                HistoryEntryType.PRIMITIVE_COMPLETED,
                description=f"Primitive {PrimitiveNames.CHECK_DISTANCE_AND_ORIENTATION} completed",
            )
            vision_output.next_task = None

        if has_canceled and vision_output_for_history:
            vision_output_for_history = vision_output.model_copy()

        return vision_output, vision_output_for_history

    def calculate_token_metrics(
        self,
        vision_output: VisionAgentOutput,
        time_elapsed: float,
    ) -> TokenMetrics:
        """Calculate token usage metrics from VLM processing."""
        input_tokens = vision_output.input_tokens
        output_tokens = vision_output.output_tokens

        total_tokens = None
        tokens_per_second = None

        if input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
            if time_elapsed > 0:
                tokens_per_second = total_tokens / time_elapsed

        return TokenMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            tokens_per_second=tokens_per_second,
            total_processing_seconds=time_elapsed,
        )

    def format_token_info(self, metrics: TokenMetrics) -> str:
        """Format token metrics for logging."""
        if metrics.total_tokens is None:
            return ""

        token_info = f", tokens: {metrics.total_tokens}"
        if metrics.tokens_per_second is not None:
            token_info += f" ({metrics.tokens_per_second:.1f} tokens/sec)"
        return token_info

    async def _call_agents_in_parallel(
        self,
        current_image_for_vlm: str,
        latest_user_message: str,
        primitive_in_execution: Optional[PrimitiveDefinition],
        primitives_list: List[PrimitiveDefinition],
        robot_coords: dict,
        directive: Optional[str],
        gemini_variant: str,
        additional_image_data: dict,
    ) -> Tuple[Optional[VisionAgentOutput], bool]:
        """Call fast and slow agents in parallel, returns (vision_output, fast_answered)."""
        current_primitive_name = (
            primitive_in_execution.name if primitive_in_execution else None
        )

        slow_coro = self.vision_service.call_visual_language_model(
            base64_img=current_image_for_vlm,
            user_prompt_text=latest_user_message,
            primitive_in_execution=primitive_in_execution,
            primitives_list=primitives_list,
            history=self.history.get_as_multimodal_list(),
            robot_coords=robot_coords,
            directive=directive,
            agent_type=VisionAgentType.NATIVE_GEMINI_MULTI,
            gemini_variant=gemini_variant,
            additional_image_data=additional_image_data,
        )

        result = await run_agents_in_parallel(
            user_message=latest_user_message,
            current_image=current_image_for_vlm,
            directive=directive,
            current_primitive_name=current_primitive_name,
            history_summary=self.history.get_brief_summary(),
            slow_agent_coro=slow_coro,
            send_chat_callback=self.send_chat_callback,
            logger=self.logger,
            primitives_list=primitives_list,
        )

        return result.vision_output, result.fast_answered
