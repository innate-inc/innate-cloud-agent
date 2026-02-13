"""
Primitive lifecycle handler for the Brain.
Handles primitive activation, completion, failure, interruption, and feedback.
"""

from typing import Callable, Dict, List, Optional

from src.agents.types import PrimitiveDefinition
from src.history.history import History, HistoryEntryType
from src.message_types import MessageOut
from src.primitives.types import Primitive


class PrimitiveHandler:
    """
    Handles skill lifecycle events:
    - Activation
    - Completion
    - Failure
    - Interruption
    - Feedback
    """

    def __init__(
        self,
        logger,
        history: History,
        primitives_list: List[PrimitiveDefinition],
        local_primitives_list: List[Primitive],
        primitive_ids_map: Dict[str, PrimitiveDefinition],
        send_callback: Callable,
        connection_id: str,
    ):
        self.logger = logger
        self.history = history
        self.primitives_list = primitives_list
        self.local_primitives_list = local_primitives_list
        self.primitive_ids_map = primitive_ids_map
        self.send_callback = send_callback
        self.connection_id = connection_id

    async def handle_primitive_completed(
        self,
        payload: dict,
        primitive_in_execution: Optional[PrimitiveDefinition],
    ) -> Optional[PrimitiveDefinition]:
        """Handle primitive completion. Returns None to clear primitive_in_execution."""
        primitive_id = payload["primitive_id"]
        primitive_name = payload["primitive_name"]

        if (
            primitive_in_execution
            and primitive_id == primitive_in_execution.primitive_id
        ):
            # Expected case: the completed primitive matches what we're tracking
            self.logger.info(
                f"Skill '{primitive_in_execution.name}' (ID: {primitive_id}) completed."
            )
            self.history.add(
                HistoryEntryType.PRIMITIVE_COMPLETED,
                description=f"Skill '{primitive_in_execution.name}' completed.",
            )
            return None

        # Stale completion - log warning but don't crash
        # This can happen due to race conditions when multiple primitives are sent quickly
        current_id = (
            primitive_in_execution.primitive_id if primitive_in_execution else None
        )
        current_name = primitive_in_execution.name if primitive_in_execution else None
        self.logger.warn(
            f"[Brain {self.connection_id}] Received completion for '{primitive_name}' "
            f"(ID: {primitive_id}) but current task is '{current_name}' "
            f"(ID: {current_id}). Ignoring stale completion."
        )
        return primitive_in_execution

    async def handle_primitive_failed(
        self,
        payload: dict,
        primitive_in_execution: Optional[PrimitiveDefinition],
    ) -> Optional[PrimitiveDefinition]:
        """Handle primitive failure. Returns None to clear primitive_in_execution."""
        primitive_id = payload["primitive_id"]
        primitive_name = payload["primitive_name"]

        if (
            primitive_in_execution
            and primitive_in_execution.primitive_id == primitive_id
        ):
            # Expected case: the failed primitive matches what we're tracking
            self.logger.info(f"Skill '{primitive_in_execution.name}' failed.")
            self.history.add(
                HistoryEntryType.PRIMITIVE_CANCELLED,
                description=f"Skill '{primitive_name}' failed.",
            )
            return None

        # Stale failure - log warning but don't crash
        # This can happen due to race conditions when multiple primitives are sent quickly
        current_id = (
            primitive_in_execution.primitive_id if primitive_in_execution else None
        )
        current_name = primitive_in_execution.name if primitive_in_execution else None
        self.logger.warn(
            f"[Brain {self.connection_id}] Received failure for '{primitive_name}' "
            f"(ID: {primitive_id}) but current task is '{current_name}' "
            f"(ID: {current_id}). Ignoring stale failure."
        )
        return primitive_in_execution

    async def handle_primitive_interrupted(
        self,
        payload: dict,
        primitive_in_execution: Optional[PrimitiveDefinition],
    ) -> Optional[PrimitiveDefinition]:
        """Handle primitive interruption. Returns None to clear primitive_in_execution."""
        primitive_id = payload["primitive_id"]
        primitive_name = payload["primitive_name"]

        if primitive_in_execution is None:
            self.logger.warn(
                f"Received interrupt for '{primitive_name}' (ID: {primitive_id}) "
                f"but no primitive is currently executing. Ignoring."
            )
            return None

        if primitive_in_execution.primitive_id != primitive_id:
            # Stale interrupt - the client is reporting an interrupt for an old primitive
            # while we've already moved on to tracking a new one. Ignore it.
            self.logger.warn(
                f"Interrupt for '{primitive_name}' (ID: {primitive_id}) doesn't match "
                f"current task '{primitive_in_execution.name}' (ID: {primitive_in_execution.primitive_id}). "
                f"Ignoring stale interrupt."
            )
            return primitive_in_execution

        self.logger.info(f"Skill '{primitive_in_execution.name}' interrupted.")
        self.history.add(
            HistoryEntryType.PRIMITIVE_INTERRUPTED,
            description=f"Skill '{primitive_name}' interrupted.",
        )
        return None

    def handle_primitive_feedback(
        self,
        payload: dict,
        primitive_in_execution: Optional[PrimitiveDefinition],
    ) -> None:
        """Handle primitive feedback. Records feedback and optional image in history."""
        feedback_text = payload.get("feedback")
        image_b64 = payload.get("image_b64")
        task_name = primitive_in_execution.name if primitive_in_execution else "unknown"

        if feedback_text:
            self.logger.info(f"Received primitive feedback: {feedback_text}")
            self.history.add(
                HistoryEntryType.PRIMITIVE_FEEDBACK,
                description=f"'{task_name}': {feedback_text}",
            )

        if image_b64:
            self.logger.info(f"Received feedback image from '{task_name}'")
            self.history.add(
                HistoryEntryType.PRIMITIVE_FEEDBACK_IMAGE,
                description=image_b64,
            )

        if not feedback_text and not image_b64:
            self.logger.warn(
                "Received primitive_feedback message with no feedback text or image."
            )

    async def handle_primitive_activated(
        self,
        payload: dict,
        primitive_in_execution: Optional[PrimitiveDefinition],
    ) -> Optional[PrimitiveDefinition]:
        """
        Handle primitive activation from client.

        Since we now set primitive_in_execution when sending (not when activated),
        this method just validates that the client activated what we expected
        and records it in history.

        Returns the current primitive_in_execution (unchanged).
        """
        primitive_id = payload["primitive_id"]
        primitive_activated = self.primitive_ids_map.get(primitive_id)

        if primitive_activated is None:
            self.logger.error(
                f"[Brain {self.connection_id}] Unknown primitive ID: {primitive_id}"
            )
            await self.send_callback(MessageOut(type="ready_for_image", payload={}))
            return primitive_in_execution

        # Check if this matches what we're currently tracking
        if (
            primitive_in_execution
            and primitive_in_execution.primitive_id == primitive_id
        ):
            # Expected case: client activated the primitive we sent
            self.logger.info(
                f"\033[92m[Brain {self.connection_id}] Task '{primitive_in_execution.name}' "
                f"(ID: {primitive_id}) activated by client.\033[0m"
            )
            self.history.add(
                HistoryEntryType.PRIMITIVE_ACTIVATED,
                description=f"Skill {primitive_in_execution.name} activated",
            )
        elif primitive_in_execution:
            # Client activated a different primitive than we're tracking
            # This can happen due to race conditions - log warning but don't crash
            self.logger.warn(
                f"[Brain {self.connection_id}] Client activated '{primitive_activated.name}' "
                f"(ID: {primitive_id}) but we're tracking '{primitive_in_execution.name}' "
                f"(ID: {primitive_in_execution.primitive_id}). Ignoring stale activation."
            )
        else:
            # No primitive in execution - unexpected activation
            self.logger.warn(
                f"[Brain {self.connection_id}] Client activated '{primitive_activated.name}' "
                f"(ID: {primitive_id}) but no primitive is in execution. Ignoring."
            )

        await self.send_callback(MessageOut(type="ready_for_image", payload={}))
        return primitive_in_execution

    def _find_matching_primitive(self, task_name: str):
        """Find a primitive by name in the registered lists."""
        return next(
            (
                prim
                for prim in self.primitives_list + self.local_primitives_list
                if prim.name == task_name
            ),
            None,
        )

    def handle_internal_feedback(
        self,
        primitive_name: str,
        feedback_message: str,
    ) -> None:
        """
        Handle feedback from a primitive called directly (not via message).
        """
        if feedback_message:
            self.logger.info(
                f"Received primitive feedback from '{primitive_name}': {feedback_message}"
            )
            entry_text = f"'{primitive_name}': {feedback_message}"
            self.history.add(
                HistoryEntryType.PRIMITIVE_FEEDBACK,
                description=entry_text,
            )
        else:
            self.logger.warn(
                f"Received empty feedback message from primitive '{primitive_name}'."
            )
