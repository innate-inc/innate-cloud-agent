"""
Primitive lifecycle handler for the Brain.
Handles primitive activation, completion, failure, interruption, and feedback.
"""

from typing import Callable, Dict, List, Optional

from src.agents.types import PrimitiveDefinition
from src.history.history import History, HistoryEntryType
from src.message_types import MessageOut
from src.primitives.transforms import primitive_to_object
from src.primitives.types import Primitive


class PrimitiveHandler:
    """
    Handles primitive lifecycle events:
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
            self.logger.info(
                f"Task '{primitive_in_execution.name}' (ID: {primitive_id}) completed."
            )
            self.history.add(
                HistoryEntryType.PRIMITIVE_COMPLETED,
                description=f"Task '{primitive_in_execution.name}' completed.",
            )
            return None

        raise ValueError(
            f"[Brain {self.connection_id}] Task '{primitive_name}' (ID: {primitive_id}) "
            f"is not the current task in execution."
        )

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
            self.logger.info(f"Task '{primitive_in_execution.name}' failed.")
            self.history.add(
                HistoryEntryType.PRIMITIVE_CANCELLED,
                description=f"Primitive '{primitive_name}' failed.",
            )
            return None

        raise ValueError(
            f"[Brain {self.connection_id}] Task '{primitive_name}' (ID: {primitive_id}) "
            f"is not the current task in execution."
        )

    async def handle_primitive_interrupted(
        self,
        payload: dict,
        primitive_in_execution: Optional[PrimitiveDefinition],
    ) -> Optional[PrimitiveDefinition]:
        """Handle primitive interruption. Returns None to clear primitive_in_execution."""
        primitive_id = payload["primitive_id"]
        primitive_name = payload["primitive_name"]

        if (
            primitive_in_execution
            and primitive_in_execution.primitive_id == primitive_id
        ):
            self.logger.info(f"Task '{primitive_in_execution.name}' interrupted.")
            self.history.add(
                HistoryEntryType.PRIMITIVE_INTERRUPTED,
                description=f"Primitive '{primitive_name}' interrupted.",
            )
            return None

        raise ValueError(
            f"[Brain {self.connection_id}] Task '{primitive_name}' (ID: {primitive_id}) "
            f"is not the current task in execution."
        )

    def handle_primitive_feedback(
        self,
        payload: dict,
        primitive_in_execution: Optional[PrimitiveDefinition],
    ) -> None:
        """Handle primitive feedback. Records feedback in history."""
        feedback_text = payload.get("feedback")
        if feedback_text:
            self.logger.info(f"Received primitive feedback: {feedback_text}")
            task_name = (
                primitive_in_execution.name if primitive_in_execution else "unknown"
            )
            self.history.add(
                HistoryEntryType.PRIMITIVE_FEEDBACK,
                description=f"'{task_name}': {feedback_text}",
            )
        else:
            self.logger.warning(
                "Received primitive_feedback message with no feedback text."
            )

    async def handle_primitive_activated(
        self,
        payload: dict,
        primitive_in_execution: Optional[PrimitiveDefinition],
    ) -> Optional[PrimitiveDefinition]:
        """Handle primitive activation. Returns the new primitive_in_execution."""
        primitive_id = payload["primitive_id"]
        primitive_activated = self.primitive_ids_map.get(primitive_id)

        if primitive_activated is None:
            self.logger.error(
                f"[Brain {self.connection_id}] Unknown primitive ID: {primitive_id}"
            )
            await self.send_callback(MessageOut(type="ready_for_image", payload={}))
            return primitive_in_execution

        if primitive_in_execution:
            self.logger.warn(
                f"[Brain {self.connection_id}] Task '{primitive_in_execution.name}' "
                f"(ID: {primitive_id}) was activated by the client, but we didn't activate it."
            )
            new_primitive_in_execution = primitive_in_execution
        else:
            task_name = primitive_activated.name
            self.logger.info(
                f"\033[92m[Brain {self.connection_id}] Task '{task_name}' "
                f"(ID: {primitive_id}) activated.\033[0m"
            )

            matched_prim = self._find_matching_primitive(task_name)
            if matched_prim is not None:
                prim_obj = (
                    primitive_to_object(matched_prim)
                    if isinstance(matched_prim, Primitive)
                    else matched_prim
                )
                if primitive_id:
                    prim_obj.primitive_id = primitive_id
                new_primitive_in_execution = prim_obj
                self.history.add(
                    HistoryEntryType.PRIMITIVE_ACTIVATED,
                    description=f"Primitive {task_name} activated",
                )
            else:
                new_primitive_in_execution = None

        await self.send_callback(MessageOut(type="ready_for_image", payload={}))
        return new_primitive_in_execution

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
            self.logger.warning(
                f"Received empty feedback message from primitive '{primitive_name}'."
            )
