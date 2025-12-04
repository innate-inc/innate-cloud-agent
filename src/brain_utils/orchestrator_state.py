"""
Orchestrator state handler for the Brain.
Handles state persistence and restoration for stateless server operation.
"""

from typing import Callable, Dict, Optional

from src.agents.types import PrimitiveDefinition
from src.history.history import History, HistoryEntryType
from src.message_types import MessageOut, MessageOutType


class OrchestratorStateHandler:
    """
    Manages orchestrator state persistence for stateless server operation.

    The robot sends its state with every message, and the server sends
    updated state after every vision output. This allows the server to
    be stateless while maintaining execution context.
    """

    def __init__(
        self,
        logger,
        history: History,
        primitive_ids_map: Dict[str, PrimitiveDefinition],
        send_callback: Callable,
        connection_id: str,
    ):
        self.logger = logger
        self.history = history
        self.primitive_ids_map = primitive_ids_map
        self.send_callback = send_callback
        self.connection_id = connection_id

    def restore_from_payload(
        self,
        payload: dict,
    ) -> Optional[PrimitiveDefinition]:
        """
        Restore orchestrator state from a message payload.
        Called on every IMAGE message to support stateless server operation.

        Args:
            payload: Message payload containing primitive_in_execution data

        Returns:
            Restored PrimitiveDefinition or None
        """
        primitive_in_execution_data = payload.get("primitive_in_execution")

        if primitive_in_execution_data:
            try:
                primitive_in_execution = PrimitiveDefinition.model_validate(
                    primitive_in_execution_data
                )
                self.logger.debug(
                    f"Restored orchestrator state from payload: "
                    f"'{primitive_in_execution.name}' "
                    f"(id: {primitive_in_execution.primitive_id})"
                )

                # Also add to primitive_ids_map for tracking
                if primitive_in_execution.primitive_id:
                    self.primitive_ids_map[primitive_in_execution.primitive_id] = (
                        primitive_in_execution
                    )

                return primitive_in_execution
            except Exception as e:
                self.logger.error(
                    f"Error restoring orchestrator state from payload: {e}"
                )
                return None
        else:
            return None

    async def handle_orchestrator_state_message(
        self,
        payload: dict,
    ) -> Optional[PrimitiveDefinition]:
        """
        Handle explicit orchestrator_state message.
        Used for state restoration on reconnection.

        Args:
            payload: Message payload containing state data

        Returns:
            Restored PrimitiveDefinition or None
        """
        primitive_in_execution = self.restore_from_payload(payload)

        if primitive_in_execution:
            self.logger.info(
                f"Restored orchestrator state via message: "
                f"'{primitive_in_execution.name}' "
                f"(id: {primitive_in_execution.primitive_id})"
            )
            self.history.add(
                HistoryEntryType.SYSTEM_MESSAGE,
                description=(
                    f"Restored orchestrator state: task "
                    f"'{primitive_in_execution.name}' in execution"
                ),
            )
        else:
            self.logger.info(
                "Restored orchestrator state via message: no primitive in execution"
            )

        return primitive_in_execution

    async def send_state(
        self,
        primitive_in_execution: Optional[PrimitiveDefinition],
    ) -> None:
        """
        Send the current orchestrator state to the robot for persistence.
        Called after every vision output to support stateless server operation.

        Args:
            primitive_in_execution: Current primitive being executed, or None
        """
        state_payload = {
            "primitive_in_execution": (
                primitive_in_execution.model_dump() if primitive_in_execution else None
            )
        }

        response = MessageOut(
            type=MessageOutType.SAVE_ORCHESTRATOR_STATE,
            payload=state_payload,
        )
        await self.send_callback(response)
        self.logger.debug(
            f"Sent orchestrator state to robot: "
            f"primitive_in_execution="
            f"{primitive_in_execution.name if primitive_in_execution else None}"
        )
