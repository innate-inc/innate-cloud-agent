"""
Chat message handler for the Brain.
Handles chat_in messages including special commands.

The chat handler runs only the fast agent for immediate responses.
If the fast agent defers, the message is stored and will be processed
by the image handler with the next fresh image.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

from src.agents.fast_answer_agent import fast_answer, FastAnswerDecision
from src.agents.types import PrimitiveDefinition
from src.brain_utils.constants import PrimitiveNames
from src.brain_utils.memory_state_manager import MemoryStateManager
from src.history.history import History, HistoryEntryType
from src.message_types import MessageOut
from src.primitives.types import Primitive


VALID_GEMINI_VARIANTS = ["gemini-flash", "gemini-flash-lite", "gemini-er"]


@dataclass
class ChatProcessingResult:
    """Result of processing a chat message."""

    new_gemini_variant: Optional[str] = None
    fast_response_sent: bool = False
    user_message_to_store: Optional[str] = None


class ChatHandler:
    """
    Handles chat_in messages and special commands.

    For regular user messages:
    - Runs only the fast agent for immediate responses
    - If fast agent can answer, sends response immediately
    - If fast agent defers, stores the message for the image handler
      to process with the next fresh image
    """

    def __init__(
        self,
        logger,
        history: History,
        local_primitives_list: List[Primitive],
        send_callback: Callable,
        memory_state_manager: Optional[MemoryStateManager] = None,
    ):
        self.logger = logger
        self.history = history
        self.local_primitives_list = local_primitives_list
        self.send_callback = send_callback
        self.memory_state_manager = memory_state_manager

    async def handle_chat_in(
        self,
        text: str,
        current_gemini_variant: str,
        enable_memory_commands: bool,
        directive: Optional[str] = None,
        primitive_in_execution: Optional[PrimitiveDefinition] = None,
        primitives_list: Optional[List[PrimitiveDefinition]] = None,
    ) -> ChatProcessingResult:
        """
        Handle a chat_in message.

        Runs the fast agent to check if we can answer immediately.
        If not, stores the message for the image handler to process
        with the next fresh image.
        """
        if text.startswith("!gemini"):
            new_variant = await self._handle_gemini_command(
                text, current_gemini_variant
            )
            return ChatProcessingResult(new_gemini_variant=new_variant)

        if enable_memory_commands and self.memory_state_manager is not None:
            handled = await self._handle_memory_commands(text)
            if handled:
                return ChatProcessingResult()

        if text.startswith("!"):
            handled = await self._handle_disabled_memory_command(text)
            if handled:
                return ChatProcessingResult()

        # Add user message to history
        self.history.add(HistoryEntryType.AUDIO_IN, description=text)

        # Get context for fast agent
        current_image = self.history.get_last_image()
        current_primitive_name = (
            primitive_in_execution.name if primitive_in_execution else None
        )
        history_summary = self.history.get_brief_summary()

        # Run fast agent only - it decides whether to answer now or defer
        self.logger.info(f"[FastAgent] Evaluating: '{text[:50]}...'")

        fast_result = await fast_answer(
            user_message=text,
            directive=directive,
            current_primitive=current_primitive_name,
            history_summary=history_summary,
            current_image=current_image,
        )

        self.logger.info(
            f"[FastAgent] Decision: {fast_result.decision.value}"
            + (f" - {fast_result.reasoning}" if fast_result.reasoning else "")
        )

        if (
            fast_result.decision == FastAnswerDecision.ANSWER_NOW
            and fast_result.response
        ):
            # Fast agent can answer - send response immediately
            await self._send_chat_response(fast_result.response)
            self.history.add(
                HistoryEntryType.SYSTEM_MESSAGE,
                description=f"Fast agent response: {fast_result.response}",
            )
            return ChatProcessingResult(fast_response_sent=True)
        else:
            # Fast agent defers - store message for image handler to process
            # with the next fresh image (both fast and slow agents will run there)
            self.logger.info(
                "[FastAgent] Deferring to vision agent - message stored for next image"
            )
            return ChatProcessingResult(user_message_to_store=text)

    async def _handle_gemini_command(
        self,
        text: str,
        current_variant: str,
    ) -> Optional[str]:
        """Handle !gemini command. Returns new variant if changed."""
        parts = text.split(maxsplit=1)

        if len(parts) > 1:
            requested_variant = parts[1].strip().lower()

            if requested_variant in VALID_GEMINI_VARIANTS:
                response_text = (
                    f"Gemini variant switched from '{current_variant}' "
                    f"to '{requested_variant}'"
                )
                self.logger.info(response_text)

                self.history.add(
                    HistoryEntryType.SYSTEM_MESSAGE,
                    description=response_text,
                )

                await self._send_chat_response(response_text)
                return requested_variant
            else:
                response_text = (
                    f"Invalid Gemini variant: '{requested_variant}'. "
                    f"Valid options are: {', '.join(VALID_GEMINI_VARIANTS)}"
                )
                await self._send_chat_response(response_text)
                return None
        else:
            response_text = (
                f"Current Gemini variant: '{current_variant}'\n"
                f"To change, use: !gemini VERSION\n"
                f"Valid versions: {', '.join(VALID_GEMINI_VARIANTS)}"
            )
            await self._send_chat_response(response_text)
            return None

    async def _handle_memory_commands(self, text: str) -> bool:
        """
        Handle memory-related commands.

        Returns:
            True if a memory command was handled, False otherwise
        """
        if text.startswith("!save_memory"):
            await self._handle_save_memory(text)
            return True

        if text.startswith("!load_memory"):
            await self._handle_load_memory(text)
            return True

        if text.startswith("!list_memory"):
            await self._handle_list_memory()
            return True

        return False

    async def _handle_save_memory(self, text: str) -> None:
        """Handle !save_memory command."""
        parts = text.split(maxsplit=1)
        state_name = parts[1] if len(parts) > 1 else ""

        navigate_through_memory = self._get_navigate_through_memory_primitive()

        success = await self.memory_state_manager.save_memory_state(
            state_name, self.history, navigate_through_memory
        )

        if success:
            response_text = f"Memory state '{state_name}' saved successfully"
        else:
            response_text = f"Failed to save memory state '{state_name}'"

        await self._send_chat_response(response_text)

    async def _handle_load_memory(self, text: str) -> tuple[bool, Optional[dict]]:
        """
        Handle !load_memory command.

        Returns:
            Tuple of (success, state_reset_info) where state_reset_info contains
            fields that need to be reset in the Brain.
        """
        parts = text.split(maxsplit=1)

        if len(parts) <= 1:
            await self._send_chat_response("Please specify a memory state name to load")
            return

        state_name = parts[1]
        navigate_through_memory = self._get_navigate_through_memory_primitive()

        success = await self.memory_state_manager.load_memory_state(
            state_name, self.history, navigate_through_memory
        )

        if success:
            response_text = f"Memory state '{state_name}' loaded successfully"
        else:
            response_text = f"Failed to load memory state '{state_name}'"

        await self._send_chat_response(response_text)

    async def _handle_list_memory(self) -> None:
        """Handle !list_memory command."""
        states = self.memory_state_manager.get_available_states()

        if states:
            states_list = "\n- " + "\n- ".join(states)
            response_text = f"Available memory states:{states_list}"
        else:
            response_text = "No memory states available"

        await self._send_chat_response(response_text)

    async def _handle_disabled_memory_command(self, text: str) -> bool:
        """
        Handle memory commands when they are disabled.

        Returns:
            True if a disabled memory command was detected and handled
        """
        memory_commands = ["!save_memory", "!load_memory", "!list_memory"]
        if any(text.startswith(cmd) for cmd in memory_commands):
            response_text = (
                "Memory management commands are disabled. "
                "They can be enabled when starting the brain."
            )
            await self._send_chat_response(response_text)
            return True
        return False

    def _get_navigate_through_memory_primitive(self) -> Optional[Primitive]:
        """Get the NavigateThroughMemory primitive from local primitives."""
        return next(
            (
                p
                for p in self.local_primitives_list
                if p.name == PrimitiveNames.NAVIGATE_THROUGH_MEMORY
            ),
            None,
        )

    async def _send_chat_response(self, text: str) -> None:
        """Send a chat response to the client."""
        await self.send_callback(
            MessageOut(type="brain/chat_out", payload={"text": text})
        )
