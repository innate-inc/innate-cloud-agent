"""
Chat message handler for the Brain.
Handles chat_in messages including special commands.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

from src.brain_utils.constants import PrimitiveNames
from src.brain_utils.memory_state_manager import MemoryStateManager
from src.history.history import History, HistoryEntryType
from src.message_types import MessageOut
from src.primitives.types import Primitive
from src.agents.fast_answer_agent import fast_answer, FastAnswerDecision


# Valid Gemini variants
VALID_GEMINI_VARIANTS = ["gemini-flash", "gemini-flash-lite", "gemini-er"]


@dataclass
class ChatProcessingResult:
    """Result of processing a chat message."""

    new_gemini_variant: Optional[str] = None
    user_message_to_store: Optional[str] = None
    fast_response_sent: bool = False
    deferred_to_vision: bool = False


class ChatHandler:
    """
    Handles chat_in messages and special commands.

    Special commands (if memory enabled):
    - !save_memory NAME: Saves the current memory state
    - !load_memory NAME: Loads a saved memory state
    - !list_memory: Lists available memory states

    Other commands (always enabled):
    - !gemini VERSION: Switches the Gemini version
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
        current_primitive_name: Optional[str] = None,
    ) -> ChatProcessingResult:
        """
        Handle a chat_in message with parallel fast/slow agent processing.

        Args:
            text: The chat message text
            current_gemini_variant: Current Gemini variant setting
            enable_memory_commands: Whether memory commands are enabled
            directive: Current directive for context
            current_primitive_name: Name of currently executing primitive

        Returns:
            ChatProcessingResult with processing outcome
        """
        # Handle Gemini version switch command (always enabled)
        if text.startswith("!gemini"):
            new_variant = await self._handle_gemini_command(
                text, current_gemini_variant
            )
            return ChatProcessingResult(new_gemini_variant=new_variant)

        # Check for special memory commands if enabled
        if enable_memory_commands and self.memory_state_manager is not None:
            handled = await self._handle_memory_commands(text)
            if handled:
                return ChatProcessingResult()

        # Handle memory command attempt when disabled
        if text.startswith("!"):
            handled = await self._handle_disabled_memory_command(text)
            if handled:
                return ChatProcessingResult()

        # Regular user message - add to history
        self.history.add(HistoryEntryType.AUDIO_IN, description=text)

        # Try fast answer agent for quick responses
        fast_result = await fast_answer(
            user_message=text,
            directive=directive,
            current_primitive=current_primitive_name,
            history_summary=self.history.get_brief_summary(),
        )

        if (
            fast_result.decision == FastAnswerDecision.ANSWER_NOW
            and fast_result.response
        ):
            await self._send_chat_response(fast_result.response)
            self.history.add(
                HistoryEntryType.SYSTEM_MESSAGE,
                description=f"Fast agent response: {fast_result.response}",
            )
            return ChatProcessingResult(
                user_message_to_store=text, fast_response_sent=True
            )

        # Defer to vision agent
        return ChatProcessingResult(user_message_to_store=text, deferred_to_vision=True)

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
