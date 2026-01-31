"""
Parallel agent processing for fast/slow agent pattern.
Shared between ChatHandler and ImageHandler.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Callable, List, Optional

from src.agents.fast_answer_agent import fast_answer, FastAnswerDecision
from src.agents.types import PrimitiveDefinition, VisionAgentOutput
from src.history.history import History


@dataclass
class ParallelAgentResult:
    """Result from parallel agent processing."""

    vision_output: Optional[VisionAgentOutput]
    fast_answered: bool
    fast_response: Optional[str] = None


def validate_vision_output(
    vision_output: VisionAgentOutput,
    primitive_in_execution: Optional[PrimitiveDefinition],
    history: History,
) -> VisionAgentOutput:
    """
    Validate and clean up the vision output.
    Ensures next_task is a PrimitiveDefinition object with a primitive_id.
    Handles discrepancies between VLM output and current state.

    Args:
        vision_output: The raw vision output from the VLM
        primitive_in_execution: Currently executing primitive, if any
        history: History instance for recording discrepancies

    Returns:
        Validated VisionAgentOutput with proper next_task type
    """
    # Validate the next task - convert from dict to PrimitiveDefinition if needed
    vision_output.next_task = (
        PrimitiveDefinition.model_validate(vision_output.next_task)
        if vision_output.next_task
        else None
    )

    # Check for discrepancy: next_task provided without stop_current_task
    # when a primitive is already running
    if (
        not vision_output.stop_current_task
        and vision_output.next_task is not None
        and primitive_in_execution is not None
    ):
        history.record_discrepancy(
            message=(
                f"The VLM returned a next_task ({vision_output.next_task.name}) "
                f"even though there is a task running "
                f"({primitive_in_execution.name}) and it did not say to "
                f"stop the current task."
            )
        )
        # Force next_task to None if stop wasn't explicitly requested
        vision_output.next_task = None

    # Ensure next_task has a primitive_id
    if vision_output.next_task and not vision_output.next_task.primitive_id:
        vision_output.next_task.primitive_id = str(uuid.uuid4())

    return vision_output


async def run_agents_in_parallel(
    user_message: str,
    current_image: Optional[str],
    directive: Optional[str],
    current_primitive_name: Optional[str],
    history_summary: str,
    slow_agent_coro,
    send_chat_callback: Optional[Callable],
    logger,
    primitives_list: Optional[List[PrimitiveDefinition]] = None,
) -> ParallelAgentResult:
    """
    Run fast and slow agents in parallel.

    Fast agent answers simple questions immediately.
    Slow agent (VLM) processes the full visual context.

    If fast agent can answer, sends response immediately and clears
    VLM's to_tell_user to avoid duplicate responses.

    If slow_agent_coro is None (e.g., no image available), only the fast
    agent will be run.
    """
    parallel_start = time.time()
    msg_preview = user_message[:50] if len(user_message) > 50 else user_message
    logger.info(f"[Parallel] Starting for: '{msg_preview}...'")

    fast_task = asyncio.create_task(
        fast_answer(
            user_message=user_message,
            directive=directive,
            current_primitive=current_primitive_name,
            history_summary=history_summary,
            current_image=current_image,
            primitives_list=primitives_list,
        )
    )

    # Only create slow task if we have a coroutine (requires an image)
    slow_task = asyncio.create_task(slow_agent_coro) if slow_agent_coro else None

    fast_answered = False
    fast_response = None
    vision_output = None

    try:
        fast_result = await fast_task
        logger.info(
            f"[Parallel] Fast agent: {fast_result.decision.value if fast_result else 'None'}"
        )

        if fast_result and fast_result.decision == FastAnswerDecision.ANSWER_NOW:
            if fast_result.response and send_chat_callback:
                logger.info(f"[Parallel] Fast agent answering: '{fast_result.response[:50]}...'")
                await send_chat_callback(fast_result.response)
                fast_answered = True
                fast_response = fast_result.response
                logger.info("[Parallel] Fast agent response sent via callback")

        # Only await slow task if it exists
        if slow_task:
            vision_output = await slow_task

            if fast_answered and vision_output:
                vision_output.to_tell_user = None

    except Exception as e:
        logger.error(f"Error in parallel agent processing: {e}")
        if slow_task and not slow_task.done():
            vision_output = await slow_task

    logger.info(
        f"[Parallel] Done in {time.time() - parallel_start:.2f}s, "
        f"fast_answered: {fast_answered}"
    )

    return ParallelAgentResult(
        vision_output=vision_output,
        fast_answered=fast_answered,
        fast_response=fast_response,
    )
