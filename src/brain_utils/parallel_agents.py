"""
Parallel agent processing for fast/slow agent pattern.
Shared between ChatHandler and ImageHandler.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Callable, Optional

from src.agents.fast_answer_agent import fast_answer, FastAnswerDecision
from src.baml_client.types import VisionAgentOutput


@dataclass
class ParallelAgentResult:
    """Result from parallel agent processing."""

    vision_output: Optional[VisionAgentOutput]
    fast_answered: bool
    fast_response: Optional[str] = None


async def run_agents_in_parallel(
    user_message: str,
    current_image: Optional[str],
    directive: Optional[str],
    current_primitive_name: Optional[str],
    history_summary: str,
    slow_agent_coro,
    send_chat_callback: Optional[Callable],
    logger,
) -> ParallelAgentResult:
    """
    Run fast and slow agents in parallel.

    Fast agent answers simple questions immediately.
    Slow agent (VLM) processes the full visual context.

    If fast agent can answer, sends response immediately and clears
    VLM's to_tell_user to avoid duplicate responses.
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
        )
    )

    slow_task = asyncio.create_task(slow_agent_coro)

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
                await send_chat_callback(fast_result.response)
                fast_answered = True
                fast_response = fast_result.response

        vision_output = await slow_task

        if fast_answered and vision_output:
            vision_output.to_tell_user = None

    except Exception as e:
        logger.error(f"Error in parallel agent processing: {e}")
        if not slow_task.done():
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

