"""
Fast Answer Agent - A lightweight agent for quick responses.

This agent answers simple questions quickly using Gemini 2.0 Flash with the
current camera image. It runs in parallel with the main vision agent.
"""

import os
import json
import asyncio
import base64
from typing import Optional
from dataclasses import dataclass
from enum import Enum

from google import genai
from google.genai import types


FAST_MODEL_NAME = "gemini-2.0-flash"
FAST_AGENT_TEMPERATURE = 0.3
FAST_AGENT_MAX_OUTPUT_TOKENS = 512


class FastAnswerDecision(str, Enum):
    """Decision made by the fast answer agent."""

    ANSWER_NOW = "answer_now"
    DEFER_TO_VISION = "defer_to_vision"


@dataclass
class FastAnswerResult:
    """Result from the fast answer agent."""

    decision: FastAnswerDecision
    response: Optional[str] = None
    reasoning: Optional[str] = None


FAST_ANSWER_PROMPT = """You are a fast-response assistant for a robot. Your job is to quickly determine if a user's question can be answered immediately, or if it requires the robot to move or perform physical actions.

You have access to the robot's current camera view, so you CAN answer questions about what the robot sees.

<context>
Current directive: {directive}
Primitive currently running: {current_primitive}
Recent history summary: {history_summary}
</context>

<user_message>
{user_message}
</user_message>

<decision_rules>
Answer "ANSWER_NOW" if the question is:
- A general knowledge question (e.g., "What is the capital of France?")
- A conversational response (e.g., "Hello", "How are you?", "Thanks")
- A question about time, date, or general facts
- A clarification question that doesn't need movement
- A question you can answer based on the provided context/history
- A question about what the robot sees (e.g., "What do you see?", "Is there a chair?", "What's in front of you?") - USE THE PROVIDED IMAGE TO ANSWER

Answer "DEFER_TO_VISION" if the question:
- Requires physical movement (e.g., "Go to the kitchen", "Turn left")
- Requires executing a primitive/action
- Relates to navigation or spatial reasoning that requires movement
- Is a complex multi-step task
</decision_rules>

<response_format>
Respond with a JSON object:
{{
  "decision": "ANSWER_NOW" or "DEFER_TO_VISION",
  "response": "Your response to the user (only if decision is ANSWER_NOW)",
  "reasoning": "Brief explanation of your decision"
}}
</response_format>
"""


# Singleton client
_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        _client = genai.Client(api_key=api_key)
    return _client


def _prepare_image_part(base64_img: str) -> types.Part:
    """Convert a base64 image to a Gemini Part object."""
    img_data = base64_img
    if img_data.startswith("data:image"):
        img_data = img_data.split(",")[1]
    return types.Part.from_bytes(
        data=base64.b64decode(img_data), mime_type="image/jpeg"
    )


async def fast_answer(
    user_message: str,
    directive: Optional[str] = None,
    current_primitive: Optional[str] = None,
    history_summary: Optional[str] = None,
    current_image: Optional[str] = None,
) -> FastAnswerResult:
    """
    Evaluate a user message and determine if it can be answered quickly.

    Returns FastAnswerResult with decision and optional response.
    On any error, defers to the vision agent.
    """
    prompt_text = FAST_ANSWER_PROMPT.format(
        directive=directive or "No directive set",
        current_primitive=current_primitive or "None",
        history_summary=history_summary or "No recent history",
        user_message=user_message,
    )

    contents = []
    if current_image:
        contents.append("This is the current camera view from the robot:")
        contents.append(_prepare_image_part(current_image))
    contents.append(prompt_text)

    try:
        response = await asyncio.to_thread(
            _get_client().models.generate_content,
            model=FAST_MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=FAST_AGENT_TEMPERATURE,
                max_output_tokens=FAST_AGENT_MAX_OUTPUT_TOKENS,
                response_mime_type="application/json",
            ),
        )

        result = json.loads(response.text)
        decision = FastAnswerDecision(result.get("decision", "defer_to_vision").lower())
        return FastAnswerResult(
            decision=decision,
            response=result.get("response")
            if decision == FastAnswerDecision.ANSWER_NOW
            else None,
            reasoning=result.get("reasoning"),
        )
    except Exception as e:
        print(f"[FastAgent] Error: {e}")
        return FastAnswerResult(
            decision=FastAnswerDecision.DEFER_TO_VISION,
            reasoning=f"Fast agent error: {e}",
        )
