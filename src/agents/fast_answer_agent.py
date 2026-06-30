"""
Fast Answer Agent - A lightweight agent for quick responses.

This agent answers simple questions quickly using Gemini 3.1 Flash-Lite with the
current camera image. It runs in parallel with the main vision agent.
"""

import os
import json
import asyncio
import base64
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

from google import genai
from google.genai import types

from src.agents.types import PrimitiveDefinition


FAST_MODEL_NAME = "gemini-3.1-flash-lite"
FAST_AGENT_TEMPERATURE = 0.3
FAST_AGENT_MAX_OUTPUT_TOKENS = 512


class FastAnswerDecision(str, Enum):
    """Decision made by the fast answer agent."""

    ANSWER_NOW = "answer_now"
    DEFER_TO_EXPERT = "defer_to_expert"


@dataclass
class FastAnswerResult:
    """Result from the fast answer agent."""

    decision: FastAnswerDecision
    response: Optional[str] = None
    reasoning: Optional[str] = None


FAST_ANSWER_PROMPT = """You are a robot assistant. Answer the user's question if possible, or defer to the expert if movement/actions are needed.

Context: {directive} | Running: {current_primitive} | History: {history_summary}
Available primitives: {primitives_list}

User: {user_message}

Rules:
- ANSWER_NOW: questions, conversation, basic what you see in the image
- DEFER_TO_EXPERT: movement commands, navigation, multi-step tasks, requests that need primitives

Respond as JSON: {{"decision": "ANSWER_NOW" or "DEFER_TO_EXPERT", "response": "your answer if ANSWER_NOW", "reasoning": "why"}}"""


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
    primitives_list: Optional[List[PrimitiveDefinition]] = None,
) -> FastAnswerResult:
    """
    Evaluate a user message and determine if it can be answered quickly.

    Returns FastAnswerResult with decision and optional response.
    On any error, defers to the vision agent.
    """
    # Format primitives list as comma-separated names
    primitives_str = "None"
    if primitives_list:
        primitives_str = ", ".join(p.name for p in primitives_list)

    prompt_text = FAST_ANSWER_PROMPT.format(
        directive=directive or "No directive set",
        current_primitive=current_primitive or "None",
        history_summary=history_summary or "No recent history",
        primitives_list=primitives_str,
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
        decision = FastAnswerDecision(result.get("decision", "defer_to_expert").lower())
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
            decision=FastAnswerDecision.DEFER_TO_EXPERT,
            reasoning=f"Fast agent error: {e}",
        )
