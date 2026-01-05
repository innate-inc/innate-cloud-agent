"""
Fast Answer Agent - A lightweight agent for quick text-only responses.

This agent is designed to answer simple questions quickly without requiring
the full vision pipeline. It runs in parallel with the main vision agent,
and if the question is simple, it responds immediately. If the question
requires movement or complex reasoning, it defers to the main vision agent.
"""

import os
import asyncio
from typing import Optional
from dataclasses import dataclass
from enum import Enum

from google import genai
from google.genai import types


# Use a fast model for quick responses
FAST_MODEL_NAME = "gemini-2.0-flash-lite"
FAST_AGENT_TEMPERATURE = 0.3
FAST_AGENT_MAX_OUTPUT_TOKENS = 512


class FastAnswerDecision(str, Enum):
    """Decision made by the fast answer agent."""

    ANSWER_NOW = "answer_now"  # Can answer immediately
    DEFER_TO_VISION = "defer_to_vision"  # Needs vision/movement, defer to main agent


@dataclass
class FastAnswerResult:
    """Result from the fast answer agent."""

    decision: FastAnswerDecision
    response: Optional[str] = None  # The response to send if decision is ANSWER_NOW
    reasoning: Optional[str] = None  # Why the decision was made


FAST_ANSWER_PROMPT = """You are a fast-response assistant for a robot. Your job is to quickly determine if a user's question can be answered immediately with text, or if it requires the robot to look at its camera, move, or perform actions.

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
- A clarification question that doesn't need visual input
- A question you can answer based on the provided context/history

Answer "DEFER_TO_VISION" if the question:
- Asks about what the robot sees (e.g., "What do you see?", "Is there a chair?")
- Requires physical movement (e.g., "Go to the kitchen", "Turn left")
- Asks about the robot's current location or surroundings
- Requires executing a primitive/action
- Is ambiguous and might need visual context to answer properly
- Relates to navigation or spatial reasoning
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


class FastAnswerAgent:
    """
    A lightweight agent for quick text-only responses.

    This agent determines if a user's question can be answered immediately
    without requiring the full vision pipeline.
    """

    def __init__(self):
        """Initialize the fast answer agent."""
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the Google Gemini client."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        try:
            self.client = genai.Client(api_key=api_key)
        except Exception as e:
            raise ValueError(f"Failed to initialize Gemini client: {e}")

    async def evaluate_and_respond(
        self,
        user_message: str,
        directive: Optional[str] = None,
        current_primitive: Optional[str] = None,
        history_summary: Optional[str] = None,
    ) -> FastAnswerResult:
        """
        Evaluate a user message and determine if it can be answered quickly.

        Args:
            user_message: The user's message/question
            directive: The current directive guiding the robot
            current_primitive: Name of the currently executing primitive
            history_summary: Brief summary of recent history

        Returns:
            FastAnswerResult with decision and optional response
        """
        import time
        start_time = time.time()
        print(f"[FastAgent] Starting evaluation for: '{user_message[:50]}...'")

        # Format the prompt
        prompt = FAST_ANSWER_PROMPT.format(
            directive=directive or "No directive set",
            current_primitive=current_primitive or "None",
            history_summary=history_summary or "No recent history",
            user_message=user_message,
        )

        try:
            # Make the API call
            generation_config = types.GenerateContentConfig(
                temperature=FAST_AGENT_TEMPERATURE,
                max_output_tokens=FAST_AGENT_MAX_OUTPUT_TOKENS,
                response_mime_type="application/json",
            )

            api_start = time.time()
            print(f"[FastAgent] Calling Gemini API ({FAST_MODEL_NAME})...")
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=FAST_MODEL_NAME,
                contents=prompt,
                config=generation_config,
            )
            api_elapsed = time.time() - api_start
            print(f"[FastAgent] API call completed in {api_elapsed:.2f}s")

            # Parse the response
            import json

            try:
                result = json.loads(response.text)
                decision = FastAnswerDecision(result.get("decision", "defer_to_vision").lower())
                total_elapsed = time.time() - start_time
                print(f"[FastAgent] Decision: {decision.value} in {total_elapsed:.2f}s total")
                return FastAnswerResult(
                    decision=decision,
                    response=result.get("response") if decision == FastAnswerDecision.ANSWER_NOW else None,
                    reasoning=result.get("reasoning"),
                )
            except (json.JSONDecodeError, ValueError) as e:
                # If parsing fails, defer to vision agent
                total_elapsed = time.time() - start_time
                print(f"[FastAgent] Parse error in {total_elapsed:.2f}s: {e}")
                return FastAnswerResult(
                    decision=FastAnswerDecision.DEFER_TO_VISION,
                    reasoning=f"Failed to parse fast agent response: {e}",
                )

        except Exception as e:
            # On any error, defer to the vision agent
            total_elapsed = time.time() - start_time
            print(f"[FastAgent] Error in {total_elapsed:.2f}s: {e}")
            return FastAnswerResult(
                decision=FastAnswerDecision.DEFER_TO_VISION,
                reasoning=f"Fast agent error: {e}",
            )


# Singleton instance for reuse
_fast_answer_agent: Optional[FastAnswerAgent] = None


def get_fast_answer_agent() -> FastAnswerAgent:
    """Get or create the singleton FastAnswerAgent instance."""
    global _fast_answer_agent
    if _fast_answer_agent is None:
        _fast_answer_agent = FastAnswerAgent()
    return _fast_answer_agent


async def fast_answer(
    user_message: str,
    directive: Optional[str] = None,
    current_primitive: Optional[str] = None,
    history_summary: Optional[str] = None,
) -> FastAnswerResult:
    """
    Convenience function to evaluate and respond to a user message quickly.

    Args:
        user_message: The user's message/question
        directive: The current directive guiding the robot
        current_primitive: Name of the currently executing primitive
        history_summary: Brief summary of recent history

    Returns:
        FastAnswerResult with decision and optional response
    """
    agent = get_fast_answer_agent()
    return await agent.evaluate_and_respond(
        user_message=user_message,
        directive=directive,
        current_primitive=current_primitive,
        history_summary=history_summary,
    )

