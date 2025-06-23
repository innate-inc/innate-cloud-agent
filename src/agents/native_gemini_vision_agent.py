import os
import json
import base64
import asyncio
from typing import Optional, List, Dict, Any, Union, Literal
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

from google import genai
from google.genai import types

from src.agents.types import MultimodalVisionAgentInput, PrimitiveDefinition
from src.agents.exceptions import MaxRetriesExceededException, UnforeseenBamlClientError
from src.agents.native_gemini_schema_builder import (
    create_gemini_schema,
    create_response_model,
    VisionAgentOutput,
    convert_to_brain_compatible_output,
)

# Gemini API constants (matching BAML configuration)
GEMINI_MODEL_NAME = "gemini-2.5-flash-preview-05-20"
GEMINI_TEMPERATURE = 0
GEMINI_TOP_P = 0.95
GEMINI_TOP_K = 64
GEMINI_MAX_OUTPUT_TOKENS = 8192
THINKING_BUDGET = 0  # Matching BAML config
EXECUTION_TIMEOUT = 5  # seconds

# Debug settings
SAVE_DEBUG_DATA = True
DEBUG_DATA_DIR = Path("test_data/debug_native_gemini_images")


class NativeVisionAgentOutput(BaseModel):
    """
    Native output model matching NewVisionAgentOutput structure from BAML.
    """

    model_config = ConfigDict(extra="allow")

    current_observation: str
    current_thoughts: str
    action_decision: Literal["continue", "change_task", "stop_task", "start_task"]
    to_tell_user: Optional[str] = None
    next_task: Optional[Any] = (
        None  # Will be dynamically typed based on available primitives
    )


class LegacyVisionAgentOutput(BaseModel):
    """
    Legacy output model matching VisionAgentOutput structure from BAML.
    This is what the vision_service.py expects.
    """

    model_config = ConfigDict(extra="allow")

    stop_current_task: bool
    observation: str
    thoughts: str
    new_goal: Optional[str] = None
    anticipation: Optional[str] = None
    to_tell_user: Optional[str] = None
    next_task: Optional[Any] = (
        None  # Will be dynamically typed based on available primitives
    )


def convert_to_legacy_output(
    native_output: NativeVisionAgentOutput,
) -> LegacyVisionAgentOutput:
    """
    Convert NativeVisionAgentOutput to LegacyVisionAgentOutput for backward compatibility.
    """
    # Convert action_decision to stop_current_task boolean
    stop_current_task = native_output.action_decision in ["stop_task", "change_task"]

    return LegacyVisionAgentOutput(
        stop_current_task=stop_current_task,
        observation=native_output.current_observation,
        thoughts=native_output.current_thoughts,
        new_goal=None,  # Not present in new output
        anticipation=None,  # Not present in new output
        to_tell_user=native_output.to_tell_user,
        next_task=native_output.next_task,
    )


class NativeGeminiVisionAgent:
    """
    Native Google Gemini implementation replacing BAML-based vision agent.
    """

    def __init__(self):
        """Initialize the native Gemini client."""
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the Google Gemini client."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        try:
            self.client = genai.Client(api_key=api_key)
            print("Native Gemini client initialized successfully.")
        except Exception as e:
            raise ValueError(f"Failed to initialize Gemini client: {e}")

    def _prepare_multimodal_content(
        self, vlm_inputs: MultimodalVisionAgentInput
    ) -> List[Union[str, dict]]:
        """
        Prepare multimodal content for Gemini API call.

        Args:
            vlm_inputs: Input data containing images and text

        Returns:
            List of content parts for Gemini API
        """
        content_parts = []

        # Add the main prompt based on BAML template
        main_prompt = self._build_main_prompt(vlm_inputs)
        content_parts.append(main_prompt)

        # Process multimodal history
        for history_item in vlm_inputs.multimodal_history:
            if history_item.type == "image":
                # Convert base64 image to genai.types.Part format
                try:
                    image_data = base64.b64decode(history_item.content)
                    image_part = types.Part.from_bytes(
                        data=image_data,
                        mime_type="image/jpeg",
                    )
                    content_parts.append(image_part)
                except Exception as e:
                    print(f"Error processing history image: {e}")
                    # Skip invalid images
                    continue
            else:  # text
                content_parts.append(history_item.content)

        # Add the main camera image
        try:
            # Remove data URL prefix if present
            img_data = vlm_inputs.base64_img
            if img_data.startswith("data:image"):
                img_data = img_data.split(",")[1]

            # Convert to bytes and create Part
            image_bytes = base64.b64decode(img_data)
            main_image_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg",
            )
            content_parts.append("This is what you see:")
            content_parts.append(main_image_part)
        except Exception as e:
            print(f"Error processing main image: {e}")
            content_parts.append("Main camera image could not be processed.")

        # Add additional camera images if present
        if vlm_inputs.additional_image_data:
            camera_type = vlm_inputs.additional_image_data.get("camera_type", "unknown")
            image_b64 = vlm_inputs.additional_image_data.get("image_b64", "")

            if image_b64:
                try:
                    # Remove data URL prefix if present
                    if image_b64.startswith("data:image"):
                        image_b64 = image_b64.split(",")[1]

                    # Convert to bytes and create Part
                    image_bytes = base64.b64decode(image_b64)
                    additional_image_part = types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/jpeg",
                    )
                    content_parts.append(
                        f"On top of that, this is what you see from the additional camera with type: {camera_type}"
                    )
                    content_parts.append(additional_image_part)
                except Exception as e:
                    print(f"Error processing additional image: {e}")
                    content_parts.append(
                        f"Additional camera ({camera_type}) image could not be processed."
                    )

        return content_parts

    def _build_main_prompt(self, vlm_inputs: MultimodalVisionAgentInput) -> str:
        """
        Build the main prompt based on BAML template.

        Args:
            vlm_inputs: Input data

        Returns:
            Main prompt string
        """
        # Start with the core role and goal definition
        prompt_parts = [
            "You are a robot navigating and executing tasks in a home.",
            "",
            "Your goal is to decide what to do right now based on the image and the context.",
            "",
            "You are being provided with what the user most recently said, the task that is currently being executed, and the image you see in front of you.",
            "",
            "If there is no task currently being executed, you can choose to do one next or nothing depending on the context.",
            "TO STOP THE CURRENT TASK, YOU HAVE TO HAVE BEEN EXPLICITLY TOLD TO DO IT BY THE USER OR BE IN A SITUATION THAT CLEARLY NEEDS YOU TO STOP ACCORDING TO YOUR DIRECTIVE.",
            "",
            "THIS IS VERY IMPORTANT. I REPEAT, DO NOT STOP THE CURRENT TASK UNLESS IT'S REQUESTED BY THE USER OR IT'S CLEARLY STATED IN YOUR DIRECTIVE.",
            "",
        ]

        # Add user message context
        if vlm_inputs.user_prompt_text:
            prompt_parts.append(f"The user said: {vlm_inputs.user_prompt_text}")
        else:
            prompt_parts.append("The user did not say anything.")
        prompt_parts.append("")

        # Add current primitive context
        if vlm_inputs.primitive_in_execution:
            prompt_parts.append(
                f"The current task is: {vlm_inputs.primitive_in_execution.model_dump_json()}"
            )
        else:
            prompt_parts.append("You are not currently executing a task.")
        prompt_parts.append("")

        # Add robot coordinates if available
        if vlm_inputs.robot_coords:
            coords = vlm_inputs.robot_coords
            prompt_parts.append(
                f"Your coordinates if useful to know are: "
                f"x={coords.get('x')}, y={coords.get('y')}, "
                f"z={coords.get('z')}, theta={coords.get('theta')}"
            )
            prompt_parts.append("")

        # Add directive if present
        if vlm_inputs.directive:
            prompt_parts.append(f"Your directive is: {vlm_inputs.directive}")
            prompt_parts.append("")

        # Add running primitive guidelines
        if (
            vlm_inputs.primitive_in_execution
            and vlm_inputs.primitive_in_execution.guidelines_when_running
        ):
            prompt_parts.extend(
                [
                    "Here are the guidelines for the task currently running. Watch them carefully:",
                    vlm_inputs.primitive_in_execution.guidelines_when_running,
                    "",
                ]
            )

        # Add task completion evaluation guidelines
        prompt_parts.extend(
            [
                "Consider the guidelines of the tasks available carefully before picking one to do (if any).",
                "",
                "Also, evaluate carefully if you need to tell something to the user, especially if the last time you talked with them was recently. The history context indicates if you're still talking.",
                "If the last time you talked with them was seconds ago and you expect some answer from them, you should wait for example, an amount of time that seems appropriate.",
                "",
                "ONCE A TASK IS COMPLETED, EVALUATE IF YOU NEED TO START IT AGAIN. COMPLETED MEANS AN ACTION IS OVER, IT **DOES NOT MEAN** THAT IT WAS SUCCESSFUL, YOU HAVE TO FIGURE IT OUT.",
                "",
            ]
        )

        # Add navigation-specific guidelines
        prompt_parts.extend(
            [
                "Guidelines for navigation and positioning:",
                "- IN THE CASE OF NAVIGATION, A TASK MIGHT NEED TO BE STARTED AGAIN TO GET CLOSER OR PURSUE THE OBJECTIVE.",
                "- WHEN NAVIGATION IN SIGHT IS COMPLETED, IT DOES NOT NECESSARILY MEAN YOU SHOULD STOP. YOU MIGHT NEED TO GET CLOSER OR PURSUE THE OBJECTIVE.",
                "- IF YOU NEED TO BE IN FRONT OF A TARGET TO EXECUTE A PHYSICAL TASK, YOU SHOULD USE CHECK_DISTANCE_AND_ORIENTATION TO MAKE SURE YOU ARE IN THE RIGHT POSITION.",
                "",
            ]
        )

        # Add available primitives
        primitive_names = [prim.name for prim in vlm_inputs.primitives_list]
        primitives_list_string = ", ".join(primitive_names)
        prompt_parts.append(
            f"You can only use one of the following tasks: {primitives_list_string}."
        )
        prompt_parts.append("")

        # Add field usage instructions
        prompt_parts.extend(
            [
                'You should use the field "observation" to describe what you see in the image as an internal thought.',
                'You should use the field "thoughts" to further think about what you should do (or not do) based on the observation and the context.',
                'You should use the field "stop_current_task" to decide whether to stop the current task.',
                'You should use the field "anticipation" to consider what might happen next and leave mental notes for you in the future.',
                'You should use the field "to_tell_user" if you need to communicate something to the user.',
                'You should use the field "next_task" to specify which task to execute next (if any).',
                "",
            ]
        )

        return "\n".join(prompt_parts)

    async def call_gemini_api(
        self, vlm_inputs: MultimodalVisionAgentInput
    ) -> "VisionAgentOutput":
        """
        Make the actual Gemini API call with structured output.

        Args:
            vlm_inputs: Input data for the vision agent

        Returns:
            Parsed response from Gemini
        """
        # Prepare multimodal content
        content_parts = self._prepare_multimodal_content(vlm_inputs)

        # Create response schema (Pydantic model)
        response_schema = create_gemini_schema(vlm_inputs.primitives_list)

        # Make the API call using the new genai.Client API
        generation_config = types.GenerateContentConfig(
            temperature=GEMINI_TEMPERATURE,
            top_p=GEMINI_TOP_P,
            top_k=GEMINI_TOP_K,
            max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
            response_mime_type="application/json",
            response_schema=response_schema,
        )

        # Use asyncio.to_thread to run the synchronous call in a thread
        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=GEMINI_MODEL_NAME,
            contents=content_parts,
            config=generation_config,
        )

        # Parse the response and convert to brain-compatible format
        try:
            parsed_response = response.parsed

            # Convert to brain-compatible output to prevent type mismatch warnings
            brain_compatible_output = convert_to_brain_compatible_output(
                parsed_response
            )

            return brain_compatible_output
        except Exception as e:
            raise ValueError(
                f"Failed to parse Gemini response: {e}, Response: {response.text}"
            )


async def native_gemini_vision_agent_multimodal_history(
    vlm_inputs: MultimodalVisionAgentInput,
) -> Optional[VisionAgentOutput]:
    """
    Native Gemini implementation of the multimodal vision agent.

    This function replaces gemini_vision_agent_multimodal_history from BAML,
    providing the same interface but using native Google Gemini API calls.

    Args:
        vlm_inputs: The MultimodalVisionAgentInput containing all required inputs

    Returns:
        VisionAgentOutput: The standardized output matching the original BAML format

    Raises:
        MaxRetriesExceededException: When the maximum number of retries is exceeded
        UnforeseenBamlClientError: When an unexpected client error occurs
    """
    # Save debug data if enabled
    if SAVE_DEBUG_DATA:
        # Save main image
        if vlm_inputs.base64_img:
            _save_base64_image(vlm_inputs.base64_img, "main_image")

        # Save additional images if present
        if vlm_inputs.additional_image_data:
            camera_type = vlm_inputs.additional_image_data.get("camera_type", "unknown")
            image_b64 = vlm_inputs.additional_image_data.get("image_b64", "")
            if image_b64:
                _save_base64_image(image_b64, f"additional_{camera_type}")

    # Initialize the agent
    agent = NativeGeminiVisionAgent()

    try:
        # Call with retry logic
        native_output = await _call_with_retry_logic(agent, vlm_inputs, max_retries=10)

        # Save context data if debug is enabled
        if SAVE_DEBUG_DATA:
            context_data = {
                "user_prompt_text": vlm_inputs.user_prompt_text,
                "primitive_in_execution": (
                    vlm_inputs.primitive_in_execution.model_dump()
                    if vlm_inputs.primitive_in_execution
                    else None
                ),
                "robot_coords": vlm_inputs.robot_coords,
                "directive": vlm_inputs.directive,
                "primitives_count": len(vlm_inputs.primitives_list),
                "multimodal_history_length": len(vlm_inputs.multimodal_history),
                "native_output": native_output.model_dump(),
            }
            _save_context_data(context_data, "native_context_data")

        # Return VisionAgentOutput directly (no conversion needed)
        return native_output

    except MaxRetriesExceededException as e:
        raise e
    except UnforeseenBamlClientError as e:
        raise e
    except Exception as e:
        print(f"Unexpected error in native Gemini API call: {e}")
        raise UnforeseenBamlClientError(
            f"Unexpected error in native Gemini agent: {e}",
            original_error=e,
        )


async def _call_with_retry_logic(
    agent: NativeGeminiVisionAgent,
    vlm_inputs: MultimodalVisionAgentInput,
    max_retries: int,
    attempt: int = 1,
) -> VisionAgentOutput:
    """
    Recursively attempts to call the native Gemini API until a successful output
    is produced or max_retries is exhausted. This mirrors the BAML retry logic.

    Args:
        agent: The initialized Gemini agent
        vlm_inputs: Input data for the API call
        max_retries: Maximum number of retry attempts
        attempt: Current attempt number

    Returns:
        VisionAgentOutput: Successful response from Gemini

    Raises:
        MaxRetriesExceededException: When max retries are exceeded
        UnforeseenBamlClientError: When an unexpected error occurs
    """

    async def recall():
        """Helper function for recursive retry calls."""
        return await _call_with_retry_logic(
            agent=agent,
            vlm_inputs=vlm_inputs,
            max_retries=max_retries,
            attempt=attempt + 1,
        )

    try:
        # Make the API call with timeout
        native_output = await agent.call_gemini_api(vlm_inputs)
        return native_output

    except asyncio.TimeoutError:
        error_msg = (
            f"Operation timed out after {EXECUTION_TIMEOUT} seconds "
            f"on attempt {attempt}/{max_retries}"
        )
        print(f"\033[1;31m{error_msg}\033[0m")

        if attempt == max_retries:
            raise MaxRetriesExceededException(
                agent_type="native_gemini_vision_agent",
                max_retries=max_retries,
                last_error=asyncio.TimeoutError(
                    f"Native Gemini API call exceeded {EXECUTION_TIMEOUT} second timeout"
                ),
            )

        await asyncio.sleep(1)
        return await recall()

    except json.JSONDecodeError as e:
        error_msg = f"JSON decode error on attempt {attempt}/{max_retries}: {e}"
        print(f"\033[1;31m{error_msg}\033[0m")

        if attempt == max_retries:
            raise MaxRetriesExceededException(
                agent_type="native_gemini_vision_agent",
                max_retries=max_retries,
                last_error=e,
            )

        await asyncio.sleep(1)
        return await recall()

    except ValueError as e:
        # This includes Gemini response parsing errors
        error_msg = f"Value error on attempt {attempt}/{max_retries}: {e}"
        print(f"\033[1;31m{error_msg}\033[0m")

        if attempt == max_retries:
            raise MaxRetriesExceededException(
                agent_type="native_gemini_vision_agent",
                max_retries=max_retries,
                last_error=e,
            )

        await asyncio.sleep(1)
        return await recall()

    except Exception as e:
        # Handle different types of errors similar to BAML implementation
        error_str = str(e)

        def error_msg_func(e_val):
            return (
                f"\033[1;31mNative Gemini error on attempt {attempt}/{max_retries}: "
                f"{e_val}\033[0m"
            )

        # Handle connection timeout errors
        if "timed out" in error_str.lower() or "timeout" in error_str.lower():
            print(error_msg_func("Timeout"))
            if attempt < max_retries:
                await asyncio.sleep(1)
                return await recall()
            else:
                raise MaxRetriesExceededException(
                    agent_type="native_gemini_vision_agent",
                    max_retries=max_retries,
                    last_error=e,
                )

        # Handle SSL errors
        if "ssl" in error_str.lower() or "certificate" in error_str.lower():
            print(error_msg_func("SSL Error"))
            if attempt < max_retries:
                await asyncio.sleep(1)
                return await recall()
            else:
                raise MaxRetriesExceededException(
                    agent_type="native_gemini_vision_agent",
                    max_retries=max_retries,
                    last_error=e,
                )

        # Handle HTTP 503 errors (service unavailable)
        if "503" in error_str or "service unavailable" in error_str.lower():
            print(error_msg_func("503 Service Unavailable"))
            if attempt < max_retries:
                await asyncio.sleep(1)
                return await recall()
            else:
                raise MaxRetriesExceededException(
                    agent_type="native_gemini_vision_agent",
                    max_retries=max_retries,
                    last_error=e,
                )

        # Handle HTTP 429 errors (rate limiting)
        if "429" in error_str or "rate limit" in error_str.lower():
            print(error_msg_func("429 Rate Limited"))
            if attempt < max_retries:
                # Longer sleep for rate limiting
                await asyncio.sleep(2)
                return await recall()
            else:
                raise MaxRetriesExceededException(
                    agent_type="native_gemini_vision_agent",
                    max_retries=max_retries,
                    last_error=e,
                )

        # Handle HTTP 500 errors (internal server error)
        if "500" in error_str or "internal server error" in error_str.lower():
            print(error_msg_func("500 Internal Server Error"))
            if attempt < max_retries:
                await asyncio.sleep(1)
                return await recall()
            else:
                raise MaxRetriesExceededException(
                    agent_type="native_gemini_vision_agent",
                    max_retries=max_retries,
                    last_error=e,
                )

        # Handle quota exceeded errors
        if "quota" in error_str.lower() or "exceeded" in error_str.lower():
            print(error_msg_func("Quota Exceeded"))
            if attempt < max_retries:
                # Longer sleep for quota issues
                await asyncio.sleep(3)
                return await recall()
            else:
                raise MaxRetriesExceededException(
                    agent_type="native_gemini_vision_agent",
                    max_retries=max_retries,
                    last_error=e,
                )

        # Handle any other unforeseen errors
        print(error_msg_func(str(e)))
        raise UnforeseenBamlClientError(
            f"Unforeseen error in native Gemini agent on attempt {attempt}/{max_retries}: {e}",
            original_error=e,
        )


# Utility functions for debug data saving (ported from BAML implementation)


def _save_base64_image(base64_img: str, filename: str) -> str:
    """
    Save a base64 encoded image to a file.
    """
    try:
        DEBUG_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Remove data URL prefix if present
        if base64_img.startswith("data:image"):
            base64_img = base64_img.split(",")[1]

        # Decode base64 data
        image_data = base64.b64decode(base64_img)

        # Determine file extension from image header
        if image_data.startswith(b"\xff\xd8\xff"):
            ext = ".jpg"
        elif image_data.startswith(b"\x89PNG"):
            ext = ".png"
        elif image_data.startswith(b"GIF"):
            ext = ".gif"
        elif image_data.startswith(b"RIFF"):
            ext = ".webp"
        else:
            ext = ".jpg"  # Default

        # Create full file path
        file_path = (
            DEBUG_DATA_DIR
            / f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        )

        # Write image data
        with open(file_path, "wb") as f:
            f.write(image_data)

        print(f"Saved debug image to {file_path}")
        return str(file_path)

    except Exception as e:
        print(f"Error saving debug image: {e}")
        return ""


def _save_context_data(context_data: dict, filename: str) -> str:
    """
    Save context data to a text file.
    """
    try:
        DEBUG_DATA_DIR.mkdir(parents=True, exist_ok=True)

        file_path = (
            DEBUG_DATA_DIR
            / f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("=== NATIVE GEMINI AGENT CONTEXT DATA ===\n\n")
            for key, value in context_data.items():
                f.write(f"{key.upper()}:\n")
                if isinstance(value, (list, dict)):
                    f.write(json.dumps(value, indent=2, ensure_ascii=False))
                else:
                    f.write(str(value))
                f.write("\n\n")

        print(f"Saved context data to {file_path}")
        return str(file_path)

    except Exception as e:
        print(f"Error saving context data: {e}")
        return ""
