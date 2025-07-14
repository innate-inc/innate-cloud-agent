import os
import json
import base64
import asyncio
from typing import Optional, List, Union
from pathlib import Path
from datetime import datetime

from google import genai
from google.genai import types

from src.agents.types import MultimodalVisionAgentInput
from src.agents.exceptions import MaxRetriesExceededException, UnforeseenBamlClientError
from src.agents.native_gemini_schema_builder import (
    create_gemini_schema,
    VisionAgentOutput,
    convert_to_brain_compatible_output,
)
from src.agents.debug_html_generator import save_content_parts_html
from src.constants_robots import ROBOT_PARAMS_TO_USE

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
DEBUG_DATA_DIR = Path("test_data/debug_native_gemini_html")

# Template for the main prompt
VISION_AGENT_PROMPT_TEMPLATE = """<system_role>
You are a robot navigating and executing primitives in a home.

You are following a directive (defined in <directive>) that guides your actions, and you can pick primitives to execute to achieve your goal.

You have to decide what to do right now based on the current image you see (in <main_camera_image>), the history of your actions and observations (in <history_of_events>), and the current primitive that is being executed (in <primitive_in_execution>).

You are also being provided with what the user most recently said (in <user_input>).
</system_role>

<operational_guidelines>
<choosing_next_primitive>
**IF NO PRIMITIVE IS RUNNING:**
- Look at your directive and what you see in the image
- Choose the primitive that makes the most progress toward your goal
- If the user just gave you a command, prioritize that
- You don't have to start a new primitive if you think you should stay idle
</choosing_next_primitive>

<stopping_running_primitives>
**IF A PRIMITIVE IS CURRENTLY RUNNING:**
Only stop it if:
- The user explicitly told you to stop it.
- Your directive clearly requires stopping it.
- You clearly can assess the primitive has completed its goal.
- You clearly can assess that something is wrong and you need to stop it.

**DO NOT STOP** running primitives for any other reason. When in doubt, let it continue.
</stopping_running_primitives>

<communication>
**TALK TO THE USER** when:
- They just spoke to you and expect a response
- You're in a situation where the directive requires you to communicate with the user

**WAIT** if you just spoke to them seconds ago and they might still be responding. The history of events indicates if you're still talking. Do not talk over yourself!
</communication>

<navigation_rules>
- Navigation primitives allow you to get closer to your objective but a completion of a navigation primitive does not mean you're done. You might need to get closer or pursue the navigation objective.
- A navigation primitive can indicate when it's close to being completed. When that is the case, if you think you need to navigate again, you should stop the current navigation primitive and start a new one.
- You are provided with previous images of what you saw in <history_of_events>. Pay attention to them when pursuing several navigation primitives.
- Your horizontal field of view is {field_of_view}, keep that in mind when turning. Too big of a turn can make you lose sight of something important, but too small might just make you be very slow.
</navigation_rules>

<awareness_rules>
- Pay attention if your <history_of_events> indicates you are stuck or repeating the same actions without progress. If that is the case, try to change your approach.
- If you seem stuck for more than 30 seconds, this where you should start acting and changing actions or plan.
</awareness_rules>

<planning_rules>
The fields observation, thoughts, anticipation are here to help you keep track of a bigger plan to achieve your directive. You can use them to plan your next actions, but you can also completely change your plan if you think you should.
</planning_rules>

<speed_rules>
Unless precised by the directive or user, decision-making should be done fast especially when pursuing a navigation objective.
</speed_rules>
</operational_guidelines>

<current_context>
<history_of_events>
{multimodal_history}
</history_of_events>

<main_camera_image>
{main_camera_image}
</main_camera_image>

<user_input>
{user_input}
</user_input>

<primitive_in_execution>
{primitive_in_execution}
</primitive_in_execution>

<robot_position>
{robot_position}
</robot_position>

<directive>
{directive}
</directive>

<current_primitive_guidelines>
{current_primitive_guidelines}
</current_primitive_guidelines>

{additional_camera_image}
</current_context>

<available_primitives>
You can only use one of the following primitives: {available_primitives}.
</available_primitives>

<response_requirements>
Use the following fields in your response:

- "observation": Describe what you see in the image as an internal thought
- "thoughts": Think about what you should do (or not do) based on the observation and context
- "stop_current_primitive": Decide whether to stop the current primitive
- "anticipation": Consider what might happen next and leave mental notes for future reference
- "to_tell_user": Communicate something to the user (if needed)
- "next_primitive": Specify which primitive to execute next (if any)
</response_requirements>
"""


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
        Prepare multimodal content for Gemini API call using integrated template formatting.

        Args:
            vlm_inputs: Input data containing images and text

        Returns:
            List of content parts for Gemini API
        """
        return self._format_multimodal_template(vlm_inputs)

    def _format_multimodal_template(
        self, vlm_inputs: MultimodalVisionAgentInput
    ) -> List[Union[str, dict]]:
        """
        Format the multimodal template with integrated image and text content.

        This function handles special placeholders in the template and returns
        a list of content parts that preserves the order and context.

        Args:
            vlm_inputs: Input data

        Returns:
            List of content parts for Gemini API
        """
        content_parts = []

        # Prepare all the template variables
        template_vars = self._prepare_template_variables(vlm_inputs)

        # Split the template into sections and process each part
        template_sections = self._split_template_by_multimodal_placeholders(
            VISION_AGENT_PROMPT_TEMPLATE
        )

        for section in template_sections:
            if section["type"] == "text":
                # Format regular text section with non-multimodal variables
                formatted_text = section["content"].format(**template_vars["text_vars"])
                if formatted_text.strip():  # Only add non-empty text
                    content_parts.append(formatted_text)

            elif section["type"] == "multimodal_history":
                # Add multimodal history content
                content_parts.extend(template_vars["multimodal_history_parts"])

            elif section["type"] == "main_camera_image":
                # Add main camera image with description
                if template_vars["main_camera_image_part"]:
                    content_parts.append("This is what you see:")
                    content_parts.append(template_vars["main_camera_image_part"])
                else:
                    content_parts.append("Main camera image could not be processed.")

            elif section["type"] == "additional_camera_image":
                # Add additional camera image if present
                if template_vars["additional_camera_parts"]:
                    content_parts.extend(template_vars["additional_camera_parts"])

        return content_parts

    def _prepare_template_variables(
        self, vlm_inputs: MultimodalVisionAgentInput
    ) -> dict:
        """
        Prepare all template variables including multimodal content.

        Args:
            vlm_inputs: Input data

        Returns:
            Dictionary containing all template variables and multimodal parts
        """
        # Prepare multimodal history parts
        multimodal_history_parts = []
        history_text_parts = []

        for history_item in vlm_inputs.multimodal_history:
            if history_item.type == "image":
                # Convert base64 image to genai.types.Part format
                try:
                    image_data = base64.b64decode(history_item.content)
                    image_part = types.Part.from_bytes(
                        data=image_data,
                        mime_type="image/jpeg",
                    )
                    multimodal_history_parts.append(image_part)
                except Exception as e:
                    print(f"Error processing history image: {e}")
                    # Skip invalid images
                    continue
            else:  # text
                history_text_parts.append(history_item.content)
                multimodal_history_parts.append(history_item.content)

        # Prepare history text for template
        history_of_events = ""
        if history_text_parts:
            history_of_events = chr(10).join(history_text_parts)

        # Prepare main camera image
        main_camera_image_part = None
        try:
            # Remove data URL prefix if present
            img_data = vlm_inputs.base64_img
            if img_data.startswith("data:image"):
                img_data = img_data.split(",")[1]

            # Convert to bytes and create Part
            image_bytes = base64.b64decode(img_data)
            main_camera_image_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg",
            )
        except Exception as e:
            print(f"Error processing main image: {e}")

        # Prepare additional camera images
        additional_camera_parts = []
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
                    additional_camera_parts.append(
                        f"On top of that, this is what you see from the additional camera with type: {camera_type}"
                    )
                    additional_camera_parts.append(additional_image_part)
                except Exception as e:
                    print(f"Error processing additional image: {e}")
                    additional_camera_parts.append(
                        f"Additional camera ({camera_type}) image could not be processed."
                    )

        # Prepare user message context
        if vlm_inputs.user_prompt_text:
            user_input = f"The user said: {vlm_inputs.user_prompt_text}"
        else:
            user_input = "The user did not say anything."

        # Prepare current primitive context
        if vlm_inputs.primitive_in_execution:
            current_primitive = f"The current primitive is: {vlm_inputs.primitive_in_execution.model_dump_json()}"
        else:
            current_primitive = "You are not currently executing a primitive."

        # Prepare robot coordinates section
        robot_coordinates = ""
        if vlm_inputs.robot_coords:
            coords = vlm_inputs.robot_coords
            robot_coordinates = f"Your coordinates if useful to know are: x={coords.get('x')}, y={coords.get('y')}, z={coords.get('z')}, theta={coords.get('theta')}"

        # Prepare directive section
        directive_section = ""
        if vlm_inputs.directive:
            directive_section = vlm_inputs.directive

        # Prepare primitive guidelines section
        primitive_guidelines = ""
        if (
            vlm_inputs.primitive_in_execution
            and vlm_inputs.primitive_in_execution.guidelines_when_running
        ):
            primitive_guidelines = f"Here are the guidelines for the primitive currently running. Watch them carefully:\n{vlm_inputs.primitive_in_execution.guidelines_when_running}"

        # Prepare available primitives
        primitive_names = [prim.name for prim in vlm_inputs.primitives_list]
        available_primitives = ", ".join(primitive_names)

        return {
            "text_vars": {
                "user_input": user_input,
                "primitive_in_execution": current_primitive,
                "robot_position": robot_coordinates,
                "directive": directive_section,
                "current_primitive_guidelines": primitive_guidelines,
                "available_primitives": available_primitives,
                # Note: multimodal placeholders are handled separately
                "multimodal_history": history_of_events,  # Fallback text version
                "main_camera_image": "[Image will be displayed here]",  # Placeholder text
                "additional_camera_image": "",  # Placeholder text
                "field_of_view": ROBOT_PARAMS_TO_USE["horizontal_fov"],
            },
            "multimodal_history_parts": multimodal_history_parts,
            "main_camera_image_part": main_camera_image_part,
            "additional_camera_parts": additional_camera_parts,
        }

    def _split_template_by_multimodal_placeholders(self, template: str) -> List[dict]:
        """
        Split the template into sections based on multimodal placeholders.

        Args:
            template: The template string

        Returns:
            List of sections with type and content
        """
        sections = []
        current_pos = 0

        # Define multimodal placeholders and their markers
        multimodal_markers = [
            ("{multimodal_history}", "multimodal_history"),
            ("{main_camera_image}", "main_camera_image"),
            ("{additional_camera_image}", "additional_camera_image"),
        ]

        # Find all marker positions
        marker_positions = []
        for marker_text, marker_type in multimodal_markers:
            pos = template.find(marker_text)
            if pos != -1:
                marker_positions.append((pos, pos + len(marker_text), marker_type))

        # Sort by position
        marker_positions.sort(key=lambda x: x[0])

        # Split template into sections
        for start_pos, end_pos, marker_type in marker_positions:
            # Add text before this marker
            if current_pos < start_pos:
                text_content = template[current_pos:start_pos]
                if text_content.strip():
                    sections.append({"type": "text", "content": text_content})

            # Add the multimodal marker section
            sections.append({"type": marker_type, "content": ""})

            current_pos = end_pos

        # Add remaining text
        if current_pos < len(template):
            remaining_text = template[current_pos:]
            if remaining_text.strip():
                sections.append({"type": "text", "content": remaining_text})

        return sections

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

        # SAVE TO FILE
        if SAVE_DEBUG_DATA:
            save_content_parts_html(
                content_parts, "gemini_content_parts", DEBUG_DATA_DIR
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
