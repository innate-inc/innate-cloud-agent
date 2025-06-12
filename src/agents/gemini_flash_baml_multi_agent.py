from typing import Optional, List
import base64
import json
from pathlib import Path
from datetime import datetime
from baml_py import Image
from baml_py.errors import BamlValidationError, BamlClientError
from src.baml_client import b
from src.primitives.transforms import create_type_builder
from src.agents.types import MultimodalVisionAgentInput
from src.baml_client.types import NewVisionAgentOutput
from src.agents.exceptions import MaxRetriesExceededException, UnforeseenBamlClientError
import asyncio

# Flag to control saving of debug data
SAVE_DEBUG_DATA = True  # Set to False to disable
DEBUG_DATA_DIR = Path("test_data/debug_agent_images")

FLASH_EXECUTION_TIMEOUT = 5

# Available Gemini agent variants
# We will only use one variant for the multi-image agent initially
# GeminiAgentVariant = Literal["gemini1", "gemini2", "gemini3", "gemini4"]


def _save_base64_image(base64_img: str, filename: str) -> str:
    """
    Save a base64 encoded image to a file.

    Args:
        base64_img: Base64 encoded image string
        filename: Name for the saved file (without extension)

    Returns:
        Path to the saved file
    """
    try:
        # Create debug images directory if it doesn't exist
        DEBUG_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
        if base64_img.startswith("data:image"):
            base64_img = base64_img.split(",")[1]

        # Decode base64 data
        image_data = base64.b64decode(base64_img)

        # Try to determine file extension from image header
        if image_data.startswith(b"\xff\xd8\xff"):
            ext = ".jpg"
        elif image_data.startswith(b"\x89PNG"):
            ext = ".png"
        elif image_data.startswith(b"GIF"):
            ext = ".gif"
        elif image_data.startswith(b"RIFF"):
            ext = ".webp"
        else:
            ext = ".jpg"  # Default to jpg

        # Create full file path
        file_path = (
            DEBUG_DATA_DIR
            / f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        )

        # Write image data to file
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

    Args:
        context_data: Dictionary containing context information
        filename: Name for the saved file (without extension)

    Returns:
        Path to the saved file
    """
    try:
        # Create debug data directory if it doesn't exist
        DEBUG_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Create full file path
        file_path = (
            DEBUG_DATA_DIR
            / f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )

        # Write context data to file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("=== GEMINI AGENT CONTEXT DATA ===\n\n")
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


async def gemini_vision_agent_multimodal_history(
    vlm_inputs: MultimodalVisionAgentInput,  # Updated type hint
    # agent_variant: GeminiAgentVariant = "gemini1", # Remove variant for now
) -> Optional[NewVisionAgentOutput]:  # Output type will be NewVisionAgentOutput
    """
    Calls the GeminiVisionAgentMultiImages function with a dynamically built union
    for next_task.

    Args:
        vlm_inputs: The VisionAgentInput containing all required inputs for the
                    vision agent, including multimodal history.

    Returns:
        NewVisionAgentOutput: The standardized output from the vision agent.

    Raises:
        MaxRetriesExceededException: When the maximum number of retries is exceeded.
        UnforeseenBamlClientError: When an unexpected BAML client error occurs.
    """
    # Save debug data if enabled
    if SAVE_DEBUG_DATA:
        # Save main image
        if vlm_inputs.base64_img:
            _save_base64_image(vlm_inputs.base64_img, "main_image")

        # Save additional images if present (before transformation)
        if vlm_inputs.additional_image_data:
            camera_type = vlm_inputs.additional_image_data.get("camera_type", "unknown")
            image_b64 = vlm_inputs.additional_image_data.get("image_b64", "")
            if image_b64:
                _save_base64_image(image_b64, f"additional_{camera_type}")

    img = Image.from_base64("image/jpeg", vlm_inputs.base64_img)
    tb = create_type_builder(vlm_inputs.primitives_list)
    primitive_names = [prim.name for prim in vlm_inputs.primitives_list]
    primitives_list_string = ", ".join(primitive_names)
    max_retries = 10

    # Construct multimodal context
    # This assumes vlm_inputs.multimodal_history is a List[Image | str]
    # and needs to be correctly populated by the caller.
    # Make a copy if multimodal_history is mutable and we intend to append to it
    context_multimodal = [
        (
            Image.from_base64("image/jpeg", item.content)
            if item.type == "image"
            else item.content
        )
        for item in vlm_inputs.multimodal_history
    ]

    # Append other context elements as text to the multimodal list
    # We might want to make these multimodal in the future

    # Construct a list of text parts to be joined and then appended
    additional_text_context_parts = []

    if vlm_inputs.user_prompt_text:
        additional_text_context_parts.append(
            f"The user said: {vlm_inputs.user_prompt_text}"
        )
    else:
        additional_text_context_parts.append("The user did not say anything.")

    if vlm_inputs.primitive_in_execution:
        additional_text_context_parts.append(
            f"The current task is: "
            f"{vlm_inputs.primitive_in_execution.model_dump_json()}"
        )
    else:
        additional_text_context_parts.append("You are not currently executing a task.")

    if vlm_inputs.robot_coords:
        coords = vlm_inputs.robot_coords
        additional_text_context_parts.append(
            f"Your coordinates if useful to know are: "
            f"x={coords.get('x')}, y={coords.get('y')}, "
            f"z={coords.get('z')}, theta={coords.get('theta')}"
        )

    if vlm_inputs.directive:
        additional_text_context_parts.append(
            f"Your directive is: {vlm_inputs.directive}"
        )

    additional_image_data = []

    if vlm_inputs.additional_image_data:
        additional_image_data.append(
            f"On top of that, this is what you see from the additional camera with type: {vlm_inputs.additional_image_data['camera_type']}"
        )
        additional_image_data.append(
            Image.from_base64(
                "image/jpeg", vlm_inputs.additional_image_data["image_b64"]
            )
        )

    # Join all additional text parts into a single string and append to multimodal context
    if additional_text_context_parts:
        context_multimodal.append("\n".join(additional_text_context_parts))

    running_primitive_guidelines = (
        vlm_inputs.primitive_in_execution.guidelines_when_running
        if vlm_inputs.primitive_in_execution
        else ""
    )

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
            "running_primitive_guidelines": running_primitive_guidelines,
            "primitives_list_string": primitives_list_string,
            "additional_image_data_original": vlm_inputs.additional_image_data,
            "context_multimodal_summary": [
                (
                    f"Image {i}"
                    if isinstance(item, Image)
                    else str(item)[:200] + "..." if len(str(item)) > 200 else str(item)
                )
                for i, item in enumerate(context_multimodal)
            ],
            "max_retries": max_retries,
        }
        _save_context_data(context_data, "context_data")

    try:
        response = await decreasesmax_retries_multi_context(
            img=img,
            context_multimodal=context_multimodal,
            running_primitive_guidelines=running_primitive_guidelines,
            primitives_list_string=primitives_list_string,
            tb=tb,
            max_retries=max_retries,
            additional_image_data=additional_image_data,
        )
        return response
    except MaxRetriesExceededException as e:
        raise e
    except UnforeseenBamlClientError as e:
        raise e


async def decreasesmax_retries_multi_context(
    img: Image,
    context_multimodal: List[Image | str],  # Updated context type
    running_primitive_guidelines: str,
    primitives_list_string: str,
    tb,
    max_retries: int,
    additional_image_data: dict = {},
    attempt: int = 1,
) -> Optional[NewVisionAgentOutput]:  # Output type is NewVisionAgentOutput
    """
    Recursively attempts to call GeminiVisionAgentMultiImages until a
    successful output is produced or max_retries is exhausted.
    """

    async def recall():
        return await decreasesmax_retries_multi_context(
            img=img,
            context_multimodal=context_multimodal,
            running_primitive_guidelines=running_primitive_guidelines,
            primitives_list_string=primitives_list_string,
            tb=tb,
            max_retries=max_retries,
            additional_image_data=additional_image_data,
            attempt=attempt + 1,
        )

    try:
        new_output = await asyncio.wait_for(
            b.GeminiVisionAgentMultiImages(
                img=img,
                additional_image_data=additional_image_data,
                running_primitive_guidelines=running_primitive_guidelines,
                context_multimodal=context_multimodal,
                primitives_list_string=primitives_list_string,
                baml_options={"tb": tb},
            ),
            timeout=FLASH_EXECUTION_TIMEOUT,
        )
        return new_output  # Directly return NewVisionAgentOutput

    except asyncio.TimeoutError:
        error_msg = (
            f"Operation timed out after {FLASH_EXECUTION_TIMEOUT} seconds "
            f"on attempt {attempt}/{max_retries}"
        )
        print(f"\033[1;31m{error_msg}\033[0m")
        if attempt == max_retries:
            raise MaxRetriesExceededException(
                agent_type="gemini_flash_multi_context",  # Updated agent type
                max_retries=max_retries,
                last_error=TimeoutError(
                    f"GeminiVisionAgentMultiImages call exceeded "
                    f"{FLASH_EXECUTION_TIMEOUT} second timeout"
                ),
            )
        await asyncio.sleep(1)
        return await recall()
    except BamlValidationError as e:
        error_msg = f"BamlValidationError on attempt {attempt}/{max_retries}: {e}"
        print(f"\033[1;31m{error_msg}\033[0m")
        if attempt == max_retries:
            raise MaxRetriesExceededException(
                agent_type="gemini_flash_multi_context",  # Updated agent type
                max_retries=max_retries,
                last_error=e,
            )
        await asyncio.sleep(1)
        return await recall()
    except BamlClientError as e:

        def error_msg_func(e_val):
            return (
                f"\033[1;31mBamlClientError on attempt {attempt}/{max_retries}: "
                f"{e_val}\033[0m"
            )

        if "hyper_util::client::legacy::Error(Connect, TimedOut)" in str(e):
            print(error_msg_func("Timeout"))
            if attempt < max_retries:
                await asyncio.sleep(1)
                return await recall()
            else:
                raise MaxRetriesExceededException(
                    agent_type="gemini_flash_multi_context",
                    max_retries=max_retries,
                    last_error=e,
                )
        if "hyper_util::client::legacy::Error(Connect, Ssl(Error" in str(e):
            print(error_msg_func("SSL Error"))
            if attempt < max_retries:
                await asyncio.sleep(1)
                return await recall()
            else:
                raise MaxRetriesExceededException(
                    agent_type="gemini_flash_multi_context",
                    max_retries=max_retries,
                    last_error=e,
                )
        if "503" in str(e):
            print(error_msg_func("503 Error"))
            if attempt < max_retries:
                await asyncio.sleep(1)
                return await recall()
            else:
                raise MaxRetriesExceededException(
                    agent_type="gemini_flash_multi_context",
                    max_retries=max_retries,
                    last_error=e,
                )
        else:
            print(error_msg_func(str(e)))
            raise UnforeseenBamlClientError(
                f"Unforeseen BamlClientError on attempt {attempt}/{max_retries}: {e}",
                original_error=e,
            )


# The conversion function might not be needed if the new agent directly returns NewVisionAgentOutput
# and the calling code expects that. If VisionAgentOutput is still needed elsewhere,
# this function (or a modified version) will be necessary.
# For now, I'll comment it out as the new agent returns NewVisionAgentOutput.

# def convert_new_output_to_vision_output(
#     new_output: NewVisionAgentOutput,
# ) -> VisionAgentOutput:
#     """
#     Converts a NewVisionAgentOutput to VisionAgentOutput.
#     """
#     next_task = new_output.next_task
#     action_decision = new_output.action_decision
#     stop_current_task = action_decision in ["stop_task", "change_task"]
#     return VisionAgentOutput(
#         stop_current_task=stop_current_task,
#         observation=new_output.current_observation,
#         thoughts=new_output.current_thoughts,
#         new_goal=None,
#         next_task=next_task,
#         anticipation=None,
#         to_tell_user=new_output.to_tell_user,
#     )
