from typing import Optional, Literal, Dict, Any
from baml_py import Image
from baml_py.errors import BamlValidationError, BamlClientError
from src.baml_client import b
from src.primitives.transforms import create_type_builder
from src.agents.types import VisionAgentInput
from src.baml_client.types import VisionAgentOutput, NewVisionAgentOutput
from src.agents.exceptions import MaxRetriesExceededException, UnforeseenBamlClientError
import asyncio
from pydantic import BaseModel


FLASH_EXECUTION_TIMEOUT = 3

# Available Gemini agent variants
GeminiAgentVariant = Literal["gemini1", "gemini2", "gemini3", "gemini4"]


async def gemini_vision_agent(
    vlm_inputs: VisionAgentInput,
    agent_variant: GeminiAgentVariant = "gemini1",
) -> Optional[VisionAgentOutput]:
    """
    Calls one of the four GeminiVisionAgent functions with a dynamically built union for next_task.

    Args:
        vlm_inputs: The VisionAgentInput containing all required inputs for the vision agent.
        agent_variant: The variant of Gemini agent to use.
            - "gemini1": Uses GeminiVisionAgent1 (returns VisionAgentOutput)
            - "gemini2": Uses GeminiVisionAgent2 (returns VisionAgentOutput)
            - "gemini3": Uses GeminiVisionAgent3 (returns NewVisionAgentOutput,
               converted to VisionAgentOutput)
            - "gemini4": Uses GeminiVisionAgent4 (returns NewVisionAgentOutput,
               converted to VisionAgentOutput)

    Returns:
        VisionAgentOutput: The standardized output from the selected vision agent.

    Raises:
        MaxRetriesExceededException: When the maximum number of retries is exceeded.
        UnforeseenBamlClientError: When an unexpected BAML client error occurs.
    """
    img = Image.from_base64("image/jpeg", vlm_inputs.base64_img)
    tb = create_type_builder(vlm_inputs.primitives_list)
    primitive_names = [prim.name for prim in vlm_inputs.primitives_list]
    primitives_list_string = ", ".join(primitive_names)
    max_retries = 10

    context_text_lines = [
        (
            f"Below is the history of your actions and exchanges so far:\n"
            f"{vlm_inputs.history_as_string}"
        ),
        (
            f"The user said: {vlm_inputs.user_prompt_text}"
            if vlm_inputs.user_prompt_text is not None
            else "The user did not say anything."
        ),
        (
            f"The current task is: "
            f"{vlm_inputs.primitive_in_execution.model_dump_json()}"
            if vlm_inputs.primitive_in_execution is not None
            else "You are not currently executing a task."
        ),
    ]

    # If robot coordinates were provided, append them to the context.
    if vlm_inputs.robot_coords:
        coords = vlm_inputs.robot_coords
        coords_text = (
            f"Your coordinates if useful to know are: "
            f"x={coords.get('x')}, y={coords.get('y')}, "
            f"z={coords.get('z')}, theta={coords.get('theta')}"
        )
        context_text_lines.append(coords_text)

    # If directive was provided, append it to the context
    if vlm_inputs.directive:
        directive_text = f"Your directive is: {vlm_inputs.directive}"
        context_text_lines.append(directive_text)

    context_text = "\n".join(context_text_lines)

    try:
        response = await decreasesmax_retries(
            img,
            context_text,
            primitives_list_string,
            tb,
            max_retries,
            agent_variant,
        )
        return response
    except MaxRetriesExceededException as e:
        # Re-raise the exception to be handled by the caller
        raise e
    except UnforeseenBamlClientError as e:
        # Re-raise the exception to be handled by the caller
        raise e


async def decreasesmax_retries(
    img: Image,
    context_text: Optional[str],
    primitives_list_string: str,
    tb,
    max_retries: int,
    agent_variant: GeminiAgentVariant = "gemini1",
    attempt: int = 1,
) -> Optional[VisionAgentOutput]:
    """
    Recursively attempts to call the selected GeminiVisionAgent until either a
    successful output is produced or the number of allowed retries (max_retries) is exhausted.

    Args:
        img (Image): The image instance built from the base64 string.
        context_text (Optional[str]): The context text.
        primitives_list_string (str): String representation of available primitives.
        tb: The type builder used in calling GeminiVisionAgent.
        max_retries (int): The maximum number of attempts allowed.
        agent_variant (GeminiAgentVariant): The variant of Gemini agent to use.
        attempt (int, optional): The current attempt number. Defaults to 1.

    Returns:
        VisionAgentOutput: The output returned by GeminiVisionAgent.

    Raises:
        MaxRetriesExceededException: When the maximum number of retries is exceeded.
        UnforeseenBamlClientError: When an unexpected BAML client error occurs.
    """
    try:
        # Set a timeout for the GeminiVisionAgent call
        if agent_variant == "gemini1":
            # Call GeminiVisionAgent1 which returns VisionAgentOutput
            output = await asyncio.wait_for(
                b.GeminiVisionAgent1(
                    img,
                    context_text,
                    primitives_list_string=primitives_list_string,
                    baml_options={"tb": tb},
                ),
                timeout=FLASH_EXECUTION_TIMEOUT,
            )
            return output
        elif agent_variant == "gemini2":
            # Call GeminiVisionAgent2 which returns VisionAgentOutput
            output = await asyncio.wait_for(
                b.GeminiVisionAgent2(
                    img,
                    context_text,
                    primitives_list_string=primitives_list_string,
                    baml_options={"tb": tb},
                ),
                timeout=FLASH_EXECUTION_TIMEOUT,
            )
            return output
        elif agent_variant == "gemini3":
            # Call GeminiVisionAgent3 which returns NewVisionAgentOutput
            new_output = await asyncio.wait_for(
                b.GeminiVisionAgent3(
                    img,
                    context_text,
                    primitives_list_string=primitives_list_string,
                    baml_options={"tb": tb},
                ),
                timeout=FLASH_EXECUTION_TIMEOUT,
            )
            # Convert NewVisionAgentOutput to VisionAgentOutput
            return convert_new_output_to_vision_output(new_output)
        elif agent_variant == "gemini4":
            # Call GeminiVisionAgent4 which returns NewVisionAgentOutput
            new_output = await asyncio.wait_for(
                b.GeminiVisionAgent4(
                    img,
                    context_text,
                    primitives_list_string=primitives_list_string,
                    baml_options={"tb": tb},
                ),
                timeout=FLASH_EXECUTION_TIMEOUT,
            )
            # Convert NewVisionAgentOutput to VisionAgentOutput
            return convert_new_output_to_vision_output(new_output)
        else:
            raise ValueError(f"Unsupported agent variant: {agent_variant}")
    except asyncio.TimeoutError:
        error_msg = (
            f"Operation timed out after {FLASH_EXECUTION_TIMEOUT} seconds "
            f"on attempt {attempt}/{max_retries}"
        )
        print(f"\033[1;31m{error_msg}\033[0m")
        if attempt == max_retries:
            raise MaxRetriesExceededException(
                agent_type=f"gemini_flash_{agent_variant}",
                max_retries=max_retries,
                last_error=TimeoutError(
                    f"GeminiVisionAgent call exceeded {FLASH_EXECUTION_TIMEOUT} second timeout"
                ),
            )
        await asyncio.sleep(1)
        return await decreasesmax_retries(
            img,
            context_text,
            primitives_list_string,
            tb,
            max_retries,
            agent_variant,
            attempt + 1,
        )
    except BamlValidationError as e:
        error_msg = f"BamlValidationError on attempt {attempt}/{max_retries}: {e}"
        print(f"\033[1;31m{error_msg}\033[0m")
        if attempt == max_retries:
            # Raise a custom exception instead of returning None
            raise MaxRetriesExceededException(
                agent_type=f"gemini_flash_{agent_variant}",
                max_retries=max_retries,
                last_error=e,
            )
        await asyncio.sleep(1)
        return await decreasesmax_retries(
            img,
            context_text,
            primitives_list_string,
            tb,
            max_retries,
            agent_variant,
            attempt + 1,
        )
    except BamlClientError as e:
        error_msg = f"BamlClientError on attempt {attempt}/{max_retries}: {e}"
        print(f"\033[1;31m{error_msg}\033[0m")
        if "hyper_util::client::legacy::Error(Connect, TimedOut)" in str(e):
            # For timeout errors, retry
            if attempt < max_retries:
                await asyncio.sleep(1)
                return await decreasesmax_retries(
                    img,
                    context_text,
                    primitives_list_string,
                    tb,
                    max_retries,
                    agent_variant,
                    attempt + 1,
                )
            else:
                # If we've reached max retries, raise the MaxRetriesExceededException
                raise MaxRetriesExceededException(
                    agent_type=f"gemini_flash_{agent_variant}",
                    max_retries=max_retries,
                    last_error=e,
                )
        if "hyper_util::client::legacy::Error(Connect, Ssl(Error" in str(e):
            # For SSL errors, retry
            if attempt < max_retries:
                await asyncio.sleep(1)
                return await decreasesmax_retries(
                    img,
                    context_text,
                    primitives_list_string,
                    tb,
                    max_retries,
                    agent_variant,
                    attempt + 1,
                )
            else:
                # If we've reached max retries, raise the MaxRetriesExceededException
                raise MaxRetriesExceededException(
                    agent_type=f"gemini_flash_{agent_variant}",
                    max_retries=max_retries,
                    last_error=e,
                )
        else:
            # For other client errors, raise a specific exception
            raise UnforeseenBamlClientError(
                f"Unforeseen BamlClientError on attempt {attempt}/{max_retries}: {e}",
                original_error=e,
            )


def convert_new_output_to_vision_output(
    new_output: Dict[str, Any],
) -> VisionAgentOutput:
    """
    Converts a NewVisionAgentOutput to VisionAgentOutput.

    Args:
        new_output: The NewVisionAgentOutput to convert.

    Returns:
        VisionAgentOutput: The converted output.
    """
    # Extract next_task from new_output if it exists
    next_task = new_output.get("next_task")

    # Map action_decision to stop_current_task
    action_decision = new_output.get("action_decision", "continue")
    stop_current_task = action_decision in ["stop_task", "change_task"]

    # Create VisionAgentOutput
    return VisionAgentOutput(
        stop_current_task=stop_current_task,
        observation=new_output.get("current_observation", ""),
        thoughts=new_output.get("current_thoughts", ""),
        new_goal=None,  # No direct mapping for new_goal
        next_task=next_task,
        anticipation=None,  # No direct mapping for anticipation
        to_tell_user=new_output.get("to_tell_user"),
    )
