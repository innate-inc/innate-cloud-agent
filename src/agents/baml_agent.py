from typing import Optional
from baml_py import Image
from baml_py.errors import BamlValidationError
from src.baml_client import b
from src.primitives.transforms import create_type_builder
from src.agents.types import VisionAgentInput
from src.baml_client.types import VisionAgentOutput
from src.agents.exceptions import MaxRetriesExceededException
import asyncio


async def vision_agent(vlm_inputs: VisionAgentInput) -> Optional[VisionAgentOutput]:
    """
    Calls the VisionAgent function with a dynamically built union for next_task.

    The VisionAgentInput model ensures the following keys are available:
      - base64_img: A base64-encoded image.
      - user_prompt_text: (Optional) The text provided by the user.
      - primitive_in_execution: (Optional) The current primitive in execution.
      - primitives_list: A list of primitives with fields:
            - name: Unique name of the task.
            - description: (Aliased to 'guidelines' internally) Guidelines for the task.
            - inputs: A dictionary mapping input fields to their type strings.
      - history_as_string: A history of events.
      - robot_coords: (Optional) A dictionary with the robot's coordinates.
      - directive: (Optional) A directive to steer the vision language model.
    """
    img = Image.from_base64("image/jpeg", vlm_inputs.base64_img)
    tb = create_type_builder(vlm_inputs.primitives_list)
    primitive_names = [prim.name for prim in vlm_inputs.primitives_list]
    primitives_list_string = ", ".join(primitive_names)
    max_retries = 3

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
        )
        return response
    except MaxRetriesExceededException as e:
        # Re-raise the exception to be handled by the caller
        raise e


async def decreasesmax_retries(
    img: Image,
    context_text: Optional[str],
    primitives_list_string: str,
    tb,
    max_retries: int,
    attempt: int = 1,
) -> Optional[VisionAgentOutput]:
    """
    Recursively attempts to call VisionAgent until either a successful output
    is produced or the number of allowed retries (max_retries) is exhausted.

    Args:
        img (Image): The image instance built from the base64 string.
        context_text (Optional[str]): The context text.
        primitives_list_string (str): String representation of available primitives.
        tb: The type builder used in calling VisionAgent.
        max_retries (int): The maximum number of attempts allowed.
        attempt (int, optional): The current attempt number. Defaults to 1.

    Returns:
        VisionAgentOutput: The output returned by VisionAgent.

    Raises:
        MaxRetriesExceededException: When the maximum number of retries is exceeded.
    """
    try:
        output = await b.VisionAgent(
            img,
            context_text,
            primitives_list_string=primitives_list_string,
            baml_options={"tb": tb},
        )
        return output
    except BamlValidationError as e:
        error_fmt = "\033[1;31mBamlValidationError on attempt {}/{}: {}\033[0m"
        error_msg = error_fmt.format(attempt, max_retries, e)
        print(error_msg)
        if attempt == max_retries:
            # Raise a custom exception instead of returning None
            raise MaxRetriesExceededException(
                agent_type="anthropic", max_retries=max_retries, last_error=e
            )
        await asyncio.sleep(1)
        return await decreasesmax_retries(
            img,
            context_text,
            primitives_list_string,
            tb,
            max_retries,
            attempt + 1,
        )
