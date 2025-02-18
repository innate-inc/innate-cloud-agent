from typing import Optional
from baml_py import Image
from baml_py.errors import BamlValidationError
from src.baml_client import b
from src.primitives.transforms import create_type_builder
from src.baml_client.types import VisionAgentOutput
import asyncio


async def vision_agent(vlm_inputs: dict) -> Optional[VisionAgentOutput]:
    """
    Calls the VisionAgent function with a dynamically built union for next_task.

    The `primitives` argument should be a list of dictionaries with keys:
      - "name": Unique name for the task (e.g., "ServeGlass")
      - "description": Guideline for the task (e.g., "Serve a glass of water. The glass has to be in sight.")
      - "inputs": A dict mapping input field names to their types as strings (e.g., {"distance": "float"})

    Example:
      primitives = [
          {
              "name": "ServeGlass",
              "description": "Serve a glass of water. The glass has to be in sight.",
              "inputs": {"distance": "float"}
          },
          {
              "name": "GrabBottle",
              "description": "Grab the bottle from the counter.",
              "inputs": {"bottle_color": "string"}
          }
      ]
    """
    img = Image.from_base64("image/png", vlm_inputs["base64_img"])
    tb = create_type_builder(vlm_inputs["primitives_list"])
    primitive_names = [prim["name"] for prim in vlm_inputs["primitives_list"]]
    primitives_list_string = ", ".join(primitive_names)
    max_retries = 3

    context_text_lines = [
        (
            f"The user said: {vlm_inputs['user_prompt_text']}"
            if vlm_inputs["user_prompt_text"] is not None
            else "The user did not say anything."
        ),
        (
            f"The current task is: {vlm_inputs['primitive_in_execution']}"
            if vlm_inputs["primitive_in_execution"] is not None
            else "You are not currently executing a task."
        ),
    ]
    context_text = "\n".join(context_text_lines)

    response = await decreasesmax_retries(
        img,
        context_text,
        primitives_list_string,
        tb,
        max_retries,
    )

    # Some post-processing if necessary

    return response


async def decreasesmax_retries(
    img: Image,
    context_text: Optional[str],
    primitives_list_string: str,
    tb,
    max_retries: int,
    attempt: int = 1,
) -> Optional[VisionAgentOutput]:
    """
    Recursively attempts to call VisionAgent until either a successful output is produced
    or the number of allowed retries (max_retries) is exhausted.

    Args:
        img (Image): The image instance built from the base64 string.
        user_prompt_text (Optional[str]): The user prompt text (or None).
        tb: The type builder used in calling VisionAgent.
        max_retries (int): The maximum number of attempts allowed.
        attempt (int, optional): The current attempt number. Defaults to 1.

    Returns:
        VisionAgentOutput: The output returned by VisionAgent.

    Raises:
        BamlValidationError: When the VisionAgent call fails on the final attempt.
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
        print(f"BamlValidationError on attempt {attempt}/{max_retries}: {e}")
        if attempt == max_retries:
            return None
        await asyncio.sleep(1)
        return await decreasesmax_retries(
            img,
            context_text,
            primitives_list_string,
            tb,
            max_retries,
            attempt + 1,
        )
