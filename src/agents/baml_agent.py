from typing import Optional
from baml_py import Image
from baml_py.errors import BamlValidationError
from src.baml_client import b
from src.primitives.transforms import create_type_builder
from src.baml_client.types import VisionAgentOutput
import asyncio


async def vision_agent(
    base64_str: str, user_prompt_text: Optional[str], primitives: list
) -> VisionAgentOutput:
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
    img = Image.from_base64("image/png", base64_str)
    tb = create_type_builder(primitives)
    max_retries = 3
    return await decreasesmax_retries(img, user_prompt_text, tb, max_retries)


async def decreasesmax_retries(
    img: Image,
    user_prompt_text: Optional[str],
    tb,
    max_retries: int,
    attempt: int = 1,
) -> VisionAgentOutput:
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
            user_prompt_text if user_prompt_text is not None else "NO MESSAGE",
            baml_options={"tb": tb},
        )
        return output
    except BamlValidationError as e:
        print(f"BamlValidationError on attempt {attempt}/{max_retries}: {e}")
        if attempt == max_retries:
            raise
        await asyncio.sleep(1)
        return await decreasesmax_retries(
            img, user_prompt_text, tb, max_retries, attempt + 1
        )
