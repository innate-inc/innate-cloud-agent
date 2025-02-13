from baml_py import Image
from src.baml_client import b
from src.primitives.transforms import create_type_builder
from src.baml_client.types import VisionAgentOutput


async def vision_agent(
    base64_str: str, last_user_message: str, primitives: list
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

    # Create the TypeBuilder using the split function.
    tb = create_type_builder(primitives)

    # Call the VisionAgent function with the dynamic types.
    output = await b.VisionAgent(img, last_user_message, baml_options={"tb": tb})
    return output
