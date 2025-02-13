from baml_py import Image
from baml_client import b
from baml_client.type_builder import TypeBuilder
from dotenv import load_dotenv

load_dotenv()


def create_type_builder(primitives: list) -> TypeBuilder:
    """
    Creates a TypeBuilder instance and dynamically builds classes based on the given primitives.

    Each primitive should be a dictionary with keys:
      - "name": Unique name for the task (e.g., "ServeGlass")
      - "description": Guideline for the task (e.g., "Serve a glass of water. The glass has to be in sight.")
      - "inputs": A dict mapping input field names to their types as strings (e.g., {"distance": "float"})
    """
    tb = TypeBuilder()
    next_task_types = []

    # Process each primitive and create a new dynamic class.
    for i, prim in enumerate(primitives):
        task_name = prim["name"]
        task_desc = prim["description"]
        dynamic_class_name = f"NextTask{i+1}"

        # Create the dynamic composite type for this primitive.
        task_class = tb.add_class(dynamic_class_name)

        # Add a mandatory 'type' field with the guideline embedded in its description.
        task_class.add_property("type", tb.string()).description(task_desc)

        # Create a dynamic composite type for the inputs.
        inputs_class_name = f"{dynamic_class_name}_Inputs"
        inputs_class = tb.add_class(inputs_class_name)

        # Add input fields for this task.
        for field_name, field_type_str in prim.get("inputs", {}).items():
            # Check for conflicting input field names within this inputs class.
            if field_name in [prop.name for prop in inputs_class.list_properties()]:
                raise ValueError(
                    f"Duplicate input field '{field_name}' in task '{task_name}'."
                )

            # Map the string type to a BAML type.
            field_type_str_lower = field_type_str.lower()
            if field_type_str_lower == "string":
                field_type = tb.string()
            elif field_type_str_lower in ("float", "double"):
                field_type = tb.float()
            elif field_type_str_lower in ("int", "integer"):
                field_type = tb.int()
            elif field_type_str_lower in ("bool", "boolean"):
                field_type = tb.bool()
            else:
                # Default fallback if unknown.
                field_type = tb.string()

            inputs_class.add_property(field_name, field_type)

        # Link the inputs composite type to the task type.
        task_class.add_property("inputs", inputs_class.type())

        # Keep track of the dynamic task type.
        next_task_types.append(task_class.type())

    # Create a union type from all dynamic NextTask types.
    union_next_task = tb.union(next_task_types)

    # Update the VisionAgentOutput with the dynamic union.
    tb.VisionAgentOutput.add_property("next_task", union_next_task).description(
        "List of the tasks the agent can perform."
    )
    return tb


async def vision_agent(base64_str: str, last_user_message: str, primitives: list):
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
