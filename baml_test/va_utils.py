from baml_py import Image
from baml_client import b
from baml_client.type_builder import TypeBuilder
from dotenv import load_dotenv

load_dotenv()


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

    # Initialize the dynamic type builder.
    tb = TypeBuilder()

    # This list will hold the dynamic types for each NextTask.
    next_task_types = []

    # Process each primitive and create a new dynamic class.
    for i, prim in enumerate(primitives):
        task_name = prim["name"]
        task_desc = prim["description"]

        # Define a dynamic class name (e.g., NextTask1, NextTask2, ...)
        dynamic_class_name = f"NextTask{i+1}"

        # Check that this dynamic class name does not conflict with existing ones.
        # if dynamic_class_name in [cls.name for cls in tb.list_classes()]:
        #     raise ValueError(
        #         f"Dynamic class name conflict: {dynamic_class_name} already exists."
        #     )

        # Create the dynamic composite type for this primitive.
        task_class = tb.add_class(dynamic_class_name)

        # Add a mandatory 'type' field with the guideline embedded in its description.
        # (This field will hold the identifier for the task; its description informs the user.)
        task_class.add_property("type", tb.string()).description(task_desc)

        # Add any input fields for this task.
        for field_name, field_type_str in prim.get("inputs", {}).items():
            # Check for conflicting input field names within this class.
            if field_name in [prop.name for prop in task_class.list_properties()]:
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

            # Add the input field (optional, since it only applies if that task is chosen).
            task_class.add_property(field_name, field_type)

        # Save the newly created type for later union construction.
        next_task_types.append(task_class.type())

    # Create a union type from all dynamic NextTask types.
    # (The API for union types may vary; assume tb.union_type accepts a list of types.)
    union_next_task = tb.union(next_task_types)

    # Set the VisionAgentOutput.next_task field to use the dynamic union.
    # (This replaces the compile-time static union with our runtime union.)
    tb.VisionAgentOutput.add_property("next_task", union_next_task).description(
        "List of the tasks the agent can perform."
    )

    # Now call the VisionAgent function with the injected dynamic types.
    output = await b.VisionAgent(img, last_user_message, baml_options={"tb": tb})
    return output
