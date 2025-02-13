import inspect
from src.primitives.types import Primitive

from src.baml_client.type_builder import TypeBuilder


def primitive_to_dict(primitive_obj: Primitive):
    """
    Given a Primitive instance, returns a dict with:
      - "guideline": the guidelines text (if a guidelines() method exists)
      - "inputs": a list of tuples (input_name, type_name) for each parameter of execute (other than self)
    """
    result = {}

    # If the object has a guidelines method, call it and store its result.
    guidelines_func = getattr(primitive_obj, "guidelines", None)
    if callable(guidelines_func):
        result["guideline"] = guidelines_func()

    # Use inspect.signature to get the parameters of the execute method.
    execute_func = getattr(primitive_obj, "execute", None)
    inputs = {}
    if callable(execute_func):
        sig = inspect.signature(execute_func)
        # Loop over parameters, skipping "self"
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            # Determine the type annotation, if any.
            if param.annotation is inspect.Parameter.empty:
                type_name = None
            else:
                # If the annotation is a type, get its __name__
                if isinstance(param.annotation, type):
                    type_name = param.annotation.__name__
                else:
                    # Otherwise, use the string representation.
                    type_name = str(param.annotation)
            inputs[name] = type_name
    result["inputs"] = inputs

    return result


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
