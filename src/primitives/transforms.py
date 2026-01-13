import inspect
import uuid
from src.agents.types import PrimitiveDefinition
from src.primitives.types import Primitive


def primitive_to_object(primitive_obj: Primitive) -> PrimitiveDefinition:
    """
    Given a Primitive instance, returns a dict with:
      - "name": the name of the primitive
      - "guidelines": the guidelines text (if a guidelines() method exists)
      - "inputs": a dict mapping each parameter (excluding 'self') of execute to its type name
      - "primitive_id": a unique identifier for this primitive instance
    """
    result = {}

    # Add the 'name' attribute from the primitive.
    result["name"] = primitive_obj.name

    # If the object has a guidelines method, call it and store its result.
    guidelines_func = getattr(primitive_obj, "guidelines", None)
    if callable(guidelines_func):
        result["guidelines"] = guidelines_func()

    guidelines_when_running_func = getattr(
        primitive_obj, "guidelines_when_running", None
    )
    if callable(guidelines_when_running_func):
        result["guidelines_when_running"] = guidelines_when_running_func()

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

    # Generate a unique primitive_id if one doesn't exist
    if hasattr(primitive_obj, "primitive_id") and primitive_obj.primitive_id:
        result["primitive_id"] = primitive_obj.primitive_id
    else:
        result["primitive_id"] = str(uuid.uuid4())

    return PrimitiveDefinition.model_validate(result)
