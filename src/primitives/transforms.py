import inspect
from src.primitives.types import Primitive


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
