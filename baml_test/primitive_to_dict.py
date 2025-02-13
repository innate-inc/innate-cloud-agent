# %%
from abc import ABC, abstractmethod


class Primitive(ABC):
    @abstractmethod
    async def execute(self):
        """
        Execute the primitive.

        Subclasses must implement this method.
        """
        pass

    def guidelines(self):
        """
        Optionally provide guidelines for this primitive.
        Subclasses may override this method if guidelines are available.
        """
        return None


# Example primitive without guidelines.
class ServeGlass(Primitive):
    def __init__(self):
        super().__init__()

    def guidelines(self):
        return "Serve a glass of water. The glass has to be in sight."

    async def execute(self, distance: float):
        # Gentle handover motion for glass.
        print("Serving glass.")
        return "Served glass.", True


# %%
import inspect


def primitive_to_dict(primitive_obj):
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


# %%
