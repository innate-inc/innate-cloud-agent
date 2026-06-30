# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Innate Inc

from typing import Dict, List, Any, Optional, Union, Literal
from pydantic import BaseModel, Field, create_model
from src.agents.types import PrimitiveDefinition


class NextPrimitive(BaseModel):
    """Next primitive model for Gemini API."""

    name: str = Field(..., description="Name of the primitive to execute")
    inputs: str = Field(
        default="{}",
        description="JSON string containing input parameters for the primitive",
    )


class VisionAgentOutput(BaseModel):
    """
    Pydantic model matching the exact BAML VisionAgentOutput schema.
    """

    stop_current_primitive: bool = Field(
        ..., description="Whether to stop the current primitive"
    )
    observation: str = Field(
        ..., description="What the robot observes in the current situation"
    )
    thoughts: str = Field(
        ..., description="The robot's internal reasoning about what to do"
    )
    new_goal: Optional[str] = Field(None, description="New goal if any")
    anticipation: Optional[str] = Field(None, description="What might happen next")
    to_tell_user: Optional[str] = Field(
        None, description="Message to tell the user (optional)"
    )
    next_primitive: Optional[NextPrimitive] = Field(
        None, description="The next primitive to execute, if any"
    )
    input_tokens: Optional[int] = Field(
        None, description="Number of input tokens used in this call"
    )
    output_tokens: Optional[int] = Field(
        None, description="Number of output tokens used in this call"
    )


class BrainCompatibleVisionAgentOutput(BaseModel):
    """
    Output class that's directly compatible with brain.py expectations.
    This prevents type mismatch warnings during serialization.
    Uses legacy field names to maintain compatibility.
    """

    stop_current_task: bool
    observation: str
    thoughts: str
    new_goal: Optional[str] = None
    anticipation: Optional[str] = None
    to_tell_user: Optional[str] = None
    next_task: Optional[Dict[str, Any]] = None  # Already in PrimitiveDefinition format
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


def _python_type_from_name(type_name: Any) -> type:
    if not isinstance(type_name, str):
        return str

    type_name_lower = type_name.lower()
    if type_name_lower in ("str", "string"):
        return str
    if type_name_lower in ("float", "double", "number"):
        return float
    if type_name_lower in ("int", "integer"):
        return int
    if type_name_lower in ("bool", "boolean"):
        return bool
    return str


def _literal_type_from_enum(enum_values: Any, fallback_type: type) -> Any:
    if not isinstance(enum_values, list) or not enum_values:
        return fallback_type

    literal_values = []
    for value in enum_values:
        if value is None:
            continue
        if fallback_type is str:
            literal_values.append(str(value))
        elif fallback_type is bool and isinstance(value, bool):
            literal_values.append(value)
        elif (
            fallback_type is int
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            literal_values.append(value)
        elif (
            fallback_type is float
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            literal_values.append(float(value))

    if not literal_values:
        return fallback_type

    return Literal[tuple(literal_values)]


def _input_field_from_schema(field_name: str, field_schema: Any) -> tuple[Any, Field]:
    """Normalize legacy string schemas and newer structured skill schemas."""
    type_name = field_schema
    enum_values = None
    required = True
    default = ...
    description = f"{field_name} parameter"

    if isinstance(field_schema, dict):
        type_name = field_schema.get("type", "str")
        enum_values = field_schema.get("enum")
        required = field_schema.get("required", True)
        description = field_schema.get("description") or description
        if "default" in field_schema:
            default = field_schema["default"]

    python_type = _python_type_from_name(type_name)
    python_type = _literal_type_from_enum(enum_values, python_type)

    if not required:
        if default is ...:
            default = None
        python_type = Optional[python_type]

    return python_type, Field(default, description=description)


def create_gemini_schema(primitives: List[PrimitiveDefinition]) -> type:
    """
    Create a Gemini-compatible Pydantic model with dynamic next_primitive Union.
    This mimics BAML's create_type_builder functionality.

    Args:
        primitives: List of available primitives to create specific primitive models for

    Returns:
        VisionAgentOutput Pydantic model class for Gemini API
    """
    # Create individual primitive models for each primitive
    primitive_models = []

    for i, primitive in enumerate(primitives):
        # Create input fields dict for this primitive
        input_fields = {}
        for field_name, field_schema in primitive.inputs.items():
            input_fields[field_name] = _input_field_from_schema(
                field_name, field_schema
            )

        # Create inputs model for this primitive
        inputs_model_name = f"Primitive{i+1}Inputs"
        if input_fields:
            inputs_model = create_model(inputs_model_name, **input_fields)
        else:
            # For primitives with no inputs, create model with dummy field to avoid empty object schema
            inputs_model = create_model(
                inputs_model_name,
                dummy=(
                    str,
                    Field("", description="No inputs required for this primitive"),
                ),
            )

        # Create primitive model for this primitive
        primitive_model_name = primitive.name
        primitive_model = create_model(
            primitive_model_name,
            name=(
                Literal[primitive.name],
                Field(
                    ...,
                    description=f"Guidelines on primitive usage: {primitive.guidelines}",
                ),
            ),
            inputs=(
                inputs_model,
                Field(..., description="Input parameters for the primitive"),
            ),
        )

        primitive_models.append(primitive_model)

    # Create Union of all primitive models
    if primitive_models:
        NextPrimitiveUnion = Union[tuple(primitive_models)]
    else:
        # Fallback if no primitives are currently registered. A concrete
        # sentinel model avoids null-only schemas that google-genai rejects.
        NoPrimitiveInputs = create_model(
            "NoPrimitiveInputs",
            dummy=(
                str,
                Field("", description="No inputs available"),
            ),
        )
        NextPrimitiveUnion = create_model(
            "NoPrimitive",
            name=(
                Literal["none"],
                Field(..., description="No primitive is currently available"),
            ),
            inputs=(
                NoPrimitiveInputs,
                Field(
                    default_factory=NoPrimitiveInputs,
                    description="No inputs available",
                ),
            ),
        )

    # Create the main VisionAgentOutput model
    VisionAgentOutput = create_model(
        "VisionAgentOutput",
        stop_current_primitive=(
            bool,
            Field(..., description="Whether to stop the current primitive"),
        ),
        observation=(
            str,
            Field(..., description="What the robot observes in the current situation"),
        ),
        thoughts=(
            str,
            Field(..., description="The robot's internal reasoning about what to do"),
        ),
        new_goal=(Optional[str], Field(None, description="New goal if any")),
        anticipation=(Optional[str], Field(None, description="What might happen next")),
        to_tell_user=(
            Optional[str],
            Field(None, description="Message to tell the user (optional)"),
        ),
        next_primitive=(
            Optional[NextPrimitiveUnion],
            Field(None, description="The next primitive to execute, if any"),
        ),
        input_tokens=(
            Optional[int],
            Field(None, description="Number of input tokens used in this call"),
        ),
        output_tokens=(
            Optional[int],
            Field(None, description="Number of output tokens used in this call"),
        ),
    )

    return VisionAgentOutput


def create_response_model(primitives: List[PrimitiveDefinition]) -> type:
    """
    Create a dynamic Pydantic response model.

    Args:
        primitives: List of available primitives

    Returns:
        VisionAgentOutput Pydantic model class
    """
    return create_gemini_schema(primitives)


def convert_to_brain_compatible_output(
    gemini_output,
) -> BrainCompatibleVisionAgentOutput:
    """
    Convert Gemini API output to a format that's directly compatible with brain.py
    to prevent type mismatch warnings during serialization.

    Args:
        gemini_output: The output from Gemini API with typed Primitive objects

    Returns:
        BrainCompatibleVisionAgentOutput with proper dictionary next_primitive
    """
    next_primitive_dict = None

    if gemini_output.next_primitive and gemini_output.next_primitive.name != "none":
        # Extract inputs from the properly typed Pydantic model
        inputs_dict = {}
        if hasattr(gemini_output.next_primitive, "inputs"):
            # Convert the inputs model to a dictionary
            if hasattr(gemini_output.next_primitive.inputs, "model_dump"):
                inputs_dict = gemini_output.next_primitive.inputs.model_dump()
            else:
                # Fallback to converting to dict, filtering private attributes
                inputs_dict = {
                    k: v
                    for k, v in gemini_output.next_primitive.inputs.__dict__.items()
                    if not k.startswith("_")
                }

            # Filter out dummy field used for empty input primitives
            inputs_dict = {k: v for k, v in inputs_dict.items() if k != "dummy"}

        # Create a PrimitiveDefinition-compatible dictionary
        next_primitive_dict = {
            "name": gemini_output.next_primitive.name,
            "inputs": inputs_dict,
            "guidelines": None,  # We don't have guidelines in the response
            "primitive_id": None,  # Will be assigned by brain.py
        }

    # Create a new object with the proper type schema
    return BrainCompatibleVisionAgentOutput(
        stop_current_task=gemini_output.stop_current_primitive,
        observation=gemini_output.observation,
        thoughts=gemini_output.thoughts,
        new_goal=gemini_output.new_goal,
        anticipation=gemini_output.anticipation,
        to_tell_user=gemini_output.to_tell_user,
        next_task=next_primitive_dict,
        input_tokens=getattr(gemini_output, 'input_tokens', None),
        output_tokens=getattr(gemini_output, 'output_tokens', None),
    )
