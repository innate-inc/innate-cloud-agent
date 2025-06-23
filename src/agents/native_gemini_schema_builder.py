from typing import Dict, List, Any, Optional, Union, Literal
from pydantic import BaseModel, Field, create_model
from enum import Enum
from src.agents.types import PrimitiveDefinition


class NextTask(BaseModel):
    """Next task model for Gemini API."""

    name: str = Field(..., description="Name of the task to execute")
    inputs: str = Field(
        default="{}", description="JSON string containing input parameters for the task"
    )


class VisionAgentOutput(BaseModel):
    """
    Pydantic model matching the exact BAML VisionAgentOutput schema.
    """

    stop_current_task: bool = Field(..., description="Whether to stop the current task")
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
    next_task: Optional[NextTask] = Field(
        None, description="The next task to execute, if any"
    )


class BrainCompatibleVisionAgentOutput(BaseModel):
    """
    Output class that's directly compatible with brain.py expectations.
    This prevents type mismatch warnings during serialization.
    """

    stop_current_task: bool
    observation: str
    thoughts: str
    new_goal: Optional[str] = None
    anticipation: Optional[str] = None
    to_tell_user: Optional[str] = None
    next_task: Optional[Dict[str, Any]] = None  # Already in PrimitiveDefinition format


def create_gemini_schema(primitives: List[PrimitiveDefinition]) -> type:
    """
    Create a Gemini-compatible Pydantic model with dynamic next_task Union.
    This mimics BAML's create_type_builder functionality.

    Args:
        primitives: List of available primitives to create specific task models for

    Returns:
        VisionAgentOutput Pydantic model class for Gemini API
    """
    # Create individual task models for each primitive
    task_models = []

    for i, primitive in enumerate(primitives):
        # Create input fields dict for this primitive
        input_fields = {}
        for field_name, field_type_str in primitive.inputs.items():
            # Map string types to Python types
            field_type_str_lower = field_type_str.lower()
            if field_type_str_lower == "string":
                python_type = str
            elif field_type_str_lower in ("float", "double"):
                python_type = float
            elif field_type_str_lower in ("int", "integer"):
                python_type = int
            elif field_type_str_lower in ("bool", "boolean"):
                python_type = bool
            else:
                # Default fallback
                python_type = str

            input_fields[field_name] = (
                python_type,
                Field(..., description=f"{field_name} parameter"),
            )

        # Create inputs model for this primitive
        inputs_model_name = f"Task{i+1}Inputs"
        if input_fields:
            inputs_model = create_model(inputs_model_name, **input_fields)
        else:
            # For primitives with no inputs, create empty model
            inputs_model = create_model(inputs_model_name)

        # Create task model for this primitive
        task_model_name = f"Task{i+1}"
        task_model = create_model(
            task_model_name,
            name=(
                Literal[primitive.name],
                Field(..., description=f"Task: {primitive.guidelines}"),
            ),
            inputs=(
                inputs_model,
                Field(..., description="Input parameters for the task"),
            ),
        )

        task_models.append(task_model)

    # Create Union of all task models
    if task_models:
        NextTaskUnion = Union[tuple(task_models)]
    else:
        # Fallback if no primitives
        NextTaskUnion = None

    # Create the main VisionAgentOutput model
    VisionAgentOutput = create_model(
        "VisionAgentOutput",
        stop_current_task=(
            bool,
            Field(..., description="Whether to stop the current task"),
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
        next_task=(
            Optional[NextTaskUnion],
            Field(None, description="The next task to execute, if any"),
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
        gemini_output: The output from Gemini API with typed Task objects

    Returns:
        BrainCompatibleVisionAgentOutput with proper dictionary next_task
    """
    next_task_dict = None

    if gemini_output.next_task:
        # Extract inputs from the properly typed Pydantic model
        inputs_dict = {}
        if hasattr(gemini_output.next_task, "inputs"):
            # Convert the inputs model to a dictionary
            if hasattr(gemini_output.next_task.inputs, "model_dump"):
                inputs_dict = gemini_output.next_task.inputs.model_dump()
            else:
                # Fallback to converting to dict, filtering private attributes
                inputs_dict = {
                    k: v
                    for k, v in gemini_output.next_task.inputs.__dict__.items()
                    if not k.startswith("_")
                }

        # Create a PrimitiveDefinition-compatible dictionary
        next_task_dict = {
            "name": gemini_output.next_task.name,
            "inputs": inputs_dict,
            "guidelines": None,  # We don't have guidelines in the response
            "primitive_id": None,  # Will be assigned by brain.py
        }

    # Create a new object with the proper type schema
    return BrainCompatibleVisionAgentOutput(
        stop_current_task=gemini_output.stop_current_task,
        observation=gemini_output.observation,
        thoughts=gemini_output.thoughts,
        new_goal=gemini_output.new_goal,
        anticipation=gemini_output.anticipation,
        to_tell_user=gemini_output.to_tell_user,
        next_task=next_task_dict,
    )
