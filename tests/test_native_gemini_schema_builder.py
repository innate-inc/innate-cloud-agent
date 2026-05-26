import pytest
from google.genai import _transformers
from pydantic import ValidationError

from src.agents.native_gemini_schema_builder import (
    convert_to_brain_compatible_output,
    create_gemini_schema,
)
from src.agents.types import PrimitiveDefinition


class _GeminiClient:
    vertexai = False


def _assert_google_genai_accepts(model):
    _transformers.process_schema(model.model_json_schema(), _GeminiClient())


def test_legacy_string_input_schema_still_works():
    model = create_gemini_schema(
        [
            PrimitiveDefinition(
                name="arm_utils",
                inputs={"command": "str", "speed": "float"},
                guidelines="Use the arm utility command.",
            )
        ]
    )

    _assert_google_genai_accepts(model)
    output = model.model_validate(
        {
            "stop_current_primitive": False,
            "observation": "ok",
            "thoughts": "call the primitive",
            "next_primitive": {
                "name": "arm_utils",
                "inputs": {"command": "torque_on", "speed": 1.5},
            },
        }
    )

    assert output.next_primitive.inputs.command == "torque_on"
    assert output.next_primitive.inputs.speed == 1.5


def test_structured_enum_input_schema_works_and_validates_values():
    model = create_gemini_schema(
        [
            PrimitiveDefinition(
                name="arm_utils",
                inputs={
                    "command": {
                        "type": "str",
                        "required": True,
                        "enum": ["torque_on", "torque_off", "reboot_arm"],
                    }
                },
                guidelines="Use the arm utility command.",
            )
        ]
    )

    _assert_google_genai_accepts(model)
    output = model.model_validate(
        {
            "stop_current_primitive": False,
            "observation": "ok",
            "thoughts": "call the primitive",
            "next_primitive": {
                "name": "arm_utils",
                "inputs": {"command": "torque_off"},
            },
        }
    )
    assert output.next_primitive.inputs.command == "torque_off"

    with pytest.raises(ValidationError):
        model.model_validate(
            {
                "stop_current_primitive": False,
                "observation": "ok",
                "thoughts": "call the primitive",
                "next_primitive": {
                    "name": "arm_utils",
                    "inputs": {"command": "dance"},
                },
            }
        )


def test_empty_primitive_schema_does_not_create_null_only_schema():
    model = create_gemini_schema([])

    _assert_google_genai_accepts(model)
    output = model.model_validate(
        {
            "stop_current_primitive": False,
            "observation": "ok",
            "thoughts": "no primitive available",
            "next_primitive": {"name": "none", "inputs": {"dummy": ""}},
        }
    )

    assert convert_to_brain_compatible_output(output).next_task is None
