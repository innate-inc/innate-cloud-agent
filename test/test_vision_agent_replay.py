import asyncio
import json
from pathlib import Path
import traceback
from typing import Any, Tuple, List, Union

from src.agents.gemini_flash_baml_multi_agent import (
    gemini_vision_agent_multimodal_history,
)
from src.agents.types import MultimodalVisionAgentInput
from src.baml_client.types import VisionAgentOutput

# Path to the golden set of VLM inputs and expected outputs
GOLDEN_VLM_IO_FILE = (
    Path(__file__).parent.parent / "test_data/golden_vlm_assertions.jsonl"
)


def summarize_vlm_input(vlm_input: MultimodalVisionAgentInput) -> str:
    summary_parts = []
    try:
        if vlm_input.base64_img:
            summary_parts.append("Image: Present")
        else:
            summary_parts.append("Image: Absent")

        if vlm_input.user_prompt_text:
            summary_parts.append(
                f"User Prompt: '{vlm_input.user_prompt_text[:50]}{'...' if len(vlm_input.user_prompt_text) > 50 else ''}'"
            )
        else:
            summary_parts.append("User Prompt: Absent")

        current_primitive_name = "None"
        if vlm_input.primitive_in_execution:
            # Assuming primitive_in_execution is a Pydantic model with a 'name' attribute
            # or a dict with a 'name' key.
            if hasattr(vlm_input.primitive_in_execution, "name"):
                current_primitive_name = vlm_input.primitive_in_execution.name
            elif (
                isinstance(vlm_input.primitive_in_execution, dict)
                and "name" in vlm_input.primitive_in_execution
            ):
                current_primitive_name = vlm_input.primitive_in_execution["name"]
            else:
                current_primitive_name = "Unknown Structure"
        summary_parts.append(f"Primitive in Execution: {current_primitive_name}")

        summary_parts.append(
            f"Primitives List Count: {len(vlm_input.primitives_list) if vlm_input.primitives_list else 0}"
        )
        summary_parts.append(
            f"Multimodal History Length: {len(vlm_input.multimodal_history) if vlm_input.multimodal_history else 0}"
        )

        if vlm_input.directive:
            summary_parts.append(
                f"Directive: '{vlm_input.directive[:50]}{'...' if len(vlm_input.directive) > 50 else ''}'"
            )
    except AttributeError as e:
        summary_parts.append(f"(Error summarizing input: {e})")
    except Exception as e:
        summary_parts.append(f"(Unexpected error summarizing input: {e})")
    return ", ".join(summary_parts)


def compare_task_objects(
    golden_task: Any, actual_task: Any, tolerance: float
) -> Tuple[bool, str]:
    """
    Compares two task objects, typically Pydantic models.
    """
    if type(golden_task) != type(actual_task):
        golden_type = type(golden_task).__name__ if golden_task is not None else "None"
        actual_type = type(actual_task).__name__ if actual_task is not None else "None"
        return (
            False,
            f"next_task type mismatch: golden is {golden_type}, actual is {actual_type}.",
        )

    if golden_task is None:  # implies actual_task is also None due to type check above
        return True, ""

    # Assuming tasks are Pydantic models or objects with model_dump() / __dict__
    try:
        golden_dict = (
            golden_task.model_dump()
            if hasattr(golden_task, "model_dump")
            else golden_task.__dict__
        )
        actual_dict = (
            actual_task.model_dump()
            if hasattr(actual_task, "model_dump")
            else actual_task.__dict__
        )
    except AttributeError:
        return (
            False,
            f"next_task content mismatch: objects of type {type(golden_task).__name__} do not support comparison (no model_dump or __dict__).",
        )

    mismatches = []
    all_keys = set(golden_dict.keys()) | set(actual_dict.keys())

    for key in all_keys:
        gv = golden_dict.get(key)
        av = actual_dict.get(key)

        if isinstance(gv, (int, float)) and isinstance(av, (int, float)):
            if abs(gv - av) > tolerance:
                mismatches.append(
                    f"Field '{key}': golden={gv}, actual={av} (numeric diff > {tolerance})"
                )
        elif gv != av:
            # For non-numeric types, require exact match.
            # This could be softened for specific string fields if needed.
            mismatches.append(
                f"Field '{key}': golden='{str(gv)[:50]}...', actual='{str(av)[:50]}...' (content differs)"
            )

    if mismatches:
        return (
            False,
            f"next_task content mismatch for {type(golden_task).__name__}: {'; '.join(mismatches)}",
        )
    return True, ""


def are_outputs_equivalent(
    golden_output: VisionAgentOutput,
    actual_output: VisionAgentOutput,
    tolerance: float = 0.01,
) -> Tuple[bool, List[str]]:
    """
    Compares a golden VisionAgentOutput with an actual one based on defined criteria.
    Returns a boolean indicating equivalence and a list of reasons for differences.
    """
    reasons: List[str] = []

    # 1. stop_current_task: Exact match
    if getattr(golden_output, "stop_current_task", None) != getattr(
        actual_output, "stop_current_task", None
    ):
        reasons.append(
            f"stop_current_task: golden={getattr(golden_output, 'stop_current_task', 'N/A')}, "
            f"actual={getattr(actual_output, 'stop_current_task', 'N/A')}"
        )

    # 2. Text presence fields: observation, thoughts, anticipation, to_tell_user
    for field_name in ["observation", "thoughts", "anticipation", "to_tell_user"]:
        golden_val = getattr(golden_output, field_name, None)
        actual_val = getattr(actual_output, field_name, None)

        golden_has_text = bool(golden_val and str(golden_val).strip())
        actual_has_text = bool(actual_val and str(actual_val).strip())

        if golden_has_text != actual_has_text:
            reasons.append(
                f"{field_name} presence: golden {'has text' if golden_has_text else 'is empty/None'}, "
                f"actual {'has text' if actual_has_text else 'is empty/None'}"
            )

    # 3. new_goal: Presence and content match if present
    golden_new_goal = getattr(golden_output, "new_goal", None)
    actual_new_goal = getattr(actual_output, "new_goal", None)
    golden_has_new_goal = bool(golden_new_goal and str(golden_new_goal).strip())
    actual_has_new_goal = bool(actual_new_goal and str(actual_new_goal).strip())

    if golden_has_new_goal != actual_has_new_goal:
        reasons.append(
            f"new_goal presence: golden {'has goal' if golden_has_new_goal else 'is empty/None'}, "
            f"actual {'has goal' if actual_has_new_goal else 'is empty/None'}"
        )
    elif (
        golden_has_new_goal and golden_new_goal != actual_new_goal
    ):  # Both have goal, content differs
        reasons.append(
            f"new_goal content: golden='{golden_new_goal}', actual='{actual_new_goal}'"
        )

    # 4. next_task: Type and content match (with tolerance)
    task_equivalent, task_reason = compare_task_objects(
        getattr(golden_output, "next_task", None),
        getattr(actual_output, "next_task", None),
        tolerance,
    )
    if not task_equivalent:
        reasons.append(task_reason)

    return not bool(reasons), reasons


async def test_against_golden_set():
    if not GOLDEN_VLM_IO_FILE.exists():
        print(f"Golden assertions file not found: {GOLDEN_VLM_IO_FILE}")
        print(
            "Please create this file with your golden input/output pairs in JSONL format."
        )
        print("Each line should be a JSON object: {'input': {...}, 'output': {...}}")
        return

    print(f"Testing VLM calls against golden set: {GOLDEN_VLM_IO_FILE}\n")

    entry_count = 0
    equivalent_results = 0
    mismatched_results = 0
    failed_processing = 0

    with open(GOLDEN_VLM_IO_FILE, "r") as f:
        for i, line in enumerate(f):
            entry_count = i + 1
            print(f"--- Golden Entry {entry_count} ---")
            try:
                logged_data = json.loads(line)
                input_data_dict = logged_data.get("input")
                golden_output_data_dict = logged_data.get("output")

                if input_data_dict is None or golden_output_data_dict is None:
                    print("Error: Malformed entry. 'input' or 'output' field missing.")
                    failed_processing += 1
                    print("-" * 20 + "\n")
                    continue

                # Validate and reconstruct input object
                vlm_input_obj = MultimodalVisionAgentInput.model_validate(
                    input_data_dict
                )

                # Validate and reconstruct GOLDEN output object
                golden_vlm_output_obj = VisionAgentOutput.model_validate(
                    golden_output_data_dict
                )

                print(f"Input Summary: {summarize_vlm_input(vlm_input_obj)}")
                # print(f"Full Input: {vlm_input_obj.model_dump_json(indent=2)}") # Uncomment for debug
                print("\nGolden Output:")
                print(golden_vlm_output_obj.model_dump_json(indent=2))

                print("\nCalling agent gemini_vision_agent_multimodal_history...")
                try:
                    actual_completion_obj = (
                        await gemini_vision_agent_multimodal_history(vlm_input_obj)
                    )
                    print("\nActual Output from Agent:")
                    print(actual_completion_obj.model_dump_json(indent=2))

                    is_equivalent, diff_reasons = are_outputs_equivalent(
                        golden_vlm_output_obj, actual_completion_obj
                    )

                    if is_equivalent:
                        print("\nComparison: Outputs are EQUIVALENT.")
                        equivalent_results += 1
                    else:
                        print("\nComparison: Outputs are MISMATCHED.")
                        mismatched_results += 1
                        print("Reasons for mismatch:")
                        for reason in diff_reasons:
                            print(f"  - {reason}")

                except Exception as agent_call_exc:
                    print(f"\nError during agent call: {agent_call_exc}")
                    traceback.print_exc()
                    failed_processing += 1

            except json.JSONDecodeError:
                print(f"Error decoding JSON from line {entry_count}")
                failed_processing += 1
            except Exception as e:  # Catches Pydantic validation errors and others
                print(f"Error processing entry {entry_count}: {e}")
                traceback.print_exc()
                failed_processing += 1
            print("-" * 20 + "\n")

    if entry_count == 0:
        print(f"No entries found in {GOLDEN_VLM_IO_FILE}.")
        return

    print("\n--- Test Summary ---")
    print(f"Total golden entries processed: {entry_count}")
    print(f"Equivalent results: {equivalent_results}")
    print(f"Mismatched results: {mismatched_results}")
    print(f"Failed processing/agent calls: {failed_processing}")


async def main():
    # BAML Client Initialization (if needed):
    # from src.baml_client import baml # Or your specific BAML client import
    # await baml.Meta. เงินสด.initialize_async() # This is a placeholder, adapt to your project
    # print("BAML client initialized (if applicable).")

    await test_against_golden_set()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            print("Detected running event loop, using loop.run_until_complete().")
            loop = asyncio.get_event_loop()
            loop.run_until_complete(main())
        else:
            raise
