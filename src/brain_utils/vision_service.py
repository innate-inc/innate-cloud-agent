import traceback
from enum import Enum
from typing import Union
import json
from pathlib import Path

from src.primitives.types import Primitive

# Flag to control serialization of VLM inputs and outputs
SERIALIZE_VLM_IO = True  # Set to False to disable
VLM_IO_DUMP_FILE = Path("test_data/vlm_io_dump.jsonl")

from src.agents.baml_agent import vision_agent
from src.agents.gemini_flash_baml_agent import gemini_vision_agent
from src.agents.gemini_flash_baml_multi_agent import (
    gemini_vision_agent_multimodal_history,
)
from src.agents.native_gemini_vision_agent import (
    native_gemini_vision_agent_multimodal_history,
)
from src.agents.types import (
    MultimodalVisionAgentInput,
    PrimitiveDefinition,
    VisionAgentInput,
)
from src.baml_client.types import VisionAgentOutput
from src.agents.exceptions import MaxRetriesExceededException, UnforeseenBamlClientError
from src.primitives.transforms import primitive_to_object


class VisionAgentType(str, Enum):
    ANTHROPIC = "anthropic"
    GEMINI_FLASH = "gemini_flash"
    GEMINI_FLASH_MULTI = "gemini_flash_multi"
    NATIVE_GEMINI_MULTI = "native_gemini_multi"


class VisionService:
    def __init__(self, logger):
        self.logger = logger

    async def call_visual_language_model(
        self,
        base64_img,
        user_prompt_text,
        primitive_in_execution: Union[PrimitiveDefinition, Primitive, None],
        primitives_list,
        history,
        robot_coords,
        directive=None,
        agent_type: VisionAgentType = VisionAgentType.NATIVE_GEMINI_MULTI,
        gemini_variant: str = "gemini1",
        additional_image_data: dict = {},
    ) -> Union[VisionAgentOutput]:
        """
        Calls the external visual language model with the given inputs.

        Args:
            base64_img: A base64-encoded image.
            user_prompt_text: The text provided by the user.
            primitive_in_execution: The current primitive in execution.
            primitives_list: A list of primitives.
            history: A history of events.
            robot_coords: A dictionary with the robot's coordinates.
            directive: A directive to steer the vision language model.
            agent_type: The type of agent to use. Options: "gemini_flash_multi" (BAML-based),
                "native_gemini_multi" (native Google Gemini implementation)
            gemini_variant: The variant of Gemini agent to use if agent_type is
                GEMINI_FLASH. Options: "gemini1", "gemini2", "gemini3", "gemini4"

        Returns:
            A VisionAgentOutput object.
        """
        try:
            current_primitive = (
                primitive_in_execution.name if primitive_in_execution else "None"
            )
            agent_name = agent_type.value
            self.logger.info(
                f"Calling {agent_name} vision model while current primitive is "
                + (
                    f"\033[1;34m{current_primitive} (id: {primitive_in_execution.primitive_id})\033[0m"
                    if current_primitive != "None"
                    else current_primitive
                )
            )
            if user_prompt_text:
                self.logger.info(
                    f"Sending user message to {agent_name} vision agent: "
                    f"{user_prompt_text}"
                )

            # Convert primitive_in_execution if needed
            primitive_object = None
            if primitive_in_execution:
                if isinstance(primitive_in_execution, Primitive):
                    primitive_object = primitive_to_object(primitive_in_execution)
                else:
                    primitive_object = primitive_in_execution

            # Call the appropriate vision agent based on the agent_type
            if agent_type in [
                VisionAgentType.ANTHROPIC,
                VisionAgentType.GEMINI_FLASH,
            ]:  # deprecated
                raise ValueError(f"Unsupported agent type: {agent_type}")
            elif agent_type == VisionAgentType.GEMINI_FLASH_MULTI:
                vlm_inputs = MultimodalVisionAgentInput(
                    base64_img=base64_img,
                    user_prompt_text=user_prompt_text,
                    primitive_in_execution=primitive_object,
                    primitives_list=primitives_list,
                    multimodal_history=history,
                    robot_coords=robot_coords,
                    directive=directive,
                    additional_image_data=additional_image_data,
                )
                completion = await gemini_vision_agent_multimodal_history(vlm_inputs)

                if SERIALIZE_VLM_IO:
                    try:
                        VLM_IO_DUMP_FILE.parent.mkdir(parents=True, exist_ok=True)
                        # Assuming vlm_inputs and completion are Pydantic models
                        data_to_serialize = {
                            "input": vlm_inputs.model_dump(mode="json"),
                            "output": completion.model_dump(mode="json"),
                        }
                        with open(VLM_IO_DUMP_FILE, "a") as f:
                            f.write(json.dumps(data_to_serialize) + "\n")
                        self.logger.info(f"Serialized VLM I/O to {VLM_IO_DUMP_FILE}")
                    except Exception as ser_exc:
                        self.logger.error(
                            f"Error during VLM I/O serialization: {ser_exc}, Input: {vlm_inputs}, Completion: {completion}"
                        )
                return completion
            elif agent_type == VisionAgentType.NATIVE_GEMINI_MULTI:
                vlm_inputs = MultimodalVisionAgentInput(
                    base64_img=base64_img,
                    user_prompt_text=user_prompt_text,
                    primitive_in_execution=primitive_object,
                    primitives_list=primitives_list,
                    multimodal_history=history,
                    robot_coords=robot_coords,
                    directive=directive,
                    additional_image_data=additional_image_data,
                )
                completion = await native_gemini_vision_agent_multimodal_history(
                    vlm_inputs
                )

                if SERIALIZE_VLM_IO:
                    try:
                        VLM_IO_DUMP_FILE.parent.mkdir(parents=True, exist_ok=True)
                        # Assuming vlm_inputs and completion are Pydantic models
                        data_to_serialize = {
                            "input": vlm_inputs.model_dump(mode="json"),
                            "output": completion.model_dump(mode="json"),
                        }
                        with open(VLM_IO_DUMP_FILE, "a") as f:
                            f.write(json.dumps(data_to_serialize) + "\n")
                        self.logger.info(f"Serialized VLM I/O to {VLM_IO_DUMP_FILE}")
                    except Exception as ser_exc:
                        self.logger.error(
                            f"Error during VLM I/O serialization: {ser_exc}, Input: {vlm_inputs}, Completion: {completion}"
                        )
                return completion
            else:
                raise ValueError(f"Unsupported agent type: {agent_type}")

        except MaxRetriesExceededException as e:
            # Handle the max retries exceeded exception specifically
            self.logger.error(
                f"Maximum retries exceeded for {e.agent_type} vision model."
            )
            return VisionAgentOutput(
                stop_current_task=True,
                observation=(
                    "The brain failed after multiple attempts, "
                    "so it stopped the current task."
                ),
                thoughts=(
                    f"Maximum retries exceeded. "
                    f"The {e.agent_type} vision agent failed to produce a valid "
                    f"response after {e.max_retries} attempts."
                ),
                new_goal=None,
                next_task=None,
                anticipation=None,
                to_tell_user=(
                    "BEEP BOOP BEEP BOOP, the brain failed after multiple attempts. "
                    "Maybe next time it will work?"
                ),
            )
        except UnforeseenBamlClientError as e:
            # Handle unforeseen BAML client errors
            self.logger.error(f"Unforeseen BAML client error: {e}")
            has_original = hasattr(e, "original_error")
            original_err = e.original_error if has_original else "Unknown"
            return VisionAgentOutput(
                stop_current_task=True,
                observation=(
                    "The brain encountered an unexpected error with the vision model."
                ),
                thoughts=(
                    f"Unforeseen BAML client error: {str(e)}\n"
                    f"Original error: {original_err}"
                ),
                new_goal=None,
                next_task=None,
                anticipation=None,
                to_tell_user=(
                    "BEEP BOOP BEEP BOOP, the brain encountered an unexpected error. "
                    "Please try again later."
                ),
            )
        except Exception as e:
            # Handle other exceptions
            self.logger.error(
                f"Error calling {agent_type.value} vision model: {e}, traceback: {traceback.format_exc()}"
            )
            return VisionAgentOutput(
                stop_current_task=True,
                observation="The brain failed, so it stopped the current task.",
                thoughts=(
                    f"Fallback due to error: {str(e)}\n"
                    f"Traceback: {traceback.format_exc()}"
                ),
                new_goal=None,
                next_task=None,
                anticipation=None,
                to_tell_user=(
                    "BEEP BOOP BEEP BOOP, the brain failed. "
                    "Stopping the current task."
                ),
            )
