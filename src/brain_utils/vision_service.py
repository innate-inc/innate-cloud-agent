import traceback
from enum import Enum
from typing import Union

from src.agents.baml_agent import vision_agent
from src.agents.gemini_flash_baml_agent import gemini_vision_agent
from src.agents.gemini_flash_baml_multi_agent import (
    gemini_vision_agent_multimodal_history,
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


class VisionService:
    def __init__(self, logger):
        self.logger = logger

    async def call_visual_language_model(
        self,
        base64_img,
        user_prompt_text,
        primitive_in_execution: Union[PrimitiveDefinition, None],
        primitives_list,
        history,
        robot_coords,
        directive=None,
        agent_type: VisionAgentType = VisionAgentType.GEMINI_FLASH,
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
            agent_type: The type of agent to use.
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
                primitive_object = primitive_to_object(primitive_in_execution)

            # Call the appropriate vision agent based on the agent_type
            if agent_type in [
                VisionAgentType.ANTHROPIC,
                VisionAgentType.GEMINI_FLASH,
            ]:
                # Create the input for the vision agent
                vlm_inputs = VisionAgentInput(
                    base64_img=base64_img,
                    user_prompt_text=user_prompt_text,
                    primitive_in_execution=primitive_object,
                    primitives_list=primitives_list,
                    history_as_string=history,
                    robot_coords=robot_coords,
                    directive=directive,
                )

                if agent_type == VisionAgentType.ANTHROPIC:
                    completion = await vision_agent(vlm_inputs)
                else:
                    # Validate gemini_variant
                    if gemini_variant not in (
                        "gemini1",
                        "gemini2",
                        "gemini3",
                        "gemini4",
                    ):
                        self.logger.warning(
                            f"Invalid Gemini variant: {gemini_variant}. Using gemini1."
                        )
                        gemini_variant = "gemini1"

                    completion = await gemini_vision_agent(
                        vlm_inputs, agent_variant=gemini_variant
                    )
                    return completion
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
