import traceback
from src.agents.baml_agent import vision_agent
from src.agents.types import VisionAgentInput, PrimitiveDefinition
from src.baml_client.types import VisionAgentOutput
from src.primitives.transforms import primitive_to_object


class VisionService:
    def __init__(self, logger):
        self.logger = logger

    async def call_visual_language_model(
        self,
        base64_img,
        user_prompt_text,
        primitive_in_execution,
        primitives_list,
        history_as_string,
        robot_coords,
    ) -> VisionAgentOutput:
        """
        Calls the external visual language model with the given inputs.
        """
        try:
            current_primitive = (
                primitive_in_execution.name if primitive_in_execution else "None"
            )
            self.logger.info(
                f"Calling visual language model while current primitive is {current_primitive}"
            )
            if user_prompt_text:
                self.logger.info(
                    f"Sending user message to vision agent: {user_prompt_text}"
                )

            # Convert primitive_in_execution if needed
            primitive_object = None
            if primitive_in_execution:
                primitive_object = primitive_to_object(primitive_in_execution)

            # Create the input for the vision agent
            vlm_inputs = VisionAgentInput(
                base64_img=base64_img,
                user_prompt_text=user_prompt_text,
                primitive_in_execution=primitive_object,
                primitives_list=[primitive_to_object(prim) for prim in primitives_list],
                history_as_string=history_as_string,
                robot_coords=robot_coords,
            )

            completion = await vision_agent(vlm_inputs)
            return completion
        except Exception as e:
            self.logger.error(f"Error calling visual language model: {e}")
            return VisionAgentOutput(
                stop_current_task=True,
                observation="The brain failed, so it stopped the current task.",
                thoughts=f"Fallback due to error: {str(e)}\nTraceback: {traceback.format_exc()}",
                new_goal=None,
                next_task=None,
                anticipation=None,
                to_tell_user="BEEP BOOP BEEP BOOP, the brain failed. Stopping the current task.",
            )
