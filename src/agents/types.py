from typing import Literal, Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict, AliasChoices


class PrimitiveDefinition(BaseModel):
    name: str
    guidelines: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("guidelines", "guideline", "description"),
        description=(
            "Guidelines for the skill. Can be provided as "
            "'guidelines', 'guideline', or 'description'."
        ),
    )
    guidelines_when_running: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "guidelines_when_running",
            "guideline_when_running",
            "description_when_running",
        ),
        description=(
            "Guidelines for the skill while running. Can be provided as "
            "'guidelines_when_running', 'guideline_when_running', "
            "or 'description_when_running'."
        ),
    )
    inputs: Dict[str, Any]
    primitive_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for tracking this primitive instance across its lifecycle",
    )

    model_config = ConfigDict(
        # Allow population using field names even when aliases are provided.
        populate_by_name=True
    )


class VisionAgentInput(BaseModel):
    base64_img: str
    user_prompt_text: Optional[str] = None
    primitive_in_execution: Optional[PrimitiveDefinition] = None
    primitives_list: List[PrimitiveDefinition]
    history_as_string: str
    robot_coords: Optional[Dict[str, Union[float, str]]] = None
    directive: Optional[str] = None


class MultimodalHistoryItem(BaseModel):
    type: Literal["text", "image"]
    content: str = Field(
        description="Content of the history item. For images, will be '[image]' when serialized"
    )

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        if self.type == "image":
            data["content"] = "[image]"
        return data


class MultimodalVisionAgentInput(BaseModel):
    base64_img: str
    user_prompt_text: Optional[str] = None
    primitive_in_execution: Optional[PrimitiveDefinition] = None
    primitives_list: List[PrimitiveDefinition]
    multimodal_history: List[MultimodalHistoryItem]
    robot_coords: Optional[Dict[str, Union[float, str]]] = None
    directive: Optional[str] = None
    additional_image_data: Optional[Dict[str, str]] = None


class VisionAgentOutput(BaseModel):
    stop_current_task: bool
    observation: str
    thoughts: str
    new_goal: Optional[str] = None
    next_task: Optional[Any] = None
    anticipation: Optional[str] = None
    to_tell_user: Optional[str] = None
