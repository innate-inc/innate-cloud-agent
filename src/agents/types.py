from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class PrimitiveDefinition(BaseModel):
    name: str
    # This field will accept either a key named "description" (via the alias) or "guideline"
    guideline: Optional[str] = Field(
        default=None,
        description="Guideline for the task. Can be provided as 'description' in inputs.",
    )
    inputs: Dict[str, Any]
    primitive_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for tracking this primitive instance across its lifecycle",
    )

    model_config = ConfigDict(
        # Allow population using the field name (guideline) even if an alias is provided.
        populate_by_name=True
    )


class VisionAgentInput(BaseModel):
    base64_img: str
    user_prompt_text: Optional[str] = None
    primitive_in_execution: Optional[PrimitiveDefinition] = None
    primitives_list: List[PrimitiveDefinition]
    history_as_string: str
    robot_coords: Optional[Dict[str, float]] = None
    directive: Optional[str] = None
