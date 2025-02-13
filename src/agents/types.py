from typing import Optional, List
from pydantic import BaseModel, field_serializer
from enum import Enum


class TaskType(Enum):
    # NAVIGATION_IN_SIGHT = "navigation_in_sight"
    # NAVIGATION_OUT_OF_SIGHT = "navigation_out_of_sight"
    NAVIGATION_TO_POSITION = "navigation_to_position"
    # ACTION_WITH_ARM = "action_with_arm"
    # ASK_FOR_INFORMATION = "ask_for_information"
    # VELOCITY_CONTROL = "velocity_control"


class NavigationToPosition(BaseModel):
    x: float
    y: float


class Task(BaseModel):
    type: TaskType
    position: NavigationToPosition

    @field_serializer("type")
    def serialize_task_type(self, value: TaskType) -> str:
        return value.value


class VisionAgentOutput(BaseModel):
    """Mirrors your orchestrator.agent.models.VisionAgentOutput shape."""

    stop_current_task: bool
    observation: str
    thoughts: str
    new_goal: Optional[str]
    next_task: Optional[Task] = None
    anticipation: Optional[str]
    to_tell_user: Optional[str]

    # If 'next_task' is provided, Task's serializer will ensure TaskType is converted.
