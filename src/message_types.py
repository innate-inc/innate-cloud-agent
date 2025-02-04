# message_types.py

from enum import Enum
from typing import Dict, Optional, List, Any
from pydantic import BaseModel, field_serializer


class TaskType(Enum):
    NAVIGATION_IN_SIGHT = "navigation_in_sight"
    NAVIGATION_OUT_OF_SIGHT = "navigation_out_of_sight"
    ACTION_WITH_ARM = "action_with_arm"
    ASK_FOR_INFORMATION = "ask_for_information"
    VELOCITY_CONTROL = "velocity_control"


class Task(BaseModel):
    type: TaskType
    description: str

    @field_serializer("type")
    def serialize_task_type(self, value: TaskType) -> str:
        return value.value


# Incoming messages from the client
class MessageInType(str, Enum):
    AUTH = "auth"
    DIRECTIVE = "directive"
    IMAGE = "image"
    CHAT_IN = "chat_in"


# Outgoing messages from the server/agent
class MessageOutType(str, Enum):
    READY_FOR_IMAGE = "ready_for_image"
    ACTION_TO_DO = "action_to_do"
    VISION_AGENT_OUTPUT = "vision_agent_output"
    DIRECTIVE_ACK = "directive_ack"  # Example: acknowledgment for a directive
    CHAT_OUT = "chat_out"


class MessageIn(BaseModel):
    type: MessageInType
    payload: Dict[str, Any]


class MessageOut(BaseModel):
    type: MessageOutType
    payload: Dict[str, Any]

    @field_serializer("type")
    def serialize_message_out_type(self, value: MessageOutType) -> str:
        return value.value


class VisionAgentOutput(BaseModel):
    """Mirrors your orchestrator.agent.models.VisionAgentOutput shape."""

    stop_current_task: bool
    observation: str
    thoughts: str
    new_goal: Optional[str]
    next_task: Optional[Task] = None
    users_implicated: List[str]
    anticipation: Optional[str]
    to_tell_user: Optional[str]

    # If 'next_task' is provided, Task's serializer will ensure TaskType is converted.
