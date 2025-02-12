from enum import Enum
from typing import Dict, Any
from pydantic import BaseModel, field_serializer


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
