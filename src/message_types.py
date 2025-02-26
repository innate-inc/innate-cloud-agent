from enum import Enum
from typing import Dict, Any
from pydantic import BaseModel, field_serializer


# Incoming messages from the client
# SHOULD CORRESPOND TO THE SAME TYPE ON THE ROBOT SIDE
class MessageInType(str, Enum):
    AUTH = "auth"
    DIRECTIVE = "directive"
    IMAGE = "image"
    CHAT_IN = "chat_in"
    PRIMITIVE_ACTIVATED = "primitive_activated"
    PRIMITIVE_COMPLETED = "primitive_completed"
    PRIMITIVE_INTERRUPTED = "primitive_interrupted"
    PRIMITIVE_FAILED = "primitive_failed"
    REGISTER_PRIMITIVES = "register_primitives"


# Outgoing messages from the server/agent
class MessageOutType(str, Enum):
    READY_FOR_IMAGE = "ready_for_image"
    VISION_AGENT_OUTPUT = "vision_agent_output"
    DIRECTIVE_ACK = "directive_ack"  # Example: acknowledgment for a directive
    CHAT_OUT = "chat_out"
    THOUGHTS = "thoughts"
    PRIMITIVES_REGISTERED = "primitives_registered"


class MessageIn(BaseModel):
    type: MessageInType
    payload: Dict[str, Any]


class MessageOut(BaseModel):
    type: MessageOutType
    payload: Dict[str, Any]

    @field_serializer("type")
    def serialize_message_out_type(self, value: MessageOutType) -> str:
        return value.value
