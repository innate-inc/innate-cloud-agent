from enum import Enum
from pydantic import BaseModel
from datetime import datetime


class HistoryEntryType(Enum):
    AUDIO_IN = "audio_in"
    VISION_AGENT_OUTPUT = "vision_agent_output"
    HISTORY_SUMMARY = "history_summary"
    SYSTEM_MESSAGE = "system_message"
    PRIMITIVE_ACTIVATED = "primitive_activated"
    PRIMITIVE_INTERRUPTED = "primitive_interrupted"
    PRIMITIVE_CANCELLED = "primitive_cancelled"
    PRIMITIVE_COMPLETED = "primitive_completed"
    PRIMITIVE_FEEDBACK = "primitive_feedback"
    GENERIC_IMAGE = "generic_image"
    IMAGE_PRE_ACTION = "image_pre_action"


class DisplayEntryType(Enum):
    OBSERVATION = "observation"
    THOUGHTS = "thoughts"
    ANTICIPATION = "anticipation"
    AUDIO_IN = "audio_in"
    AUDIO_OUT = "audio_out"
    SYSTEM_MESSAGE = "system_message"
    NEXT_PRIMITIVE_DECIDED = "next_primitive_decided"
    PRIMITIVE_ACTIVATED = "primitive_activated"
    PRIMITIVE_INTERRUPTED = "primitive_interrupted"
    PRIMITIVE_CANCELLED = "primitive_cancelled"
    PRIMITIVE_COMPLETED = "primitive_completed"
    HISTORY_SUMMARY = "history_summary"
    PRIMITIVE_FEEDBACK = "primitive_feedback"


class HistoryEntry(BaseModel):
    timestamp: datetime
    type: HistoryEntryType
    description: str
