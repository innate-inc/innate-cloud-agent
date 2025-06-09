from enum import Enum
from pydantic import BaseModel
from datetime import datetime


class HistoryEntryType(Enum):
    AUDIO_IN = "audio_in"
    VISION_AGENT_OUTPUT = "vision_agent_output"
    HISTORY_SUMMARY = "history_summary"
    SYSTEM_MESSAGE = "system_message"
    TASK_ACTIVATED = "task_activated"
    TASK_INTERRUPTED = "task_interrupted"
    TASK_CANCELLED = "task_cancelled"
    TASK_COMPLETED = "task_completed"
    TASK_FEEDBACK = "task_feedback"
    GENERIC_IMAGE = "generic_image"
    IMAGE_PRE_ACTION = "image_pre_action"


class DisplayEntryType(Enum):
    OBSERVATION = "observation"
    THOUGHTS = "thoughts"
    ANTICIPATION = "anticipation"
    AUDIO_IN = "audio_in"
    AUDIO_OUT = "audio_out"
    SYSTEM_MESSAGE = "system_message"
    NEXT_TASK_DECIDED = "next_task_decided"
    TASK_ACTIVATED = "task_activated"
    TASK_INTERRUPTED = "task_interrupted"
    TASK_CANCELLED = "task_cancelled"
    TASK_COMPLETED = "task_completed"
    HISTORY_SUMMARY = "history_summary"
    TASK_FEEDBACK = "task_feedback"


class HistoryEntry(BaseModel):
    timestamp: datetime
    type: HistoryEntryType
    description: str
