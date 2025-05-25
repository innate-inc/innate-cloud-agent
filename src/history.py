from enum import Enum
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime, timezone
import os
import json

from src.agents.types import MultimodalHistoryItem


# Define history entry types – here we include only entries that are relevant.
class HistoryEntryType(Enum):
    AUDIO_IN = "audio_in"
    VISION_AGENT_OUTPUT = "vision_agent_output"
    HISTORY_SUMMARY = "history_summary"
    SYSTEM_MESSAGE = "system_message"
    TASK_ACTIVATED = "task_activated"
    TASK_INTERRUPTED = "task_interrupted"
    TASK_CANCELLED = "task_cancelled"
    TASK_COMPLETED = "task_completed"
    GENERIC_IMAGE = "generic_image"
    IMAGE_PRE_ACTION = "image_pre_action"


# Internal display types for formatting vision agent outputs
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


class HistoryEntry(BaseModel):
    timestamp: datetime
    type: HistoryEntryType
    description: str


def get_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class History:
    MAX_HISTORY_LENGTH = 40
    NUM_HISTORY_TO_SUMMARIZE = 20
    # Number of entries to consider for get_as_multimodal_list
    MULTIMODAL_HISTORY_COUNT = 50

    def __init__(
        self, max_recent_generic_images: int = 3, max_recent_pre_action_images: int = 3
    ):
        # Entries that have not yet been summarized.
        self.entries: List[HistoryEntry] = []
        self.non_summarized_entries: List[HistoryEntry] = []
        # Simple list for tracking discrepancies as timestamped strings
        self.discrepancies: List[Dict[str, Any]] = []
        self.history_start_time = get_now()
        self.is_summarizing = False
        # Max number of recent generic images to include in multimodal history
        self.max_recent_generic_images = max_recent_generic_images
        # Max number of recent pre-action images to include in multimodal history
        self.max_recent_pre_action_images = max_recent_pre_action_images

    def reset(self):
        """Reset the history to an empty state."""
        self.entries = []
        self.discrepancies = []
        self.history_start_time = get_now()
        self.is_summarizing = False

    def add(
        self,
        entry_type: HistoryEntryType,
        description: str,
    ):
        entry = HistoryEntry(
            timestamp=get_now(),
            type=entry_type,
            description=description,
        )
        self.entries.append(entry)
        self.non_summarized_entries.append(entry)
        self.check_and_summarize()

    def record_discrepancy(
        self,
        message: str,
    ):
        """
        Record a discrepancy or anomaly that seems out of the ordinary.
        This is intended to be called from outside the class.

        Args:
            message: A simple description of the discrepancy
        """
        print(f"\033[33mRecording discrepancy: {message}\033[0m")

        discrepancy = {
            "timestamp": get_now(),
            "message": message,
        }

        # Only add to the discrepancies list, not to the regular history
        self.discrepancies.append(discrepancy)

    def estimate_speech_duration(self, text: str) -> float:
        """Estimate speech duration in seconds based on word count."""
        # Average speaking rate is about 150 words per minute
        words = len(text.split())
        return (words / 150) * 60  # Convert to seconds

    def get_as_string(self) -> str:
        now = get_now()
        lines = []
        term_width = 80  # Standard terminal width

        # Get only the latest 50 entries
        latest_entries = self.entries

        # First loop: Convert entries to display entries
        display_entries = []
        for entry in latest_entries:
            if entry.type == HistoryEntryType.VISION_AGENT_OUTPUT:
                try:
                    data = json.loads(entry.description)
                    if isinstance(data, dict):
                        if "observation" in data and data["observation"]:
                            display_entries.append(
                                {
                                    "type": DisplayEntryType.OBSERVATION,
                                    "message": data["observation"],
                                    "timestamp": entry.timestamp,
                                }
                            )
                        if "thoughts" in data and data["thoughts"]:
                            display_entries.append(
                                {
                                    "type": DisplayEntryType.THOUGHTS,
                                    "message": data["thoughts"],
                                    "timestamp": entry.timestamp,
                                }
                            )
                        if "anticipation" in data and data["anticipation"]:
                            display_entries.append(
                                {
                                    "type": DisplayEntryType.ANTICIPATION,
                                    "message": data["anticipation"],
                                    "timestamp": entry.timestamp,
                                }
                            )
                        if "next_task" in data and data["next_task"]:
                            task = data["next_task"]
                            task_name = task["name"]
                            task_inputs = task["inputs"]
                            # Adjusted to fit within line limits
                            message_lines = [
                                f"Next task decided: {task_name}",
                                f"  Inputs: {task_inputs}",
                            ]
                            message = "\n".join(message_lines)
                            display_entries.append(
                                {
                                    "type": DisplayEntryType.NEXT_TASK_DECIDED,
                                    "message": message,
                                    "timestamp": entry.timestamp,
                                }
                            )
                        if "to_tell_user" in data and data["to_tell_user"]:
                            message = f"To tell user: {data['to_tell_user']}"
                            display_entries.append(
                                {
                                    "type": DisplayEntryType.AUDIO_OUT,
                                    "message": message,
                                    "timestamp": entry.timestamp,
                                }
                            )
                except json.JSONDecodeError:
                    display_entries.append(
                        {
                            "type": DisplayEntryType.SYSTEM_MESSAGE,
                            "message": entry.description,
                            "timestamp": entry.timestamp,
                        }
                    )
            else:
                # Map HistoryEntryType to DisplayEntryType, handling IMAGE type
                if entry.type == HistoryEntryType.GENERIC_IMAGE:
                    display_entries.append(
                        {
                            "type": DisplayEntryType.SYSTEM_MESSAGE,
                            "message": "[Image data]",
                            "timestamp": entry.timestamp,
                        }
                    )
                elif entry.type == HistoryEntryType.IMAGE_PRE_ACTION:
                    display_entries.append(
                        {
                            "type": DisplayEntryType.SYSTEM_MESSAGE,
                            "message": "[Image Before Action]",
                            "timestamp": entry.timestamp,
                        }
                    )
                else:
                    display_type_value = entry.type.value
                    # Ensure the value is a valid DisplayEntryType member name
                    # This handles cases where HistoryEntryType might have values
                    # not in DisplayEntryType
                    try:
                        display_type = DisplayEntryType(display_type_value)
                    except ValueError:
                        # Fallback for types not directly in DisplayEntryType (e.g.
                        # custom tasks if any) or treat as a generic system message
                        display_type = DisplayEntryType.SYSTEM_MESSAGE
                        entry.description = (
                            f"{display_type_value.capitalize()}: {entry.description}"
                        )

                    display_entries.append(
                        {
                            "type": display_type,
                            "message": entry.description,
                            "timestamp": entry.timestamp,
                        }
                    )

        # Function to preprocess messages for comparison
        def preprocess_message(message: str) -> str:
            """Normalize message by trimming whitespace and converting to lowercase."""
            return message.strip().lower()

        # Second loop: Deduplicate entries
        # Don't deduplicate task status messages, system messages, or audio messages.
        deduplicated_entries = []
        last_values: Dict[Any, str] = {}
        for entry in display_entries:
            if entry["type"] in [
                DisplayEntryType.TASK_ACTIVATED,
                DisplayEntryType.TASK_INTERRUPTED,
                DisplayEntryType.TASK_CANCELLED,
                DisplayEntryType.TASK_COMPLETED,
                DisplayEntryType.SYSTEM_MESSAGE,
                DisplayEntryType.AUDIO_IN,
                DisplayEntryType.AUDIO_OUT,
                DisplayEntryType.HISTORY_SUMMARY,
            ]:
                entry["processed_message"] = entry["message"]
                deduplicated_entries.append(entry)
            else:
                processed_message = preprocess_message(entry["message"])
                processed_last_message = (
                    preprocess_message(last_values.get(entry["type"], ""))
                    if entry["type"] in last_values
                    else ""
                )

                if processed_message != processed_last_message:
                    # Add processed message to the entry for future reference
                    entry["processed_message"] = processed_message
                    deduplicated_entries.append(entry)
                    last_values[entry["type"]] = entry["message"]

        # Third loop: Format and display entries
        for entry in deduplicated_entries:
            suffix = ""
            # Format the entry based on its type
            if entry["type"] == DisplayEntryType.SYSTEM_MESSAGE:
                prefix = "System:"
            elif entry["type"] == DisplayEntryType.AUDIO_IN:
                prefix = "Audio In:"
            elif entry["type"] == DisplayEntryType.AUDIO_OUT:
                prefix = "Audio Out:"

                # Determine if we have spoken the message.
                time_since_started_speaking = now - entry["timestamp"]

                if (
                    time_since_started_speaking.total_seconds()
                    > self.estimate_speech_duration(entry["message"])
                ):
                    suffix = ""
                else:
                    suffix = (
                        " || STILL SPEAKING, I SHOULD NOT REPEAT A SIMILAR MESSAGE. "
                    )

            elif entry["type"] == DisplayEntryType.OBSERVATION:
                prefix = "Observation:"
            elif entry["type"] == DisplayEntryType.THOUGHTS:
                prefix = "Thoughts:"
            elif entry["type"] == DisplayEntryType.ANTICIPATION:
                prefix = "Anticipation:"
            elif entry["type"] == DisplayEntryType.TASK_ACTIVATED:
                prefix = "Task Activated:"
            elif entry["type"] == DisplayEntryType.TASK_INTERRUPTED:
                prefix = "Task Interrupted:"
            elif entry["type"] == DisplayEntryType.TASK_CANCELLED:
                prefix = "Task Cancelled:"
            elif entry["type"] == DisplayEntryType.HISTORY_SUMMARY:
                prefix = "Summary:"
            elif entry["type"] == DisplayEntryType.NEXT_TASK_DECIDED:
                prefix = "Next Task Decided:"
                # Adjusted to fit within line limits
                # The suffix implies waiting, so it's context for the agent.
                suffix_lines = [
                    " I am waiting for confirmation this task gets activated,",
                    " after which I should be aware that it is running until",
                    " cancelled, interrupted, or completed.",
                ]
                suffix = "".join(suffix_lines)
            else:
                prefix = f"{entry['type'].value}:"

            # Calculate time difference
            time_diff = now - entry["timestamp"]
            secs = abs(int(time_diff.total_seconds()))
            if secs < 60:
                time_str = f"{secs}s ago"
            elif secs < 3600:
                minutes = secs // 60
                time_str = f"{minutes}m ago"
            else:
                hours = secs // 3600
                time_str = f"{hours}h ago"

            # Format the line with proper alignment
            time_col = 16  # Width for the timestamp column
            prefix_col = 11  # Width for the prefix column

            # Add the line
            # Adjusted to fit within line limits
            full_message = f"{entry['message']}{suffix}"
            # Split message if too long for a single line, though usually
            # handled by terminal wrapping
            lines.append(
                f"{time_str:>{time_col}} | {prefix:<{prefix_col}} {full_message}"
            )

        # Add a separator line before the current time
        lines.append("-" * term_width)
        lines.append(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        return "\n".join(lines)

    def _get_text_lines_from_entry(self, entry: HistoryEntry) -> List[str]:
        """Converts a history entry to a list of formatted text lines."""
        lines = []
        if entry.type == HistoryEntryType.AUDIO_IN:
            lines.append(f"User: {entry.description}")
        elif entry.type == HistoryEntryType.VISION_AGENT_OUTPUT:
            try:
                data = json.loads(entry.description)
                if isinstance(data, dict):
                    if data.get("observation"):
                        lines.append(f"Observation: {data['observation']}")
                    if data.get("thoughts"):
                        lines.append(f"Thoughts: {data['thoughts']}")
                    if data.get("to_tell_user"):
                        lines.append(f"Assistant: {data['to_tell_user']}")
                    # If no specific fields are found, we don't add generic text here
                    # to keep the multimodal history clean for specific content types.
            except json.JSONDecodeError:
                # Log malformed JSON from vision agent output as a system log.
                lines.append(
                    f"System Log (Undecipherable Vision Output): {entry.description}"
                )
        elif entry.type == HistoryEntryType.SYSTEM_MESSAGE:
            lines.append(f"System: {entry.description}")
        elif entry.type == HistoryEntryType.HISTORY_SUMMARY:
            lines.append(f"Summary: {entry.description}")
        elif entry.type in [
            HistoryEntryType.TASK_ACTIVATED,
            HistoryEntryType.TASK_COMPLETED,
            HistoryEntryType.TASK_CANCELLED,
            HistoryEntryType.TASK_INTERRUPTED,
        ]:
            lines.append(f"Task Status ({entry.type.value}): {entry.description}")
        # IMAGE types are handled by the caller (get_as_multimodal_list),
        # so no 'else' needed for them here in _get_text_lines_from_entry.
        elif entry.type not in [
            HistoryEntryType.GENERIC_IMAGE,
            HistoryEntryType.IMAGE_PRE_ACTION,
        ]:
            lines.append(f"{entry.type.value.capitalize()}: {entry.description}")
        return lines

    def get_as_multimodal_list(self) -> List[MultimodalHistoryItem]:
        """
        Convert history entries to a list of MultimodalHistoryItem objects,
        merging consecutive text entries and including specific recent images.
        """
        multimodal_list: List[MultimodalHistoryItem] = []
        current_text_lines_block: List[str] = []

        start_index = max(0, len(self.entries) - self.MULTIMODAL_HISTORY_COUNT)
        latest_entries = self.entries[start_index:]

        # Identify indices of GENERIC_IMAGE entries
        generic_image_indices_in_latest = [
            i
            for i, entry in enumerate(latest_entries)
            if entry.type == HistoryEntryType.GENERIC_IMAGE
        ]
        selected_generic_image_indices = set(
            generic_image_indices_in_latest[-self.max_recent_generic_images :]
        )

        # Identify indices of IMAGE_PRE_ACTION entries
        pre_action_image_indices_in_latest = [
            i
            for i, entry in enumerate(latest_entries)
            if entry.type == HistoryEntryType.IMAGE_PRE_ACTION
        ]
        selected_pre_action_image_indices = set(
            pre_action_image_indices_in_latest[-self.max_recent_pre_action_images :]
        )

        # Combine selected image indices
        selected_image_indices = selected_generic_image_indices.union(
            selected_pre_action_image_indices
        )

        for i, entry in enumerate(latest_entries):
            if (
                entry.type == HistoryEntryType.GENERIC_IMAGE
                or entry.type == HistoryEntryType.IMAGE_PRE_ACTION
            ):
                # This is an image entry
                if current_text_lines_block:
                    merged_text = "\n".join(current_text_lines_block)
                    multimodal_list.append(
                        MultimodalHistoryItem(type="text", content=merged_text)
                    )
                    current_text_lines_block = []

                if i in selected_image_indices:
                    multimodal_list.append(
                        MultimodalHistoryItem(type="image", content=entry.description)
                    )
            else:
                # This is a text-based entry
                text_lines_for_entry = self._get_text_lines_from_entry(entry)
                current_text_lines_block.extend(text_lines_for_entry)

        if current_text_lines_block:
            merged_text = "\n".join(current_text_lines_block)
            multimodal_list.append(
                MultimodalHistoryItem(type="text", content=merged_text)
            )

        return multimodal_list

    def check_and_summarize(self):
        if len(self.entries) > self.MAX_HISTORY_LENGTH:
            self.summarize()

    def summarize(self):
        if self.is_summarizing:
            return  # Skip if a summarization is already in progress.
        if len(self.entries) <= self.NUM_HISTORY_TO_SUMMARIZE:
            return  # Not enough entries to summarize.

        return

    def save(self):
        try:
            folder = os.path.expanduser("./histories/")
            os.makedirs(folder, exist_ok=True)

            # Save current history
            serializable_history = []
            for entry in self.entries:
                entry_data = entry.model_dump()
                entry_data["timestamp"] = entry.timestamp.isoformat()
                entry_data["type"] = entry.type.value

                if entry.type == HistoryEntryType.GENERIC_IMAGE:
                    entry_data["description"] = "[Image data]"
                elif entry.type == HistoryEntryType.IMAGE_PRE_ACTION:
                    entry_data["description"] = "[Image Before Action]"

                serializable_history.append(entry_data)

            filename = os.path.join(
                folder,
                f"history_{self.history_start_time.strftime('%Y%m%d_%H%M%S')}.json",
            )
            with open(filename, "w") as f:
                json.dump(serializable_history, f, indent=2)

            filename_txt = os.path.join(
                folder,
                f"history_{self.history_start_time.strftime('%Y%m%d_%H%M%S')}.txt",
            )
            with open(filename_txt, "w") as f:
                f.write(self.get_as_string())

            # Save discrepancy history if there are any discrepancies
            if self.discrepancies:
                serializable_discrepancies = [
                    {
                        **entry,
                        "timestamp": entry["timestamp"].isoformat(),
                    }
                    for entry in self.discrepancies
                ]
                discrepancy_filename = os.path.join(
                    folder,
                    f"discrepancies_"
                    f"{self.history_start_time.strftime('%Y%m%d_%H%M%S')}.json",
                )
                print(f"Saving discrepancies to {discrepancy_filename}")
                with open(discrepancy_filename, "w") as f:
                    json.dump(serializable_discrepancies, f, indent=2)
        except Exception as e:
            print(f"Error saving history: {e}")
