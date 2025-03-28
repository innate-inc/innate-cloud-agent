from enum import Enum
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime, timezone
import os
import json


# Define history entry types – here we include only entries that are relevant.
class HistoryEntryType(Enum):
    AUDIO_IN = "audio_in"
    VISION_AGENT_OUTPUT = "vision_agent_output"
    HISTORY_SUMMARY = "history_summary"
    SYSTEM_MESSAGE = "system_message"
    TASK_ACTIVATED = "task_activated"
    TASK_INTERRUPTED = "task_interrupted"
    TASK_CANCELLED = "task_cancelled"


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

    def __init__(self):
        # Entries that have not yet been summarized.
        self.entries: List[HistoryEntry] = []
        self.non_summarized_entries: List[HistoryEntry] = []
        # Simple list for tracking discrepancies as timestamped strings
        self.discrepancies: List[Dict[str, Any]] = []
        self.history_start_time = get_now()
        self.is_summarizing = False

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
                            message = f"Next task decided: {task_name} with inputs: {task_inputs}"
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
                # Map HistoryEntryType to DisplayEntryType
                display_type = DisplayEntryType(entry.type.value)
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
        deduplicated_entries = []
        last_values: Dict[Any, str] = {}
        for entry in display_entries:
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
                suffix = f" I am waiting for confirmation this task gets activated, after which I should be aware that it is running until cancelled, interrupted, or completed."
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
            lines.append(
                f"{time_str:>{time_col}} | {prefix:<{prefix_col}} {entry['message']}{suffix}"
            )

        # Add a separator line before the current time
        lines.append("-" * term_width)
        lines.append(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        return "\n".join(lines)

    def check_and_summarize(self):
        if len(self.entries) > self.MAX_HISTORY_LENGTH:
            self.summarize()

    def summarize(self):
        if self.is_summarizing:
            return  # Skip if a summarization is already in progress.
        if len(self.entries) <= self.NUM_HISTORY_TO_SUMMARIZE:
            return  # Not enough entries to summarize.

        return

    #     self.is_summarizing = True
    #     try:
    #         # Take the first NUM_HISTORY_TO_SUMMARIZE entries.
    #         to_summarize = self.entries[: self.NUM_HISTORY_TO_SUMMARIZE]

    #         # Generate a textual summary.
    #         summary = self.generate_summary(to_summarize)

    #         # Replace the summarized block with a summary entry.
    #         summary_entry = HistoryEntry(
    #             timestamp=get_now(),
    #             type=HistoryEntryType.HISTORY_SUMMARY,
    #             description=summary,
    #         )
    #         # Replace the summarized block with the summary entry.
    #         self.entries = [summary_entry] + self.entries[
    #             self.NUM_HISTORY_TO_SUMMARIZE :
    #         ]
    #     finally:
    #         self.is_summarizing = False

    # def generate_summary(self, entries: List[HistoryEntry]) -> str:
    #     # Build a string representation of the subset of entries.
    #     history_str = "\n".join(
    #         [
    #             f"{entry.timestamp.strftime('%H:%M:%S')} - {entry.type.value}: {entry.description}"
    #             for entry in entries
    #         ]
    #     )
    #     try:
    #         # Call OpenAI's ChatCompletion API to generate a summary.
    #         response = openai.ChatCompletion.create(
    #             model="gpt-4",  # Adjust the model as needed.
    #             messages=[
    #                 {
    #                     "role": "system",
    #                     "content": "You are an AI summarizer. Given a sequence of events from a robot history, provide a concise summary of the key events and interactions.",
    #                 },
    #                 {
    #                     "role": "user",
    #                     "content": f"Summarize the following robot history:\n\n{history_str}",
    #                 },
    #             ],
    #             max_tokens=150,
    #         )
    #         summary_text = response.choices[0].message["content"].strip()
    #         return summary_text
    #     except Exception as e:
    #         print(f"Error generating summary: {e}")
    #         return "Summary generation failed."

    def save(self):
        try:
            folder = os.path.expanduser("./histories/")
            os.makedirs(folder, exist_ok=True)

            # Save current history
            serializable_history = [
                {
                    **entry.model_dump(),
                    "timestamp": entry.timestamp.isoformat(),
                    "type": entry.type.value,
                }
                for entry in self.entries
            ]
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
