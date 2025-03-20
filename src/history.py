from enum import Enum
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import os
import json


# Define history entry types – here we include only entries that are relevant.
class HistoryEntryType(Enum):
    CHAT_MESSAGE = "chat_message"
    VISION_AGENT_OUTPUT = "vision_agent_output"
    HISTORY_SUMMARY = "history_summary"
    SYSTEM_MESSAGE = "system_message"


class HistoryEntry(BaseModel):
    timestamp: datetime
    type: HistoryEntryType
    description: str
    users_implicated: List[str] = []


class History:
    MAX_HISTORY_LENGTH = 40
    NUM_HISTORY_TO_SUMMARIZE = 20

    def __init__(self):
        # Entries that have not yet been summarized.
        self.entries: List[HistoryEntry] = []
        # Simple list for tracking discrepancies as timestamped strings
        self.discrepancies: List[Dict[str, Any]] = []
        self.history_start_time = datetime.now(timezone.utc)
        self.is_summarizing = False

    def reset(self):
        """Reset the history to an empty state."""
        self.entries = []
        self.discrepancies = []
        self.history_start_time = datetime.now(timezone.utc)
        self.is_summarizing = False

    def add(
        self,
        entry_type: HistoryEntryType,
        description: str,
        users_implicated: Optional[List[str]] = None,
    ):
        if users_implicated is None:
            users_implicated = []
        entry = HistoryEntry(
            timestamp=datetime.now(),
            type=entry_type,
            description=description,
            users_implicated=users_implicated,
        )
        self.entries.append(entry)
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
            "timestamp": datetime.now(),
            "message": message,
        }

        # Only add to the discrepancies list, not to the regular history
        self.discrepancies.append(discrepancy)

    def get_discrepancies_as_string(self) -> str:
        """Get a string representation of the discrepancy history."""
        now = datetime.now()
        lines = []

        for entry in self.discrepancies:
            time_diff = now - entry["timestamp"]
            seconds_diff = int(time_diff.total_seconds())

            if seconds_diff < 60:
                time_str = f"{seconds_diff} seconds ago"
            else:
                minutes_diff = seconds_diff // 60
                time_str = (
                    f"{minutes_diff} minute{'' if minutes_diff == 1 else 's'} ago"
                )

            lines.append(f"{time_str} - {entry['message']}")
            if entry["users_implicated"]:
                users = ", ".join(entry["users_implicated"])
                lines.append(f"  Users: {users}")

        lines.append(f"\nCurrent time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        return "\n".join(lines)

    def get_as_string(self) -> str:
        now = datetime.now()
        lines = []

        # Get only the latest 50 entries
        latest_entries = self.entries[-50:] if len(self.entries) > 50 else self.entries

        for entry in latest_entries:
            time_diff = now - entry.timestamp
            seconds_diff = int(time_diff.total_seconds())

            if seconds_diff < 60:
                time_str = f"{seconds_diff} seconds ago"
            else:
                minutes_diff = seconds_diff // 60
                time_str = (
                    f"{minutes_diff} minute{'' if minutes_diff == 1 else 's'} ago"
                )

            lines.append(f"{time_str} - {entry.type.value}: {entry.description}")

        lines.append(f"\nCurrent time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        return "\n".join(lines)

    def check_and_summarize(self):
        if len(self.entries) > self.MAX_HISTORY_LENGTH:
            self.summarize()

    def summarize(self):
        if self.is_summarizing:
            return  # Skip if a summarization is already in progress.
        if len(self.entries) <= self.NUM_HISTORY_TO_SUMMARIZE:
            return  # Not enough entries to summarize.

        # TODO: Implement below with BAML. For now, we don't summarize yet

        return

    #     self.is_summarizing = True
    #     try:
    #         # Take the first NUM_HISTORY_TO_SUMMARIZE entries.
    #         to_summarize = self.entries[: self.NUM_HISTORY_TO_SUMMARIZE]

    #         # Generate a textual summary.
    #         summary = self.generate_summary(to_summarize)

    #         # Replace the summarized block with a summary entry.
    #         summary_entry = HistoryEntry(
    #             timestamp=datetime.now(),
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
            print(f"Saving history to {filename}")
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

                discrepancy_filename_txt = os.path.join(
                    folder,
                    f"discrepancies_"
                    f"{self.history_start_time.strftime('%Y%m%d_%H%M%S')}.txt",
                )
                with open(discrepancy_filename_txt, "w") as f:
                    f.write(self.get_discrepancies_as_string())
        except Exception as e:
            print(f"Error saving history: {e}")
