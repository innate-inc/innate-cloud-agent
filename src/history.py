from enum import Enum
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
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
        # Full history including those that have been summarized.
        self.full_entries: List[HistoryEntry] = []
        self.history_start_time = datetime.now()
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

    def get_as_string(self) -> str:
        now = datetime.now()
        lines = []
        for entry in self.entries:
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

    #         # Add the summarized items to the full history.
    #         self.full_entries.extend(to_summarize)
    #         summary_entry = HistoryEntry(
    #             timestamp=datetime.now(),
    #             type=HistoryEntryType.HISTORY_SUMMARY,
    #             description=summary,
    #         )
    #         self.full_entries.append(summary_entry)

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
            # Save full (unsummarized) history.
            serializable_full_history = [
                {
                    **entry.model_dump(),
                    "timestamp": entry.timestamp.isoformat(),
                    "type": entry.type.value,
                }
                for entry in self.full_entries
            ]
            folder = os.path.expanduser("./histories/")
            os.makedirs(folder, exist_ok=True)

            full_filename = os.path.join(
                folder,
                f"history_full_{self.history_start_time.strftime('%Y%m%d_%H%M%S')}.json",
            )
            with open(full_filename, "w") as f:
                json.dump(serializable_full_history, f, indent=2)

            full_filename_txt = os.path.join(
                folder,
                f"history_full_{self.history_start_time.strftime('%Y%m%d_%H%M%S')}.txt",
            )
            with open(full_filename_txt, "w") as f:
                f.write(
                    "\n".join(
                        [
                            f"{entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')} - {entry.type.value}: {entry.description}"
                            for entry in self.full_entries
                        ]
                    )
                )

            # Save summarized current history.
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

            print(f"History saved to {filename} and {filename_txt}")
        except Exception as e:
            print(f"Error saving history: {e}")
