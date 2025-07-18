from enum import Enum
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import os
import json
import threading
import math

from src.agents.types import MultimodalHistoryItem
from src.history.types import HistoryEntryType, DisplayEntryType, HistoryEntry
from src.history.summarizer import HistorySummarizer


def get_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class History:
    NUM_HISTORY_TO_SUMMARIZE = 150
    # Number of entries to consider for get_as_multimodal_list
    # At 0.3-0.4Hz, this value at 5 usually keeps fresh the last 10 seconds of history, the rest is summarized
    # So to get at least 2mn of history at the same rate, we need 5 * 6 * 2 = 60
    # For 5 minutes, we need 5 * 6 * 5 = 150
    MULTIMODAL_HISTORY_COUNT = (
        500  # We make it very large for now as we don't summarize yet.
    )

    def __init__(
        self, max_recent_generic_images: int = 0, max_recent_pre_action_images: int = 0
    ):
        # Entries that have not yet been summarized.
        self.entries: List[HistoryEntry] = []
        self.non_summarized_entries: List[HistoryEntry] = []
        # Simple list for tracking discrepancies as timestamped strings
        self.discrepancies: List[Dict[str, Any]] = []
        self.history_start_time = get_now()
        self.is_summarizing = False
        self.lock = threading.Lock()
        # Max number of recent generic images to include in multimodal history
        self.max_recent_generic_images = max_recent_generic_images
        # Max number of recent pre-action images to include in multimodal history
        self.max_recent_pre_action_images = max_recent_pre_action_images

        # Initialize HistorySummarizer
        self.summarizer = HistorySummarizer()
        if not self.summarizer.genai_client:
            print(
                "Warning: HistorySummarizer's Gemini client failed to initialize. "
                "History summarization will not be available via History class."
            )

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
        robot_coords: Optional[Dict[str, Any]] = None,
    ):
        entry = HistoryEntry(
            timestamp=get_now(),
            type=entry_type,
            description=description,
            robot_coords=robot_coords,
        )
        self.entries.append(entry)
        self.non_summarized_entries.append(entry)

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

    def _get_intermediate_display_entries(
        self, entries_to_process: List[HistoryEntry]
    ) -> List[Dict[str, Any]]:
        """
        Converts raw HistoryEntry objects to a list of intermediate display dicts.
        Each dict contains enough info for deduplication and final formatting.
        """
        intermediate_display_entries: List[Dict[str, Any]] = []
        for i, entry in enumerate(entries_to_process):
            entry_data_common = {
                "timestamp": entry.timestamp,
                "source_index": i,
                "original_raw_type": entry.type,
                "original_raw_description": entry.description,
            }

            if entry.type == HistoryEntryType.VISION_AGENT_OUTPUT:
                try:
                    data = json.loads(entry.description)
                    if isinstance(data, dict):
                        if "observation" in data and data["observation"]:
                            intermediate_display_entries.append(
                                {
                                    **entry_data_common,
                                    "type": DisplayEntryType.OBSERVATION,
                                    "message": data["observation"],
                                }
                            )
                        if "thoughts" in data and data["thoughts"]:
                            intermediate_display_entries.append(
                                {
                                    **entry_data_common,
                                    "type": DisplayEntryType.THOUGHTS,
                                    "message": data["thoughts"],
                                }
                            )
                        if "anticipation" in data and data["anticipation"]:
                            intermediate_display_entries.append(
                                {
                                    **entry_data_common,
                                    "type": DisplayEntryType.ANTICIPATION,
                                    "message": data["anticipation"],
                                }
                            )
                        if "next_primitive" in data and data["next_primitive"]:
                            primitive = data["next_primitive"]
                            primitive_name = primitive["name"]
                            primitive_inputs = primitive["inputs"]
                            message_lines = [
                                f"Next primitive decided: {primitive_name}",
                                f"  Inputs: {primitive_inputs}",
                            ]
                            message = "\n".join(message_lines)
                            intermediate_display_entries.append(
                                {
                                    **entry_data_common,
                                    "type": DisplayEntryType.NEXT_PRIMITIVE_DECIDED,
                                    "message": message,
                                }
                            )
                        if "to_tell_user" in data and data["to_tell_user"]:
                            message = f"To tell user: {data['to_tell_user']}"
                            intermediate_display_entries.append(
                                {
                                    **entry_data_common,
                                    "type": DisplayEntryType.AUDIO_OUT,
                                    "message": message,
                                }
                            )
                except json.JSONDecodeError:
                    intermediate_display_entries.append(
                        {
                            **entry_data_common,
                            "type": DisplayEntryType.SYSTEM_MESSAGE,
                            "message": entry.description,
                        }
                    )
            else:
                message_content = entry.description
                display_type_value = entry.type.value
                display_type = DisplayEntryType.SYSTEM_MESSAGE  # Default

                if entry.type == HistoryEntryType.GENERIC_IMAGE:
                    message_content = "[Image data]"
                    display_type = DisplayEntryType.SYSTEM_MESSAGE
                elif entry.type == HistoryEntryType.IMAGE_PRE_ACTION:
                    message_content = "[Image Before Action]"
                    display_type = DisplayEntryType.SYSTEM_MESSAGE
                else:
                    try:
                        display_type = DisplayEntryType(display_type_value)
                    except ValueError:
                        # Fallback for types not directly in DisplayEntryType
                        display_type = DisplayEntryType.SYSTEM_MESSAGE
                        message_content = (
                            f"{display_type_value.capitalize()}: {entry.description}"
                        )

                intermediate_display_entries.append(
                    {
                        **entry_data_common,
                        "type": display_type,
                        "message": message_content,
                    }
                )
        return intermediate_display_entries

    def _deduplicate_intermediate_entries(
        self, intermediate_entries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Deduplicates a list of intermediate display entries."""
        deduplicated_entries: List[Dict[str, Any]] = []
        last_values: Dict[Any, str] = {}

        def preprocess_message(message: str) -> str:
            return message.strip().lower()

        for entry_dict in intermediate_entries:
            # Pass through types that should not be deduplicated
            if entry_dict["type"] in [
                DisplayEntryType.PRIMITIVE_ACTIVATED,
                DisplayEntryType.PRIMITIVE_INTERRUPTED,
                DisplayEntryType.PRIMITIVE_CANCELLED,
                DisplayEntryType.PRIMITIVE_COMPLETED,
                # System messages (incl. image placeholders) are not deduped
                DisplayEntryType.SYSTEM_MESSAGE,
                DisplayEntryType.AUDIO_IN,
                # Audio out has "still speaking" logic, not simple deduplication
                DisplayEntryType.AUDIO_OUT,
                DisplayEntryType.HISTORY_SUMMARY,
            ]:
                deduplicated_entries.append(entry_dict)
            else:
                # Apply deduplication for other types
                processed_message = preprocess_message(entry_dict["message"])
                processed_last_message = (
                    preprocess_message(last_values.get(entry_dict["type"], ""))
                    if entry_dict["type"] in last_values
                    else ""
                )

                if processed_message != processed_last_message:
                    deduplicated_entries.append(entry_dict)
                    last_values[entry_dict["type"]] = entry_dict["message"]
        return deduplicated_entries

    def _format_intermediate_entry_to_string(
        self, intermediate_entry_dict: Dict[str, Any], now: datetime
    ) -> str:
        """Formats a single deduplicated intermediate entry into a display string."""
        entry_display_type = intermediate_entry_dict["type"]
        message = intermediate_entry_dict["message"]
        timestamp = intermediate_entry_dict["timestamp"]

        prefix = ""
        suffix = ""
        time_col = 16
        prefix_col = 11

        # Determine prefix based on DisplayEntryType
        if entry_display_type == DisplayEntryType.SYSTEM_MESSAGE:
            prefix = "System:"
        elif entry_display_type == DisplayEntryType.AUDIO_IN:
            prefix = "Audio In:"
        elif entry_display_type == DisplayEntryType.AUDIO_OUT:
            prefix = "Audio Out:"
            time_since_started_speaking = now - timestamp
            if (
                time_since_started_speaking.total_seconds()
                > self.estimate_speech_duration(message)
            ):
                suffix = ""
            else:
                suffix = " || STILL SPEAKING, I SHOULD NOT SPEAK YET UNLESS IMPORTANT TO GO OVER MYSELF. "
        elif entry_display_type == DisplayEntryType.OBSERVATION:
            prefix = "Observation:"
        elif entry_display_type == DisplayEntryType.THOUGHTS:
            prefix = "Thoughts:"
        elif entry_display_type == DisplayEntryType.ANTICIPATION:
            prefix = "Anticipation:"
        elif entry_display_type == DisplayEntryType.PRIMITIVE_ACTIVATED:
            prefix = "Primitive Activated:"
        elif entry_display_type == DisplayEntryType.PRIMITIVE_INTERRUPTED:
            prefix = "Primitive Interrupted:"
        elif entry_display_type == DisplayEntryType.PRIMITIVE_CANCELLED:
            prefix = "Primitive Cancelled:"
        elif (
            entry_display_type == DisplayEntryType.PRIMITIVE_COMPLETED
        ):  # Added for completeness
            prefix = "Primitive Completed:"
        elif entry_display_type == DisplayEntryType.HISTORY_SUMMARY:
            prefix = "Summary:"
        elif entry_display_type == DisplayEntryType.NEXT_PRIMITIVE_DECIDED:
            prefix = "Next Primitive Decided:"
            suffix_lines = [
                " I am waiting for confirmation this primitive gets activated,",
                " after which I should be aware that it is running until",
                " cancelled, interrupted, or completed.",
            ]
            suffix = "".join(suffix_lines)
        else:
            prefix = f"{entry_display_type.value}:"

        # Calculate time difference string
        time_diff = now - timestamp
        secs = abs(int(time_diff.total_seconds()))
        if secs < 60:
            time_str = f"{secs}s ago"
        elif secs < 3600:
            minutes = secs // 60
            time_str = f"{minutes}m ago"
        else:
            hours = secs // 3600
            time_str = f"{hours}h ago"

        full_message = f"{message}{suffix}"
        return f"{time_str:>{time_col}} | {prefix:<{prefix_col}} {full_message}"

    def _prepare_unified_display_items(
        self, entries_to_process: List[HistoryEntry], now: datetime
    ) -> List[Dict[str, Any]]:
        """
        Prepares a list of unified display items, each containing formatted text
        and multimodal image information if applicable.
        """
        intermediate_entries = self._get_intermediate_display_entries(
            entries_to_process
        )
        deduplicated_intermediate_entries = self._deduplicate_intermediate_entries(
            intermediate_entries
        )

        unified_items: List[Dict[str, Any]] = []

        # Determine selected images based on entries_to_process
        generic_image_indices = [
            i
            for i, entry in enumerate(entries_to_process)
            if entry.type == HistoryEntryType.GENERIC_IMAGE
        ]
        selected_generic_indices = set(
            generic_image_indices[-self.max_recent_generic_images :]
        )

        pre_action_image_indices = [
            i
            for i, entry in enumerate(entries_to_process)
            if entry.type == HistoryEntryType.IMAGE_PRE_ACTION
        ]
        selected_pre_action_indices = set(
            pre_action_image_indices[-self.max_recent_pre_action_images :]
        )

        selected_image_source_indices = selected_generic_indices.union(
            selected_pre_action_indices
        )

        for d_entry in deduplicated_intermediate_entries:
            source_idx = d_entry["source_index"]  # Index in entries_to_process
            original_raw_type = d_entry["original_raw_type"]
            original_raw_description = d_entry["original_raw_description"]

            is_selected_for_multimodal_image_role = (
                original_raw_type == HistoryEntryType.GENERIC_IMAGE
                or original_raw_type == HistoryEntryType.IMAGE_PRE_ACTION
            ) and source_idx in selected_image_source_indices

            if is_selected_for_multimodal_image_role:
                # Get the original entry to access robot coordinates
                original_entry = entries_to_process[source_idx]
                
                # Create coordinate information for pre-action images
                prefix_message = "This is what I was seeing."
                if (original_raw_type == HistoryEntryType.IMAGE_PRE_ACTION and 
                    original_entry.robot_coords):
                    coords = original_entry.robot_coords
                    x = coords.get('x', 0.0)
                    y = coords.get('y', 0.0)
                    theta = coords.get('theta', 0.0)
                    # Convert theta from radians to degrees for display
                    theta_deg = theta * 180.0 / math.pi
                    prefix_message = f"This is what I was seeing at position (x={x:.2f}, y={y:.2f}, θ={theta_deg:.1f}°)."
                
                # Create and add the prefix text item
                prefix_text_intermediate_entry = {
                    "timestamp": d_entry["timestamp"],  # Use image's timestamp
                    "type": DisplayEntryType.SYSTEM_MESSAGE,
                    "message": prefix_message,
                    # Pass through other fields; safer for formatter.
                    "source_index": d_entry["source_index"],
                    "original_raw_type": DisplayEntryType.SYSTEM_MESSAGE,  # Arbitrary
                    "original_raw_description": prefix_message,
                }
                prefix_formatted_line = self._format_intermediate_entry_to_string(
                    prefix_text_intermediate_entry, now
                )
                unified_items.append(
                    {
                        "formatted_line": prefix_formatted_line,
                        "is_multimodal_image": False,
                        "multimodal_image_content": None,
                    }
                )

            # Format and add the original item (image placeholder or other text)
            formatted_line = self._format_intermediate_entry_to_string(d_entry, now)

            unified_items.append(
                {
                    "formatted_line": formatted_line,
                    "is_multimodal_image": is_selected_for_multimodal_image_role,
                    "multimodal_image_content": (
                        original_raw_description
                        if is_selected_for_multimodal_image_role
                        else None
                    ),
                }
            )
        return unified_items

    def get_as_string(self) -> str:
        now = get_now()
        term_width = 80  # Standard terminal width

        # Consistent with MULTIMODAL_HISTORY_COUNT for display window
        start_index = max(0, len(self.entries) - self.MULTIMODAL_HISTORY_COUNT)
        entries_for_display = self.entries[start_index:]

        if not entries_for_display:  # Handle empty history case
            unified_display_items = []
        else:
            unified_display_items = self._prepare_unified_display_items(
                entries_for_display, now
            )

        output_lines = [item["formatted_line"] for item in unified_display_items]

        # Add a separator line before the current time
        output_lines.append("-" * term_width)
        output_lines.append(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        return "\n".join(output_lines)

    def get_as_multimodal_list(self) -> List[MultimodalHistoryItem]:
        """
        Convert history entries to a list of MultimodalHistoryItem objects,
        merging consecutive text entries and including specific recent images.
        Text content is formatted consistently with get_as_string.
        """
        now = get_now()
        multimodal_list: List[MultimodalHistoryItem] = []
        current_text_lines_block: List[str] = []
        term_width = 80  # For separator

        start_index = max(0, len(self.entries) - self.MULTIMODAL_HISTORY_COUNT)
        relevant_entries = self.entries[start_index:]

        if not relevant_entries:  # Handle empty history case
            return []

        unified_display_items = self._prepare_unified_display_items(
            relevant_entries, now
        )

        for item in unified_display_items:
            if item["is_multimodal_image"]:
                if current_text_lines_block:
                    merged_text = "\n".join(current_text_lines_block)
                    multimodal_list.append(
                        MultimodalHistoryItem(type="text", content=merged_text)
                    )
                    current_text_lines_block = []

                multimodal_list.append(
                    MultimodalHistoryItem(
                        type="image", content=item["multimodal_image_content"]
                    )
                )
            else:
                # Skip image placeholders entirely - we don't want them in multimodal history
                # Check if this is an image placeholder by looking at the formatted line content
                formatted_line = item["formatted_line"]
                if (
                    "[Image data]" in formatted_line
                    or "[Image Before Action]" in formatted_line
                ):
                    continue  # Skip image placeholders

                # This is a text-based entry
                current_text_lines_block.append(item["formatted_line"])

        if current_text_lines_block:
            merged_text = "\n".join(current_text_lines_block)
            multimodal_list.append(
                MultimodalHistoryItem(type="text", content=merged_text)
            )

        # Add current time to the multimodal list
        current_time_str = f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
        separator_line = "-" * term_width

        if multimodal_list and multimodal_list[-1].type == "text":
            multimodal_list[-1].content += f"\n{separator_line}\n{current_time_str}"
        else:
            multimodal_list.append(
                MultimodalHistoryItem(
                    type="text", content=f"{separator_line}\n{current_time_str}"
                )
            )

        # For debugging purposes, save to a file
        with open("multimodal_history.json", "w") as f:
            json.dump([item.model_dump() for item in multimodal_list], f, indent=2)

        return multimodal_list

    def check_and_summarize(self):
        # Check if summarization is needed and not already in progress
        # No lock needed for this initial check of length and is_summarizing flag,
        # as is_summarizing is the primary gatekeeper for launching a new thread.
        if (
            len(self.entries) > 2 * self.NUM_HISTORY_TO_SUMMARIZE
            and not self.is_summarizing
        ):
            self.is_summarizing = (
                True  # Set flag immediately to prevent multiple threads
            )

            # Determine the exact entries to summarize at this moment.
            # We operate on a copy to avoid issues if self.entries is modified
            # by another thread (e.g., add()) before the new thread starts processing.
            # However, the actual modification of self.entries will be locked later.

            # The number of entries to take from the beginning for summarization
            num_to_summarize_snapshot = (
                len(self.entries) - self.NUM_HISTORY_TO_SUMMARIZE
            )

            if num_to_summarize_snapshot <= 0:
                # This can happen if entries were removed between the initial length check
                # and this point, though unlikely in single-threaded add.
                # Or if NUM_HISTORY_TO_SUMMARIZE is very close to the current length.
                self.is_summarizing = False
                return

            # These are the entries that will be replaced by the summary.
            # We pass a copy of these entries to the thread.
            entries_to_process_for_summary_copy = [
                entry.model_copy(deep=True)
                for entry in self.entries[:num_to_summarize_snapshot]
            ]

            if not entries_to_process_for_summary_copy:
                self.is_summarizing = (
                    False  # Should not happen if num_to_summarize_snapshot > 0
                )
                return

            print(
                f"Preparing to summarize {len(entries_to_process_for_summary_copy)} entries in a new thread."
            )

            # Create and start a new daemon thread for summarization.
            # Pass the *copy* of entries and the *count* of entries that this batch represents.
            thread = threading.Thread(
                target=self._perform_threaded_summarization_and_update_history,
                args=(
                    entries_to_process_for_summary_copy,
                    len(entries_to_process_for_summary_copy),
                ),
                daemon=True,  # Daemon threads exit when the main program exits
            )
            thread.start()

    def _perform_threaded_summarization_and_update_history(
        self,
        entries_to_process_for_summary: List[HistoryEntry],
        num_original_entries_to_replace: int,
    ):
        """Internal method to perform summarization in a separate thread and update history."""
        if not self.summarizer or not self.summarizer.genai_client:
            print(
                "History summarization skipped in thread: Summarizer or its Gemini client not available."
            )
            with self.lock:
                self.is_summarizing = False
            return

        print(
            f"Thread started to summarize {len(entries_to_process_for_summary)} entries using HistorySummarizer."
        )

        try:
            # Generate summary using the HistorySummarizer instance
            summary_text = self.summarizer.create_summary(
                entries_to_process_for_summary
            )

            if not summary_text:
                print("Thread received no summary text from HistorySummarizer.")
                # is_summarizing will be reset in finally
                return

            print(f"Thread generated summary via HistorySummarizer: {summary_text}")

            # Create a new summary entry
            summary_entry_timestamp = get_now()
            if entries_to_process_for_summary:
                summary_entry_timestamp = entries_to_process_for_summary[-1].timestamp

            summary_entry = HistoryEntry(
                timestamp=summary_entry_timestamp,
                type=HistoryEntryType.HISTORY_SUMMARY,
                description=summary_text,
            )

            # CRITICAL SECTION: Update shared history lists
            with self.lock:
                if num_original_entries_to_replace > len(self.entries):
                    print(f"Warning: History shrunk unexpectedly. Prepending summary.")
                    self.entries.insert(0, summary_entry)
                else:
                    remaining_entries_after_summary_target = self.entries[
                        num_original_entries_to_replace:
                    ]
                    self.entries = [
                        summary_entry
                    ] + remaining_entries_after_summary_target

                self.non_summarized_entries = list(
                    remaining_entries_after_summary_target
                )

            print(
                f"Thread finished summarizing. {num_original_entries_to_replace} entries replaced with 1 summary entry."
            )

        except Exception as e:
            print(f"Error during threaded history summarization: {e}")
            import traceback

            traceback.print_exc()
        finally:
            with self.lock:
                self.is_summarizing = False
                print("is_summarizing flag reset to False.")

    def summarize(self):
        # This method is now effectively a placeholder or can be removed
        # if check_and_summarize is the sole entry point.
        # For now, let's make it clear it shouldn't be called directly for threaded summarization.
        print(
            "Warning: summarize() called directly. Threaded summarization is initiated via check_and_summarize()."
        )
        # Optionally, it could call check_and_summarize, but that might lead to confusion.
        # Or, it could perform a blocking summarization if really needed (but that defeats the purpose).
        # For now, let it do nothing or raise an error.
        pass  # Or raise NotImplementedError("Use check_and_summarize for threaded summarization")

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
