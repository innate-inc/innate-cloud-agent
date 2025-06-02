import os
import json
from typing import List, Any, Dict
from google import genai
from google.genai import types as genai_types  # Renamed to avoid conflict

from src.history.types import (
    HistoryEntry,
    HistoryEntryType,
    DisplayEntryType,
)  # Assuming types are moved


class HistorySummarizer:
    # Gemini API constants for summarization
    GEMINI_MODEL_NAME = "gemini-2.5-flash-preview-05-20"
    GEMINI_TEMPERATURE = 0.7  # Adjusted for more creative summarization
    GEMINI_TOP_P = 0.95
    GEMINI_TOP_K = 64
    GEMINI_MAX_OUTPUT_TOKENS = 2048  # Adjusted for potentially longer summaries

    def __init__(self):
        self.genai_client = None
        # Initialize Gemini API client for summarization
        api_key = os.getenv(
            "GEMINI_API_KEY"
        )  # Used to check if configuration is likely present
        if (
            api_key
        ):  # Check if API key is set in env, assuming genai.configure() or GOOGLE_API_KEY handles actual auth
            try:
                # Instantiate the main client. It will use GOOGLE_API_KEY or prior genai.configure().
                self.genai_client = genai.Client(api_key=api_key)
                print(
                    "Gemini client for history summarization initialized successfully."
                )
            except Exception as e:
                self.genai_client = None
                print(
                    f"Failed to initialize Gemini client for history summarization: {e}"
                )
        else:
            self.genai_client = None
            print(
                "Warning: GEMINI_API_KEY not found in environment variables. "
                "History summarization will not be available."
            )

    def _get_intermediate_display_entries_for_summary(
        self, entries_to_process: List[HistoryEntry]
    ) -> List[Dict[str, Any]]:
        """
        Converts raw HistoryEntry objects to a list of intermediate display dicts for summarization.
        This is a simplified version of the one in History, tailored for summarization input.
        """
        intermediate_display_entries: List[Dict[str, Any]] = []
        for entry in entries_to_process:
            entry_data_common = {
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
                        if "next_task" in data and data["next_task"]:
                            task = data["next_task"]
                            task_name = task["name"]
                            task_inputs = task["inputs"]
                            message_lines = [
                                f"Next task decided: {task_name}",
                                f"  Inputs: {task_inputs}",
                            ]
                            message = "\n".join(message_lines)
                            intermediate_display_entries.append(
                                {
                                    **entry_data_common,
                                    "type": DisplayEntryType.NEXT_TASK_DECIDED,
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
                            "message": entry.description,  # Show raw if JSON fails
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

    def create_summary(self, entries_to_summarize: List[HistoryEntry]) -> str | None:
        """Generates a summary for the given list of history entries."""
        if not self.genai_client:
            print("History summarization skipped: Gemini client not available.")
            return None

        if not entries_to_summarize:
            print("No entries provided to summarize.")
            return None

        # Prepare text for summarization
        intermediate_summary_input_entries = (
            self._get_intermediate_display_entries_for_summary(entries_to_summarize)
        )

        summary_input_lines = []
        for entry_dict in intermediate_summary_input_entries:
            entry_display_type = entry_dict["type"]
            message = entry_dict["message"]
            original_raw_type = entry_dict["original_raw_type"]

            # Override message for image types specifically for the summary prompt
            if original_raw_type == HistoryEntryType.GENERIC_IMAGE:
                message = "[Image data was present]"
            elif original_raw_type == HistoryEntryType.IMAGE_PRE_ACTION:
                message = "[Image data before action was present]"

            prefix = ""
            if entry_display_type == DisplayEntryType.SYSTEM_MESSAGE:
                prefix = "System:"
            elif entry_display_type == DisplayEntryType.AUDIO_IN:
                prefix = "Audio In:"
            elif entry_display_type == DisplayEntryType.AUDIO_OUT:
                prefix = "Audio Out:"
            elif entry_display_type == DisplayEntryType.OBSERVATION:
                prefix = "Observation:"
            elif entry_display_type == DisplayEntryType.THOUGHTS:
                prefix = "Thoughts:"
            elif entry_display_type == DisplayEntryType.ANTICIPATION:
                prefix = "Anticipation:"
            elif entry_display_type == DisplayEntryType.TASK_ACTIVATED:
                prefix = "Task Activated:"
            elif entry_display_type == DisplayEntryType.TASK_INTERRUPTED:
                prefix = "Task Interrupted:"
            elif entry_display_type == DisplayEntryType.TASK_CANCELLED:
                prefix = "Task Cancelled:"
            elif entry_display_type == DisplayEntryType.TASK_COMPLETED:
                prefix = "Task Completed:"
            elif entry_display_type == DisplayEntryType.NEXT_TASK_DECIDED:
                prefix = "Next Task Decided:"
            else:
                # Fallback for any other DisplayEntryType that might be introduced
                prefix = f"{entry_display_type.value.replace('_', ' ').capitalize()}:"
            summary_input_lines.append(f"{prefix} {message}")

        summary_input_text = "\n".join(summary_input_lines)

        if not summary_input_text.strip():
            print("No text content to summarize after formatting.")
            return None

        prompt = f"""
Based on the following sequence of events, generate a concise internal monologue for a robot. 
This monologue should reflect on what the robot has observed and done. It can also include self-reflective questions or ponderings, much like a character in Westworld might introspect.
Be detailed but avoid conversational filler. Focus on the key information and internal state.

Event Log:
---
{summary_input_text}
---

Internal Monologue:
"""

        print(
            f"Sending {len(summary_input_lines)} lines of history to Gemini for summarization."
        )

        try:
            response = self.genai_client.models.generate_content(
                contents=[prompt],  # Pass prompt as a list to contents
                model=self.GEMINI_MODEL_NAME,
                config=genai_types.GenerationConfig(  # Use genai_types
                    temperature=self.GEMINI_TEMPERATURE,
                    top_p=self.GEMINI_TOP_P,
                    top_k=self.GEMINI_TOP_K,
                    max_output_tokens=self.GEMINI_MAX_OUTPUT_TOKENS,
                    # Add thinking_config similar to navigate_in_sight.py
                    # Assuming a default thinking_budget if not specified otherwise.
                    # The value 1024 is taken from navigate_in_sight.py
                    # Removed for now as it's not available in the current API.  We can add it back once available.
                    # thinking_config=genai_types.ThinkingConfig(thinking_budget=1024),
                ),
            )
            summary_text = response.text.strip()
            print(f"Generated summary: {summary_text}")
            return summary_text
        except Exception as e:
            print(f"Error during Gemini API call for summarization: {e}")
            import traceback

            traceback.print_exc()
            return None
