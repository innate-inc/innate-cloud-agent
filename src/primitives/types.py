# %%
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class Primitive(ABC):
    def __init__(self):
        self._feedback_callback = None

    @property
    @abstractmethod
    def name(self):
        """
        The name of the primitive.
        Must be defined by every subclass.
        """
        pass

    @abstractmethod
    def execute(self, *args, **kwargs):
        """
        Execute the primitive.

        Subclasses must implement this method.
        Returns a tuple of (result_message, result_status) where result_status
        is a PrimitiveResult enum value.
        """
        pass

    def guidelines(self):
        """
        Optionally provide guidelines for this primitive.
        Subclasses may override this method if guidelines are available.
        """
        return None

    def guidelines_when_running(self):
        """
        Optionally provide guidelines for this primitive when it is running.
        Subclasses may override this method if guidelines are available.
        """
        return None

    def few_shot_examples(self) -> List[Dict[str, Any]]:
        """
        Optionally provide few-shot examples for this primitive.
        
        Each example should be a dictionary with:
        - 'image_path': Path to the example image (relative to few_shot directory)
        - 'situation': Description of the situation
        - 'choice': The primitive choice and parameters that were made
        - 'reasoning': Why this choice was good
        
        Returns:
            List of few-shot example dictionaries
        """
        return []

    def set_feedback_callback(self, callback):
        """Sets the feedback callback function."""
        self._feedback_callback = callback
        print(f"Feedback callback set for primitive {self.name}.")

    def _send_feedback(self, message: str):
        """Sends feedback if the callback is set."""
        if self._feedback_callback:
            try:
                self._feedback_callback(message)
            except Exception as e:
                print(
                    f"ERROR: Error sending feedback for primitive {self.name}: {e} (logger not available)."
                )
