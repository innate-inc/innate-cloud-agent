class MaxRetriesExceededException(Exception):
    """
    Exception raised when the maximum number of retries has been exceeded
    when calling a vision agent.
    """

    def __init__(self, agent_type: str, max_retries: int, last_error: Exception = None):
        """
        Initialize the exception with information about the agent and retries.

        Args:
            agent_type: The type of agent that failed (anthropic, gemini_flash)
            max_retries: The maximum number of retries that were attempted
            last_error: The last error that occurred during the final retry attempt
        """
        self.agent_type = agent_type
        self.max_retries = max_retries
        self.last_error = last_error

        message = (
            f"Maximum retries ({max_retries}) exceeded when calling {agent_type} "
            f"vision agent. The agent failed to produce a valid response after "
            f"multiple attempts."
        )

        if last_error:
            message += f" Last error: {str(last_error)}"

        super().__init__(message)


class UnforeseenBamlClientError(Exception):
    """
    Exception raised when an unexpected BAML client error occurs that is not
    handled by the retry mechanism.
    """

    def __init__(self, message: str, original_error: Exception = None):
        """
        Initialize the exception with information about the unexpected error.

        Args:
            message: A descriptive message about the error
            original_error: The original exception that was caught
        """
        self.original_error = original_error
        super().__init__(message)
