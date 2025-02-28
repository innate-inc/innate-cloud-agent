import traceback


class BrainLogger:
    def __init__(self, connection_id):
        self.connection_id = connection_id

    def info(self, message):
        print(f"[Brain {self.connection_id}] {message}")

    def error(self, message, exception=None):
        error_msg = f"[Brain {self.connection_id}] Error: {message}"
        if exception:
            error_msg += f". {str(exception)}\nTraceback: {traceback.format_exc()}"
        print(error_msg)
