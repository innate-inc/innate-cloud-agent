# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Innate Inc

import traceback


class BrainLogger:
    def __init__(self, connection_id):
        self.connection_id = connection_id

    def debug(self, message):
        pass  # For now, disable debug logging
        # print(f"[Brain {self.connection_id}] {message}")

    def warn(self, message):
        print(f"\033[93m[Brain {self.connection_id}] {message}\033[0m")

    def info(self, message):
        print(f"[Brain {self.connection_id}] {message}")

    def error(self, message, exception=None):
        error_msg = f"[Brain {self.connection_id}] Error: {message}"
        if exception:
            error_msg += f". {str(exception)}\nTraceback: {traceback.format_exc()}"
        print(error_msg)
