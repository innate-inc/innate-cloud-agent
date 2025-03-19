import os
import json
import shutil
from datetime import datetime
from typing import Optional, List

from src.history import History, HistoryEntryType
from src.brain_utils.logger import BrainLogger
from src.primitives.navigate_through_memory import NavigateThroughMemory


class MemoryStateManager:
    """
    Manages saving and loading memory states.
    This class handles the storage and retrieval of brain state.
    """

    def __init__(self, logger: BrainLogger, connection_id: str):
        """
        Initialize the MemoryStateManager.

        Args:
            logger: The logger instance from the Brain
            connection_id: The connection ID for the current brain instance
        """
        self.logger = logger
        self.connection_id = connection_id

        # Create directory for memory states if it doesn't exist
        root_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        self.memory_states_dir = os.path.join(root_dir, "memory_states")
        os.makedirs(self.memory_states_dir, exist_ok=True)

    def get_available_states(self) -> List[str]:
        """
        Get a list of available memory states.

        Returns:
            List of state names
        """
        try:
            if os.path.exists(self.memory_states_dir):
                return [
                    d
                    for d in os.listdir(self.memory_states_dir)
                    if os.path.isdir(os.path.join(self.memory_states_dir, d))
                ]
            return []
        except Exception as e:
            self.logger.error(f"Error listing memory states: {e}")
            return []

    async def save_memory_state(
        self,
        state_name: str,
        history: History,
        pose_graph_primitive: Optional[NavigateThroughMemory],
    ) -> bool:
        """
        Save the current brain state (history and pose graph memory) to a directory.

        Args:
            state_name: Name for the saved state
            history: The History object from the Brain
            pose_graph_primitive: The NavigateThroughMemory primitive instance

        Returns:
            Boolean indicating success or failure
        """
        try:
            # Create timestamp-based state name if none provided
            if not state_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                state_name = f"memory_state_{timestamp}"

            # Create directory for this memory state
            state_dir = os.path.join(self.memory_states_dir, state_name)
            if os.path.exists(state_dir):
                self.logger.warning(
                    f"Memory state '{state_name}' already exists, overwriting"
                )
                shutil.rmtree(state_dir)

            os.makedirs(state_dir, exist_ok=True)

            # Create subdirectories
            history_dir = os.path.join(state_dir, "histories")
            pose_graph_dir = os.path.join(state_dir, "pose_graphs")
            images_dir = os.path.join(state_dir, "images")

            os.makedirs(history_dir, exist_ok=True)
            os.makedirs(pose_graph_dir, exist_ok=True)
            os.makedirs(images_dir, exist_ok=True)

            # 1. Save history
            # First save the current history
            history.save()

            # Copy history files
            src_history_dir = os.path.expanduser("./histories/")
            if os.path.exists(src_history_dir):
                history_timestamp = history.history_start_time.strftime("%Y%m%d_%H%M%S")
                for filename in os.listdir(src_history_dir):
                    if filename.startswith(f"history_{history_timestamp}"):
                        src_file = os.path.join(src_history_dir, filename)
                        dst_file = os.path.join(history_dir, filename)
                        shutil.copy2(src_file, dst_file)

            # 2. Save pose graph memory if available
            if pose_graph_primitive:
                pose_graph_memory = pose_graph_primitive.pose_graph_memory

                # Copy pose graph file
                src_graph_file = os.path.join(
                    pose_graph_memory.graphs_dir, f"{self.connection_id}.pkl"
                )
                if os.path.exists(src_graph_file):
                    dst_graph_file = os.path.join(
                        pose_graph_dir, f"{self.connection_id}.pkl"
                    )
                    shutil.copy2(src_graph_file, dst_graph_file)

                # Copy images
                src_images_dir = os.path.join(
                    pose_graph_memory.images_dir, self.connection_id
                )
                if os.path.exists(src_images_dir):
                    dst_user_images_dir = os.path.join(images_dir, self.connection_id)
                    os.makedirs(dst_user_images_dir, exist_ok=True)

                    for filename in os.listdir(src_images_dir):
                        src_file = os.path.join(src_images_dir, filename)
                        dst_file = os.path.join(dst_user_images_dir, filename)
                        if os.path.isfile(src_file):
                            shutil.copy2(src_file, dst_file)
            else:
                self.logger.error(
                    "Navigate primitive not found, couldn't save pose graph"
                )
                return False

            self.logger.info(f"Successfully saved memory state '{state_name}'")
            return True

        except Exception as e:
            self.logger.error(f"Error saving memory state: {e}")
            return False

    async def load_memory_state(
        self,
        state_name: str,
        history: History,
        pose_graph_primitive: Optional[NavigateThroughMemory],
    ) -> bool:
        """
        Load a saved brain state (history and pose graph memory).

        Args:
            state_name: Name of the state to load
            history: The History object to update
            pose_graph_primitive: The NavigateThroughMemory primitive instance

        Returns:
            Boolean indicating success or failure
        """
        try:
            # Check if the specified memory state exists
            state_dir = os.path.join(self.memory_states_dir, state_name)
            if not os.path.exists(state_dir):
                self.logger.error(f"Memory state '{state_name}' does not exist")
                return False

            # Get source directories
            src_history_dir = os.path.join(state_dir, "histories")
            src_pose_graph_dir = os.path.join(state_dir, "pose_graphs")
            src_images_dir = os.path.join(state_dir, "images")

            # Reset history
            history.reset()

            # Load pose graph memory if available
            if pose_graph_primitive:
                pose_graph_memory = pose_graph_primitive.pose_graph_memory

                # 1. Copy pose graph file
                src_graph_file = os.path.join(
                    src_pose_graph_dir, f"{self.connection_id}.pkl"
                )
                if os.path.exists(src_graph_file):
                    # Reset pose graph memory first to clear current data
                    pose_graph_memory.reset_user_data(self.connection_id)

                    # Copy graph file
                    dst_graph_file = os.path.join(
                        pose_graph_memory.graphs_dir, f"{self.connection_id}.pkl"
                    )
                    shutil.copy2(src_graph_file, dst_graph_file)

                    # Reload the graph in memory
                    if self.connection_id in pose_graph_memory._user_graphs:
                        del pose_graph_memory._user_graphs[self.connection_id]
                    pose_graph_memory.get_user_graph(self.connection_id)

                # 2. Copy images
                src_user_images_dir = os.path.join(src_images_dir, self.connection_id)
                if os.path.exists(src_user_images_dir):
                    dst_user_images_dir = os.path.join(
                        pose_graph_memory.images_dir, self.connection_id
                    )

                    # Clear destination directory first
                    if os.path.exists(dst_user_images_dir):
                        for filename in os.listdir(dst_user_images_dir):
                            file_path = os.path.join(dst_user_images_dir, filename)
                            if os.path.isfile(file_path):
                                os.unlink(file_path)

                    # Create directory if it doesn't exist
                    os.makedirs(dst_user_images_dir, exist_ok=True)

                    # Copy all images
                    for filename in os.listdir(src_user_images_dir):
                        src_file = os.path.join(src_user_images_dir, filename)
                        dst_file = os.path.join(dst_user_images_dir, filename)
                        if os.path.isfile(src_file):
                            shutil.copy2(src_file, dst_file)
            else:
                self.logger.error(
                    "Navigate primitive not found, couldn't load pose graph"
                )
                return False

            # 3. Load history from files
            # This requires us to reconstruct the history from the saved files
            # Find the latest history files
            if os.path.exists(src_history_dir):
                # Find the most recent history file
                history_files = [
                    f
                    for f in os.listdir(src_history_dir)
                    if f.startswith("history_") and f.endswith(".json")
                ]
                if history_files:
                    # Get the timestamp from the filename to set history_start_time
                    latest_file = sorted(history_files)[-1]
                    timestamp_str = latest_file.replace("history_", "").replace(
                        ".json", ""
                    )
                    try:
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        history.history_start_time = timestamp

                        # Load entries from the file
                        src_file = os.path.join(src_history_dir, latest_file)
                        with open(src_file, "r") as f:
                            entries_data = json.load(f)

                        # Reconstruct history entries
                        for entry_data in entries_data:
                            entry_type = HistoryEntryType(entry_data["type"])
                            description = entry_data["description"]
                            users_implicated = entry_data.get("users_implicated", [])

                            # Add entry to history
                            history.add(entry_type, description, users_implicated)
                    except Exception as e:
                        self.logger.error(f"Error loading history: {e}")
                        # Continue with empty history

            self.logger.info(f"Successfully loaded memory state '{state_name}'")
            return True

        except Exception as e:
            self.logger.error(f"Error loading memory state: {e}")
            return False
