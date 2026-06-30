# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Innate Inc

"""
Pose graph handler for the Brain.
Handles pose_image messages and manages the pose graph memory.
"""

from typing import List, Optional, Tuple

from src.brain_utils.constants import PrimitiveNames
from src.constants_robots import ROBOT_PARAMS_TO_USE
from src.primitives.types import Primitive


AVERAGE_POS_COV_THRESHOLD = ROBOT_PARAMS_TO_USE["average_pos_cov_threshold"]
AVERAGE_YAW_COV_THRESHOLD = ROBOT_PARAMS_TO_USE["average_yaw_cov_threshold"]


class PoseGraphHandler:
    """
    Handles pose_image messages for building and maintaining the pose graph.
    Used for navigate_through_memory functionality.
    """

    def __init__(
        self,
        logger,
        local_primitives_list: List[Primitive],
        connection_id: str,
    ):
        self.logger = logger
        self.local_primitives_list = local_primitives_list
        self.connection_id = connection_id

    def handle_pose_image(
        self,
        image_b64: str,
        x: float,
        y: float,
        theta: float,
        cov_x: float,
        cov_y: float,
        cov_yaw: float,
    ) -> Tuple[Optional[dict], Optional[list]]:
        """
        Handle a pose_image message.

        Args:
            image_b64: Base64 encoded image
            x, y, theta: Robot position and orientation
            cov_x, cov_y, cov_yaw: Position/orientation covariances

        Returns:
            Tuple of (robot_coords, positions):
            - robot_coords: Updated robot coordinates dict if successful, None if skipped
            - positions: All pose graph positions if a node was added, None otherwise
        """
        # Check if covariance is too high (robot position uncertain)
        if self._is_covariance_too_high(cov_x, cov_y, cov_yaw):
            self.logger.debug(
                f"Skipping image addition to pose graph because covariances are too high: "
                f"cov_x={cov_x}, cov_y={cov_y}, cov_yaw={cov_yaw}"
            )
            return None, None

        robot_coords = {"x": x, "y": y, "theta": theta}

        # Always use connection_id as the user token for pose graph memory
        user_token = self.connection_id

        navigate_through_memory = self._get_navigate_through_memory_primitive()

        if navigate_through_memory is None:
            self.logger.error("NavigateThroughMemory primitive not found")
            return robot_coords, None

        pose_graph_memory = navigate_through_memory.pose_graph_memory

        # Check if we should add a node based on coverage gain
        should_add, node_to_evict = pose_graph_memory.should_add_node(
            user_token, x, y, theta
        )
        if not should_add:
            self.logger.debug(
                "Skipping image addition to pose graph: insufficient coverage gain"
            )
            return robot_coords, None

        # Add the image to the pose graph (with optional eviction)
        self.logger.debug(f"Adding image to pose graph with user_token: {user_token}")
        if node_to_evict is not None:
            self.logger.debug(
                f"Evicting node {node_to_evict} to maintain coverage balance"
            )
        node_id = pose_graph_memory.add_image_to_graph(
            user_token, image_b64, x, y, theta, node_to_evict=node_to_evict
        )

        self.logger.info(f"Added image to pose graph with node ID: {node_id}")

        # Return positions so caller can send them to the client
        positions = pose_graph_memory.get_all_positions(user_token)
        return robot_coords, positions

    def _is_covariance_too_high(
        self,
        cov_x: float,
        cov_y: float,
        cov_yaw: float,
    ) -> bool:
        """Check if position covariance is above threshold."""
        avg_pos_cov = (cov_x + cov_y) / 2
        return (
            avg_pos_cov > AVERAGE_POS_COV_THRESHOLD
            or cov_yaw > AVERAGE_YAW_COV_THRESHOLD
        )

    def _get_navigate_through_memory_primitive(self) -> Optional[Primitive]:
        """Get the NavigateThroughMemory primitive from local primitives."""
        return next(
            (
                p
                for p in self.local_primitives_list
                if p.name == PrimitiveNames.NAVIGATE_THROUGH_MEMORY
            ),
            None,
        )

    def reset_pose_graph(self) -> bool:
        """
        Reset the pose graph memory for this connection.

        Returns:
            True if reset was successful, False if primitive not found
        """
        navigate_through_memory = self._get_navigate_through_memory_primitive()

        if navigate_through_memory is None:
            self.logger.error(
                "NavigateThroughMemory primitive not found, "
                "couldn't reset pose graph memory"
            )
            return False

        pose_graph_memory = navigate_through_memory.pose_graph_memory
        pose_graph_memory.reset_user_data(self.connection_id)
        self.logger.info(f"Reset pose graph memory for connection {self.connection_id}")
        return True
