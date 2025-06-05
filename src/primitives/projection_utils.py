import numpy as np
import math
from math import atan
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.diagonal_movement import DiagonalMovement


def deg2rad(deg):
    """Convert degrees to radians."""
    return deg * np.pi / 180


def rad2deg(rad):
    """Convert radians to degrees."""
    return rad * 180 / np.pi


def compute_the_vignesh_transform(h=0.19663, x_cam=0.0197, pitch_deg=-10):
    """
    Compute homography matrix for ground plane projection.
    """
    # Convert pitch to radians. We define theta = -pitch so that if pitch>0
    # (camera looks downward) then theta is negative.
    # 1) Camera mount in base_link (meters)
    t = np.array([x_cam, 0.0, h])

    # 2) Camera pitch-down 10° about base_link Y
    theta = np.deg2rad(-pitch_deg)
    c, s = np.cos(theta), np.sin(theta)

    # 3) Rotation from base_link → cam frame (about Y):
    R = np.array(
        [
            [c, 0, -s],
            [0, 1, 0],
            [s, 0, c],
        ]
    )

    # 4) Build the 4×4 extrinsic T_cam←base so that
    #    p_cam = T_cam←base @ p_base:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R.dot(t)

    return T


def is_point_in_fov(angle, distance, h_fov_deg):
    """
    Check if a point at given angle and distance is within the camera's field of view.

    Args:
        angle (float): Angle in radians relative to robot's forward direction
        distance (float): Distance in meters
        h_fov_deg (float): Horizontal field of view in degrees
        v_fov_deg (float): Vertical field of view in degrees

    Returns:
        bool: True if the point is within the field of view, False otherwise
    """
    # Convert angle to degrees for easier comparison
    angle_deg = rad2deg(angle)

    # Check if the angle is within the horizontal field of view
    # We need to compare to half the FOV since the angle is relative to center
    if abs(angle_deg) > (h_fov_deg / 2):
        return False

    # For a typical camera, we'd also check vertical FOV,
    # but for navigation points on the ground plane, this is usually not needed
    # as points that are at a reasonable distance will be in the vertical FOV

    # Check if distance is reasonable (not too close or too far)
    # These values can be adjusted based on your robot's camera setup
    if distance < 0.2 or distance > 5.0:
        return False

    return True


def angle_distance_to_image_coordinates(angle, distance, camera_info):
    """
    Convert angle and distance to image coordinates using proper camera projection.

    Args:
        angle (float): Angle in radians relative to robot's forward direction
        distance (float): Distance in meters
        camera_info (dict): Camera information with keys "width", "height",
                            "horizontal_fov", "vertical_fov"
        height_cam (float): Camera height in centimeters
        pitch_deg (float): Camera pitch angle in degrees (90° is looking forward)

    Returns:
        tuple: (x, y) coordinates in the image, or (None, None) if out of field of view
    """
    # Camera parameters
    width = camera_info["width"]
    height = camera_info["height"]
    h_fov_deg = camera_info["horizontal_fov"]
    v_fov_deg = camera_info["vertical_fov"]
    pitch_deg = camera_info["pitch_deg"]
    x_cam = camera_info["x_cam"]
    height_cam = camera_info["height_cam"]

    # Check if the point is within the field of view first
    if not is_point_in_fov(angle, distance, h_fov_deg):
        return (None, None)

    # Compute camera intrinsics and homography
    T_cam_base = compute_the_vignesh_transform(height_cam, x_cam, pitch_deg)

    z = 0
    x = distance * np.cos(angle)
    y = distance * np.sin(angle)

    p_base = np.array([x, y, z, 1.0])
    p_cam = T_cam_base @ p_base

    v_rad = atan(p_cam[2] / p_cam[0])
    h_rad = atan(p_cam[1] / p_cam[0])

    v_deg = rad2deg(v_rad)
    h_deg = rad2deg(h_rad)

    u = width / 2 - h_deg * width / h_fov_deg
    v = height / 2 - v_deg * height / v_fov_deg

    # Check if coordinates are within image bounds
    if u < 0 or u >= width or v < 0 or v >= height:
        return (None, None)

    # Ensure coordinates are within image bounds
    u = max(0, min(width - 1, u))
    v = max(0, min(height - 1, v))

    return (int(u), int(v))


def is_map_location_valid(
    start_world_x,
    start_world_y,
    point_x,
    point_y,
    map_array,
    map_info,
    min_obstacle_distance_m,
    path_to_direct_ratio_threshold=2.0,  # Max path/direct ratio
):
    """
    Check if a given world coordinate is a valid and reachable navigation point.

    Args:
        start_world_x (float): Robot's current X world coordinate.
        start_world_y (float): Robot's current Y world coordinate.
        point_x (float): X world coordinate of the target point.
        point_y (float): Y world coordinate of the target point.
        map_array (np.ndarray): Map occupancy grid data (0=free, 100=obstacle,
                                -1=unknown).
        map_info (dict): Map metadata (resolution, origin_x, origin_y, width, height).
        min_obstacle_distance_m (float): Minimum allowable distance from obstacles
                                     for the target point's vicinity.
        path_to_direct_ratio_threshold (float): Max allowed ratio of
                                                path length to direct distance.

    Returns:
        bool: True if the point is valid and reachable, False otherwise.
    """
    resolution = map_info["resolution"]
    origin_x = map_info["origin_x"]
    origin_y = map_info["origin_y"]
    map_width = map_info["width"]
    map_height = map_info["height"]

    # 1. Initial check: Vicinity of the target point (existing logic)
    # Convert target to grid coordinates
    target_grid_x = int((point_x - origin_x) / resolution)
    target_grid_y = int((point_y - origin_y) / resolution)

    # Skip if target is outside map bounds
    if not (0 <= target_grid_x < map_width and 0 <= target_grid_y < map_height):
        return False

    # Convert start to grid coordinates for line-of-sight check
    start_grid_x_los = int((start_world_x - origin_x) / resolution)
    start_grid_y_los = int((start_world_y - origin_y) / resolution)

    # 1.5. Line-of-sight check
    # Iterate over cells in a straight line from start_grid_los to target_grid
    # Using a simple DDA-like approach for line rasterization
    dx = target_grid_x - start_grid_x_los
    dy = target_grid_y - start_grid_y_los
    steps = max(abs(dx), abs(dy))

    if steps > 0:  # Avoid division by zero if start and end are the same cell
        x_increment = dx / steps
        y_increment = dy / steps
        current_x_check = float(start_grid_x_los)
        current_y_check = float(start_grid_y_los)

        for _ in range(int(steps) + 1):
            check_gx = int(round(current_x_check))
            check_gy = int(round(current_y_check))

            # Ensure the check cell is within map bounds
            if not (0 <= check_gx < map_width and 0 <= check_gy < map_height):
                # Line goes out of bounds, could be considered obstructed or handled as per specific needs.
                # For now, if it goes out of bounds towards the target, the target check will fail.
                # If it goes out of bounds before reaching an obstacle, it's complex.
                # Let's consider a line out of bounds as not clear for simplicity if it's not the target itself.
                if not (check_gx == target_grid_x and check_gy == target_grid_y):
                    # If the out-of-bounds cell is not the target itself, then it's an obstruction.
                    # However, the existing target bounds check should catch this if the target is OOB.
                    # This logic mainly focuses on obstacles *within* bounds along the path.
                    pass  # Let boundary checks for target and start handle OOB.

            # Check for obstacle (value 100)
            # We only care about obstacles within the map bounds.
            if 0 <= check_gx < map_width and 0 <= check_gy < map_height:
                if map_array[check_gy, check_gx] == 100:
                    return False  # Obstacle (100) found in direct line of sight

            current_x_check += x_increment
            current_y_check += y_increment
    # else: start and target are in the same cell or adjacent, line-of-sight is trivial for one cell.

    # Calculate obstacle radius in grid cells for local check
    obstacle_radius_cells = int(min_obstacle_distance_m / resolution)
    min_gx = max(0, target_grid_x - obstacle_radius_cells)
    max_gx = min(map_width - 1, target_grid_x + obstacle_radius_cells)
    min_gy = max(0, target_grid_y - obstacle_radius_cells)
    max_gy = min(map_height - 1, target_grid_y + obstacle_radius_cells)

    for y_cell in range(min_gy, max_gy + 1):
        for x_cell in range(min_gx, max_gx + 1):
            cell_dist_grid = math.sqrt(
                (x_cell - target_grid_x) ** 2 + (y_cell - target_grid_y) ** 2
            )
            if (
                cell_dist_grid <= obstacle_radius_cells
                and map_array[y_cell, x_cell] > 5  # Obstacle
            ):
                return False  # Obstacle found in target vicinity

    # 2. Prepare for Pathfinding
    start_grid_x = int((start_world_x - origin_x) / resolution)
    start_grid_y = int((start_world_y - origin_y) / resolution)

    # Skip if start is outside map bounds
    if not (0 <= start_grid_x < map_width and 0 <= start_grid_y < map_height):
        return False

    # 3. Create Pathfinding Grid
    # python-pathfinding Grid: 0 is blocked, >0 is cost.
    # map_array: 0=free, 100=obstacle, -1=unknown
    walkable_matrix = np.ones_like(map_array, dtype=int)  # Cost 1 for walkable
    walkable_matrix[map_array == 100] = 0  # Obstacles are blocked
    walkable_matrix[map_array == -1] = 0  # Unknown areas are blocked

    # Ensure the matrix is C-contiguous for pathfinding library if it's picky
    # (usually numpy outputs are fine, but an explicit copy can ensure it)
    pf_matrix = np.ascontiguousarray(walkable_matrix)

    pathfinding_grid = Grid(matrix=pf_matrix)

    start_node = pathfinding_grid.node(start_grid_x, start_grid_y)
    end_node = pathfinding_grid.node(target_grid_x, target_grid_y)

    # Check if start or end nodes themselves are on non-walkable cells
    # (e.g. robot spawned inside an obstacle, or target cell is an obstacle)
    # The previous vicinity check for target_grid already handles some of this,
    # but this is an exact cell check.
    if not start_node.walkable or not end_node.walkable:
        return False

    # 4. Run A* Finder
    finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
    path, runs = finder.find_path(start_node, end_node, pathfinding_grid)

    # 5. Evaluate Path
    if not path or len(path) == 0:  # No path found
        return False

    # Heuristic: Check if path length is excessive compared to direct distance
    if len(path) > 1:  # Path exists
        path_length_m = (len(path) - 1) * resolution
        direct_distance_m = math.sqrt(
            (point_x - start_world_x) ** 2 + (point_y - start_world_y) ** 2
        )

        if direct_distance_m == 0:  # Start and end are the same point
            # If direct dist is 0, path len should be 0 (or 1 node if start==end).
            # A path of len 1 means start=end. Len 0 means no path.
            # If path len > 1 here, it's odd, but ratio check might handle it.
            pass  # Valid if path is trivial (len 1) or effectively 0 length.
        elif path_length_m > (path_to_direct_ratio_threshold * direct_distance_m):
            return False  # Path too indirect

    # All checks passed
    return True


def sample_valid_navigation_points(
    current_x,
    current_y,
    current_yaw,
    map_array,
    map_info,
    h_fov_deg,
    distances=[0.5, 1.5],
    angles_deg=[-30, 0, 30],
    min_obstacle_distance=0.20,
    path_ratio_threshold=2.0,  # Max path/direct ratio
    check_map_location_valid=True,
):
    """
    Sample valid navigation points in front of the robot.

    Args:
        current_x (float): Current robot X position in world coordinates
        current_y (float): Current robot Y position in world coordinates
        current_yaw (float): Current robot orientation in radians
        map_array (np.ndarray): Map occupancy grid data
        map_info (dict): Map metadata
        h_fov_deg (float): Horizontal field of view in degrees for FOV check.
        distances (list): List of distances to sample points at (meters).
        angles_deg (list): List of angles to sample points at
                           (degrees, relative to robot).
        min_obstacle_distance (float): Minimum allowable distance from obstacles
            (meters) for the target point's vicinity.
        path_ratio_threshold (float): Max allowed ratio of path len to direct dist
                                     for reachability check.

    Returns:
        tuple: (
            valid_points_absolute, # List of (x,y,theta) world coords
            valid_points_angle_distance, # List of (angle,distance) relative
            invalid_points_absolute, # List of (x,y,theta) world coords (invalid)
            invalid_points_angle_distance # List of (angle,distance) relative (invalid)
        )
    """
    # Get map metadata (not strictly needed here anymore, but good for context)
    # resolution = map_info["resolution"]
    # origin_x = map_info["origin_x"]
    # origin_y = map_info["origin_y"]

    # Sample points in a sector in front of the robot
    valid_points_absolute = []
    valid_points_angle_distance = []
    filtered_points = 0  # Count points filtered due to FOV constraints

    angles = [deg2rad(angle) for angle in angles_deg]

    # For each combination of angle and distance
    invalid_points_absolute = []
    invalid_points_angle_distance = []

    # We should leave a warning here if the minimum obstacle distance
    #  is lower than the resolution of the map.
    if min_obstacle_distance < map_info["resolution"]:
        print(
            f"Warning: Minimum obstacle distance ({min_obstacle_distance}m) is "
            f"lower than the map resolution ({map_info['resolution']}m)."
        )

    for angle_robot_rel in angles:
        print(f"Angle: {angle_robot_rel}")
        for distance in distances:
            # Calculate world coordinates
            # angle_world_vector is the world angle of the vector from robot to point.
            # Convention used: world_angle = robot_yaw - angle_relative_to_forward
            angle_world_vector = current_yaw - angle_robot_rel
            point_x = current_x + distance * np.cos(angle_world_vector)
            point_y = current_y + distance * np.sin(angle_world_vector)
            # point_absolute stores (x, y, orientation_of_robot_at_target)
            # The orientation is aligned with the vector from current pos to target pos.
            point_absolute = (point_x, point_y, angle_world_vector)
            point_angle_distance = (angle_robot_rel, distance)

            # Check if the point would be visible in the camera FOV
            if not is_point_in_fov(angle_robot_rel, distance, h_fov_deg):
                filtered_points += 1
                invalid_points_absolute.append(point_absolute)
                invalid_points_angle_distance.append(point_angle_distance)
                continue

            # Check if point is a valid navigable location using the helper function
            if (
                is_map_location_valid(
                    current_x,
                    current_y,
                    point_x,
                    point_y,
                    map_array,
                    map_info,
                    min_obstacle_distance,
                    path_ratio_threshold,
                )
                or not check_map_location_valid
            ):
                # The navigation target will face in direction from robot to point
                valid_points_absolute.append(point_absolute)
                valid_points_angle_distance.append(point_angle_distance)
            else:
                invalid_points_absolute.append(point_absolute)
                invalid_points_angle_distance.append(point_angle_distance)

    return (
        valid_points_absolute,  # List of (x, y, theta) tuples
        valid_points_angle_distance,  # List of (angle, distance) tuples
        invalid_points_absolute,  # List of (x, y, theta) tuples
        invalid_points_angle_distance,  # List of (angle, distance) tuples
    )


def world_to_grid_coordinates(x, y, map_info):
    """Convert world coordinates to grid coordinates."""
    pixel_x = int((x - map_info["origin_x"]) / map_info["resolution"])
    pixel_y = int((y - map_info["origin_y"]) / map_info["resolution"])
    return pixel_x, pixel_y
