import numpy as np
import math
from math import atan


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

    print(f"Camera info: {camera_info}")

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
    point_x, point_y, map_array, map_info, min_obstacle_distance_m
):
    """
    Check if a given world coordinate is a valid navigation point on the map.

    Args:
        point_x (float): X world coordinate of the point.
        point_y (float): Y world coordinate of the point.
        map_array (np.ndarray): Map occupancy grid data.
        map_info (dict): Map metadata (resolution, origin_x, origin_y, width, height).
        min_obstacle_distance_m (float): Minimum allowable distance from obstacles
                                     (meters).

    Returns:
        bool: True if the point is valid, False otherwise.
    """
    resolution = map_info["resolution"]
    origin_x = map_info["origin_x"]
    origin_y = map_info["origin_y"]
    map_width = map_info["width"]
    map_height = map_info["height"]

    # Convert to grid coordinates
    grid_x = int((point_x - origin_x) / resolution)
    grid_y = int((point_y - origin_y) / resolution)

    # Skip if outside map bounds
    if grid_x < 0 or grid_x >= map_width or grid_y < 0 or grid_y >= map_height:
        return False

    # Calculate obstacle radius in grid cells
    obstacle_radius_cells = int(min_obstacle_distance_m / resolution)

    # Define a search window around the point
    min_gx = max(0, grid_x - obstacle_radius_cells)
    max_gx = min(map_width - 1, grid_x + obstacle_radius_cells)
    min_gy = max(0, grid_y - obstacle_radius_cells)
    max_gy = min(map_height - 1, grid_y + obstacle_radius_cells)

    # Check if any cell within the radius is an obstacle (value = 100)
    for y_cell in range(min_gy, max_gy + 1):
        for x_cell in range(min_gx, max_gx + 1):
            # Calculate distance from this cell to the target cell in grid units
            cell_dist_grid = math.sqrt((x_cell - grid_x) ** 2 + (y_cell - grid_y) ** 2)

            # If this cell is within our search radius and is an obstacle
            # (Occupancy grid value 100 indicates obstacle)
            if (
                cell_dist_grid <= obstacle_radius_cells
                and map_array[y_cell, x_cell] == 100
            ):
                return False  # Obstacle found

    return True  # Point is valid


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
            (meters)

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

    for angle_robot_rel in angles:
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
            if is_map_location_valid(
                point_x, point_y, map_array, map_info, min_obstacle_distance
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
