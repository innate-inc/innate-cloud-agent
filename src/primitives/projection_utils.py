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
        camera_info (dict): Camera information with keys "width", "height", "horizontal_fov", "vertical_fov"
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
        min_distance (float): Minimum distance from robot to sample points (meters)
        max_distance (float): Maximum distance from robot to sample points (meters)
        min_obstacle_distance (float): Minimum allowable distance from obstacles
            (meters)
        num_samples (int): Number of points to sample
        visualize (bool): Whether to visualize the sampling process

    Returns:
        tuple: (
            valid_points_absolute,
            valid_points_angle_distance,
            invalid_points_absolute,
            invalid_points_angle_distance
        )
            - valid_points_absolute: List of (x,y,theta) world coords
            - valid_points_angle_distance: List of (angle,distance) relative
            - invalid_points_absolute: List of (x,y,theta) world coords (invalid)
            - invalid_points_angle_distance: List of (angle,distance) relative (invalid)
    """
    # Get map metadata
    resolution = map_info["resolution"]
    origin_x = map_info["origin_x"]
    origin_y = map_info["origin_y"]

    # Sample points in a sector in front of the robot
    valid_points_absolute = []
    valid_points_angle_distance = []
    filtered_points = 0  # Count points filtered due to FOV constraints

    # Calculate obstacle radius in grid cells
    obstacle_radius = int(min_obstacle_distance / resolution)

    angles = [deg2rad(angle) for angle in angles_deg]

    # For each combination of angle and distance
    invalid_points_absolute = []
    invalid_points_angle_distance = []

    for angle in angles:
        for distance in distances:
            # Calculate world coordinates
            angle_rel = -(
                angle - current_yaw
            )  # Get angle relative to current orientation
            point_x = current_x + distance * np.cos(angle_rel)
            point_y = current_y + distance * np.sin(angle_rel)
            point_absolute = (point_x, point_y, angle_rel)
            point_angle_distance = (angle, distance)

            # Check if the point would be visible in the camera FOV
            if not is_point_in_fov(angle, distance, h_fov_deg):
                filtered_points += 1
                invalid_points_absolute.append(point_absolute)
                invalid_points_angle_distance.append(point_angle_distance)
                continue

            # Convert to grid coordinates
            grid_x = int((point_x - origin_x) / resolution)
            grid_y = int((point_y - origin_y) / resolution)

            # Skip if outside map bounds
            if (
                grid_x < 0
                or grid_x >= map_info["width"]
                or grid_y < 0
                or grid_y >= map_info["height"]
            ):
                invalid_points_absolute.append(point_absolute)
                invalid_points_angle_distance.append(point_angle_distance)
                continue

            # Check if point is a valid navigable location (away from obstacles)
            is_valid = True

            # Define a search window around the point
            min_x = max(0, grid_x - obstacle_radius)
            max_x = min(map_info["width"] - 1, grid_x + obstacle_radius)
            min_y = max(0, grid_y - obstacle_radius)
            max_y = min(map_info["height"] - 1, grid_y + obstacle_radius)

            # Check if any cell within the radius is an obstacle (value = 100)
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    # Calculate distance from this cell to the target cell
                    cell_distance = math.sqrt((x - grid_x) ** 2 + (y - grid_y) ** 2)

                    # If this cell is within our search radius and is an obstacle
                    if cell_distance <= obstacle_radius and map_array[y, x] == 100:
                        is_valid = False
                        break

                if not is_valid:
                    break

            # Add the point if it's valid
            if is_valid:
                # The navigation target will face in the direction from robot to point
                target_theta = angle_rel
                valid_points_absolute.append((point_x, point_y, target_theta))
                valid_points_angle_distance.append((angle, distance))
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
