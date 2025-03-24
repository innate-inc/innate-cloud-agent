import numpy as np
import math

# Constants for the camera
HORIZONTAL_FOV = 96.4  # Camera horizontal field of view in degrees
VERT_FOV = 80.0  # Camera vertical field of view in degrees
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


def deg2rad(deg):
    """Convert degrees to radians."""
    return deg * np.pi / 180


def rad2deg(rad):
    """Convert radians to degrees."""
    return rad * 180 / np.pi


def compute_intrinsics(width, height, h_fov_deg, v_fov_deg):
    """
    Compute camera intrinsic matrix from field of view.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        h_fov_deg: Horizontal field of view in degrees
        v_fov_deg: Vertical field of view in degrees

    Returns:
        K: 3x3 camera intrinsic matrix
    """
    h_fov = deg2rad(h_fov_deg)
    v_fov = deg2rad(v_fov_deg)
    f_x = (width / 2) / np.tan(h_fov / 2)
    f_y = (height / 2) / np.tan(v_fov / 2)
    c_x = width / 2
    c_y = height / 2
    K = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])
    return K


def compute_homography(K, h, pitch_deg):
    """
    Compute homography matrix for ground plane projection.

    Args:
        K: Camera intrinsic matrix
        h: Camera height in centimeters
        pitch_deg: Camera pitch angle in degrees (90° is looking straight forward)

    Returns:
        H: 3x3 homography matrix
    """
    # Convert pitch to radians. We define theta = -pitch so that if pitch>0
    # (camera looks downward) then theta is negative.
    pitch = deg2rad(pitch_deg)
    theta = -pitch

    # Rotation matrix about the x-axis:
    R = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )

    # Extract columns of R:
    r1 = R[:, 0]  # [1, 0, 0]
    r2 = R[:, 1]  # [0, cos(theta), sin(theta)]
    r3 = R[:, 2]  # [0, -sin(theta), cos(theta)]

    # For ground points (world z=0), the projection is:
    # p ~ K * [ r1  r2  -h*r3 ] * [x, y, 1]^T.
    M = np.column_stack((r1, r2, -h * r3))
    H = K @ M
    return H


def project_ground_point(H, D, gamma_deg):
    """
    Project ground point to image coordinates.

    Args:
        H: Homography matrix
        D: Distance from camera in centimeters
        gamma_deg: Angle in degrees from camera forward direction

    Returns:
        (u, v): Image coordinates in pixels
    """
    # Convert gamma from degrees to radians
    gamma = deg2rad(gamma_deg)

    # Ground point (with origin at camera's vertical projection on the ground)
    x = D * np.sin(gamma)
    y = D * np.cos(gamma)
    ground_pt = np.array([x, y, 1])

    # Project point
    p = H @ ground_pt
    u = p[0] / p[2]
    v = p[1] / p[2]
    return u, v


def is_point_in_fov(angle, distance, h_fov_deg=HORIZONTAL_FOV, v_fov_deg=VERT_FOV):
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


def angle_distance_to_image_coordinates(angle, distance, h_cam=20, pitch_deg=90):
    """
    Convert angle and distance to image coordinates using proper camera projection.

    Args:
        angle (float): Angle in radians relative to robot's forward direction
        distance (float): Distance in meters
        h_cam (float): Camera height in centimeters
        pitch_deg (float): Camera pitch angle in degrees (90° is looking forward)

    Returns:
        tuple: (x, y) coordinates in the image, or (None, None) if out of field of view
    """
    # Check if the point is within the field of view first
    if not is_point_in_fov(angle, distance):
        return (None, None)

    # Camera parameters
    width = IMAGE_WIDTH
    height = IMAGE_HEIGHT
    h_fov_deg = HORIZONTAL_FOV
    v_fov_deg = VERT_FOV

    # Convert distance to centimeters
    distance_cm = distance * 100

    # Convert angle to degrees
    angle_deg = rad2deg(angle)

    # Compute camera intrinsics and homography
    K = compute_intrinsics(width, height, h_fov_deg, v_fov_deg)
    H = compute_homography(K, h_cam, pitch_deg)

    # Project the point
    u, v = project_ground_point(H, distance_cm, angle_deg)

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
    distances=[0.5, 1.5],
    angles_deg=[-30, 0, 30],
    min_obstacle_distance=0.50,
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
        tuple: (valid_points_absolute, valid_points_angle_distance)
            - valid_points_absolute: List of (x, y, theta) tuples in world coordinates
            - valid_points_angle_distance: List of (angle, distance) tuples relative
              to robot
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

    # Check if we're trying to sample at distances or angles that are incompatible with the FOV
    h_fov_rad = deg2rad(HORIZONTAL_FOV)

    for angle in angles:
        if abs(angle) > h_fov_rad:
            print(
                f"WARNING: Sampling angle ({angle:.2f} rad) exceeds camera FOV ({h_fov_rad:.2f} rad)"
            )
            print("Some points may be outside the camera view")

    # For each combination of angle and distance
    for angle in angles:
        for distance in distances:
            # Check if the point would be visible in the camera FOV
            if not is_point_in_fov(angle, distance):
                filtered_points += 1
                continue

            # Calculate world coordinates
            point_x = current_x + distance * np.cos(angle)
            point_y = current_y + distance * np.sin(angle)

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
                target_theta = angle
                valid_points_absolute.append((point_x, point_y, target_theta))
                valid_points_angle_distance.append((angle, distance))

    return valid_points_absolute, valid_points_angle_distance


def world_to_grid_coordinates(x, y, map_info):
    """Convert world coordinates to grid coordinates."""
    pixel_x = int((x - map_info["origin_x"]) / map_info["resolution"])
    pixel_y = int((y - map_info["origin_y"]) / map_info["resolution"])
    return pixel_x, pixel_y
