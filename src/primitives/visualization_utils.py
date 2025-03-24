import cv2
import numpy as np
from datetime import datetime
import os

# Constants for visualization
COLORS = {
    "in_fov": (0, 255, 0),  # Green
    "out_fov": (0, 0, 255),  # Red
    "selected": (255, 0, 0),  # Blue
    "background": (255, 255, 255),  # White
    "robot": (255, 0, 255),  # Magenta
    "text": (0, 0, 0),  # Black
}


def draw_navigation_point(image, x, y, point_id, circle_radius=20):
    """Draw a single navigation point with ID on the image."""
    # Ensure coordinates are integers
    x, y = int(x), int(y)

    # Draw black outline circle
    cv2.circle(image, (x, y), circle_radius + 2, COLORS["text"], -1)

    # Draw filled circle
    cv2.circle(image, (x, y), circle_radius, COLORS["in_fov"], -1)

    # Add number text
    font_size = 1.0
    font_thickness = 2
    text = str(point_id)
    text_size, _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thickness
    )
    text_x = int(x - text_size[0] / 2)
    text_y = int(y + text_size[1] / 2)

    # Draw number with outline
    cv2.putText(
        image,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        COLORS["text"],
        font_thickness + 2,
    )
    cv2.putText(
        image,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        COLORS["background"],
        font_thickness,
    )


def annotate_camera_view(image, navigation_points, point_converter):
    """
    Annotate camera view with navigation points.

    Args:
        image: Original camera image
        navigation_points: List of (angle, distance) tuples
        point_converter: Function that converts (angle, distance) to image coordinates
    """
    annotated_img = image.copy()

    for i, (angle, distance) in enumerate(navigation_points):
        img_x, img_y = point_converter(angle, distance)
        draw_navigation_point(annotated_img, img_x, img_y, i + 1)

    return annotated_img


def draw_robot_on_map(vis_map, robot_x, robot_y, robot_yaw, scale=1):
    """Draw robot position and orientation on map."""
    # Ensure coordinates are integers
    robot_x, robot_y = int(robot_x), int(robot_y)

    # Draw robot position
    cv2.circle(vis_map, (robot_x, robot_y), 8 * scale, COLORS["robot"], -1)

    # Draw robot orientation
    orientation_length = 20 * scale
    endpoint_x = int(robot_x + orientation_length * np.cos(robot_yaw))
    endpoint_y = int(robot_y + orientation_length * np.sin(robot_yaw))
    cv2.line(
        vis_map,
        (robot_x, robot_y),
        (endpoint_x, endpoint_y),
        COLORS["robot"],
        3 * scale,
    )

    # Add robot label
    cv2.putText(
        vis_map,
        "Robot",
        (robot_x + 10 * scale, robot_y - 10 * scale),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5 * scale,
        COLORS["robot"],
        scale,
    )


def create_map_visualization(
    map_array, robot_pos, navigation_points, map_info, scale_factor=2
):
    """
    Create a visualization of the map with robot and navigation points.

    Args:
        map_array: 2D numpy array of map data
        robot_pos: (x, y, yaw) tuple in grid coordinates
        navigation_points: List of (x, y, theta) tuples in grid coordinates
        map_info: Dictionary with map metadata
        scale_factor: Factor to scale up the final visualization
    """
    # Create RGB visualization
    vis_map = np.zeros((map_array.shape[0], map_array.shape[1], 3), dtype=np.uint8)
    vis_map[map_array == 0] = [255, 255, 255]  # Free space = white
    vis_map[map_array == 100] = [0, 0, 0]  # Obstacles = black
    vis_map[map_array == -1] = [128, 128, 128]  # Unknown = gray

    # Draw robot
    robot_x, robot_y, robot_yaw = robot_pos
    draw_robot_on_map(vis_map, robot_x, robot_y, robot_yaw)

    # Draw navigation points
    for i, (point_x, point_y, point_theta) in enumerate(navigation_points):
        # Ensure coordinates are integers
        point_x, point_y = int(point_x), int(point_y)

        # Draw point
        cv2.circle(vis_map, (point_x, point_y), 6, COLORS["in_fov"], -1)
        cv2.circle(vis_map, (point_x, point_y), 7, COLORS["text"], 1)

        # Draw orientation
        orientation_length = 15
        endpoint_x = int(point_x + orientation_length * np.cos(point_theta))
        endpoint_y = int(point_y + orientation_length * np.sin(point_theta))
        cv2.line(
            vis_map, (point_x, point_y), (endpoint_x, endpoint_y), COLORS["in_fov"], 2
        )

        # Add point ID
        cv2.putText(
            vis_map,
            str(i + 1),
            (point_x + 7, point_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            COLORS["text"],
            2,
        )

    # Scale up the visualization
    vis_map_large = cv2.resize(
        vis_map, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST
    )

    # Add border and title
    border_size = 50
    map_with_border = (
        np.ones(
            (
                vis_map_large.shape[0] + 2 * border_size,
                vis_map_large.shape[1] + 2 * border_size,
                3,
            ),
            dtype=np.uint8,
        )
        * 200
    )

    map_with_border[
        border_size : border_size + vis_map_large.shape[0],
        border_size : border_size + vis_map_large.shape[1],
    ] = vis_map_large

    # Add title
    title = f"Navigation Map - Robot at ({robot_pos[0]:.2f}, {robot_pos[1]:.2f})"
    cv2.putText(
        map_with_border,
        title,
        (border_size, int(border_size / 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        COLORS["text"],
        2,
    )

    # Add legend
    y_pos = border_size + vis_map_large.shape[0] + 15
    legend_items = [
        ("Robot", COLORS["robot"]),
        ("Navigation Points", COLORS["in_fov"]),
        ("Obstacles", COLORS["text"]),
    ]

    x_offset = border_size
    for text, color in legend_items:
        cv2.circle(map_with_border, (x_offset + 15, y_pos), 6, color, -1)
        cv2.putText(
            map_with_border,
            text,
            (x_offset + 30, y_pos + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLORS["text"],
            1,
        )
        x_offset += 150

    # Add min obstacle distance info
    min_obstacle_distance = map_info.get("min_obstacle_distance", 0.25)
    cv2.putText(
        map_with_border,
        f"Min obstacle distance: {min_obstacle_distance} m",
        (x_offset + 30, y_pos + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        COLORS["text"],
        1,
    )

    return map_with_border


def save_navigation_visualizations(camera_image, map_vis, timestamp=None):
    """Save both camera and map visualizations with timestamp."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs("navigation_points", exist_ok=True)

    camera_path = f"navigation_points/camera_with_points_{timestamp}.jpg"
    map_path = f"navigation_points/map_with_points_{timestamp}.jpg"

    cv2.imwrite(camera_path, camera_image)
    cv2.imwrite(map_path, map_vis)

    print(f"Saved camera visualization to {camera_path}")
    print(f"Saved map visualization to {map_path}")

    return camera_path, map_path
