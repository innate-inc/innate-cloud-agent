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


def draw_navigation_point(image, x, y, point_id, circle_radius=15):
    """Draw a single navigation point with ID on the image."""
    # Ensure coordinates are integers
    x, y = int(x), int(y)

    # Draw black outline circle
    cv2.circle(image, (x, y), circle_radius + 2, COLORS["text"], -1)

    # Draw filled circle
    cv2.circle(image, (x, y), circle_radius, COLORS["in_fov"], -1)

    # Add number text
    font_size = 0.8
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
        font_thickness + 1,
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
        navigation_points: List of (angle, distance, point_id) tuples
        point_converter: Function that converts (angle, distance) to image coordinates
    """
    annotated_img = image.copy()
    in_view_points = 0
    out_of_view_points = 0

    # Get image dimensions for drawing the out-of-view indicator
    height, width = annotated_img.shape[:2]

    # Add a small overlay at the bottom with stats
    info_bar_height = 40
    overlay = annotated_img.copy()
    cv2.rectangle(
        overlay, (0, height - info_bar_height), (width, height), (0, 0, 0), -1
    )
    annotated_img = cv2.addWeighted(overlay, 0.3, annotated_img, 0.7, 0)

    for point_data in navigation_points:
        if len(point_data) == 3:  # Using (angle, distance, point_id) format
            angle, distance, point_id = point_data
        else:  # Backwards compatibility for (angle, distance) format
            angle, distance = point_data
            point_id = 0  # Default point ID

        img_x, img_y = point_converter(angle, distance)

        if img_x is None or img_y is None:
            # Point is outside field of view
            out_of_view_points += 1
            continue

        # Point is within field of view
        in_view_points += 1
        draw_navigation_point(annotated_img, img_x, img_y, point_id)

    # Add text showing how many points are in view vs out of view
    status_text = (
        f"Points in view: {in_view_points}/{in_view_points + out_of_view_points}"
    )
    cv2.putText(
        annotated_img,
        status_text,
        (10, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    # Print warning if there are out-of-view points
    if out_of_view_points > 0:
        print(
            f"WARNING: {out_of_view_points} navigation points are outside the camera field of view"
        )

    return annotated_img


def draw_robot_on_map(vis_map, robot_x, robot_y, robot_yaw, scale=1):
    """Draw robot position and orientation on map."""
    # Ensure coordinates are integers
    robot_x, robot_y = int(robot_x), int(robot_y)

    # Draw robot position
    cv2.circle(vis_map, (robot_x, robot_y), 6 * scale, COLORS["robot"], -1)

    # Draw robot orientation
    orientation_length = 15 * scale
    endpoint_x = int(robot_x + orientation_length * np.cos(robot_yaw))
    endpoint_y = int(robot_y + orientation_length * np.sin(robot_yaw))
    cv2.line(
        vis_map,
        (robot_x, robot_y),
        (endpoint_x, endpoint_y),
        COLORS["robot"],
        2 * scale,
    )

    # Add robot label
    cv2.putText(
        vis_map,
        "Robot",
        (robot_x + 10 * scale, robot_y - 10 * scale),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6 * scale,
        COLORS["robot"],
        max(1, int(scale)),
    )


def create_map_visualization(
    map_array, robot_pos, navigation_points, map_info, scale_factor=4
):
    """
    Create a visualization of the map with robot and navigation points.

    Args:
        map_array: 2D numpy array of map data
        robot_pos: (x, y, yaw) tuple in grid coordinates
        navigation_points: List of (x, y, theta, point_id) tuples in grid coordinates
        map_info: Dictionary with map metadata
        scale_factor: Factor to scale up the final visualization
    """
    # Create RGB visualization
    vis_map = np.zeros((map_array.shape[0], map_array.shape[1], 3), dtype=np.uint8)
    vis_map[map_array == 0] = [255, 255, 255]  # Free space = white
    vis_map[map_array == 100] = [0, 0, 0]  # Obstacles = black
    vis_map[map_array == -1] = [128, 128, 128]  # Unknown = gray

    # Scale up the visualization FIRST
    vis_map_large = cv2.resize(
        vis_map, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST
    )

    # Scale robot and navigation point coordinates based on scale_factor
    scaled_robot_x = int(robot_pos[0] * scale_factor)
    scaled_robot_y = int(robot_pos[1] * scale_factor)
    scaled_robot_yaw = robot_pos[2]  # Angle doesn't change

    # Draw robot on scaled map
    draw_robot_on_map(
        vis_map_large, scaled_robot_x, scaled_robot_y, scaled_robot_yaw, scale=2
    )

    # Draw navigation points on scaled map
    for point_data in navigation_points:
        if len(point_data) == 4:  # Using (x, y, theta, point_id) format
            point_x, point_y, point_theta, point_id = point_data
        else:  # Backwards compatibility for (x, y, theta) format
            point_x, point_y, point_theta = point_data
            point_id = 0  # Default point ID

        # Scale coordinates for the enlarged map
        point_x = int(point_x * scale_factor)
        point_y = int(point_y * scale_factor)

        # Draw point (circles appropriate for the scaled map)
        cv2.circle(vis_map_large, (point_x, point_y), 8, COLORS["in_fov"], -1)
        cv2.circle(vis_map_large, (point_x, point_y), 10, COLORS["text"], 1)

        # Draw orientation (lines appropriate for the scaled map)
        orientation_length = 20
        endpoint_x = int(point_x + orientation_length * np.cos(point_theta))
        endpoint_y = int(point_y + orientation_length * np.sin(point_theta))
        cv2.line(
            vis_map_large,
            (point_x, point_y),
            (endpoint_x, endpoint_y),
            COLORS["in_fov"],
            2,
        )

        # Add point ID (text size appropriate for the scaled map)
        cv2.putText(
            vis_map_large,
            str(point_id),
            (point_x + 12, point_y - 12),  # Offset label to prevent overlap with point
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,  # Larger font size for better visibility
            COLORS["text"],
            2,  # Thicker text
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
