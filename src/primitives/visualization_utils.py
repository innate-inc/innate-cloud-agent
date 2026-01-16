import cv2
import numpy as np
from datetime import datetime
import os

# Constants for visualization
COLORS = {
    "in_fov": (0, 255, 0),  # Green
    "invalid": (0, 0, 255),  # Red - Changed from out_fov
    "selected": (255, 0, 0),  # Blue
    "background": (255, 255, 255),  # White
    "robot": (255, 0, 255),  # Magenta
    "text": (0, 0, 0),  # Black
    "line_green": (0, 200, 0),  # Darker Green for the line
}


def draw_navigation_point(
    image, x, y, point_id, circle_radius=10, point_color_key="in_fov"
):
    """Draw a single navigation point with ID on the image."""
    # Ensure coordinates are integers
    x, y = int(x), int(y)

    # Create an overlay for the transparent circle
    overlay = image.copy()

    # Draw a filled circle on the overlay
    cv2.circle(overlay, (x, y), circle_radius, COLORS[point_color_key], -1)

    # Blend the overlay with the original image for transparency
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Draw outline for the circle
    cv2.circle(image, (x, y), circle_radius, COLORS["text"], 2)  # 2 pixel outline

    # Add number text
    font_size = 0.6
    font_thickness = 1
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
        navigation_points: List of (angle, distance, point_id) tuples for valid points
        point_converter: Function that converts (angle, distance) to image coordinates
    """
    annotated_img = image.copy()
    in_view_points = 0  # Renamed from in_view_valid_points
    out_of_view_points = 0

    # Get image dimensions for drawing the out-of-view indicator
    height, width = annotated_img.shape[:2]

    # Draw valid points (renamed from navigation_points to be clear)
    for point_data in navigation_points:
        if len(point_data) == 3:  # Using (angle, distance, point_id) format
            angle, distance, point_id = point_data
        else:  # Backwards compatibility for (angle, distance) format
            angle, distance = point_data
            point_id = 0  # Default point ID

        # IN THE SIM WE HAVE TO INVERT THE ANGLE AND I DONT KNOW IF THIS
        # WILL BE THE SAME FOR THE REAL ROBOT
        angle = -angle
        img_x, img_y = point_converter(angle, distance)

        if img_x is None or img_y is None:
            # Point is outside field of view
            out_of_view_points += 1
            continue

        # Point is within field of view
        in_view_points += 1
        draw_navigation_point(
            annotated_img, img_x, img_y, point_id, point_color_key="in_fov"
        )

    # Print warning if there are out-of-view points
    if out_of_view_points > 0:
        print(
            f"WARNING: {out_of_view_points} navigation points are outside the camera field of view"
        )

    return annotated_img


def annotate_camera_view_with_line(image, navigation_points, point_converter):
    """
    Annotate camera view with a line representing a constant distance.

    Args:
        image: Original camera image
        navigation_points: List of (angle, distance, point_id) tuples
        point_converter: Function that converts (angle, distance) to image coordinates
    """
    annotated_img = image.copy()
    image_points = []
    out_of_view_points = 0

    # Convert all navigation points to image coordinates
    for point_data in navigation_points:
        angle, distance, _ = point_data
        # IN THE SIM WE HAVE TO INVERT THE ANGLE AND I DONT KNOW IF THIS
        # WILL BE THE SAME FOR THE REAL ROBOT
        angle = -angle
        img_x, img_y = point_converter(angle, distance)

        if img_x is not None and img_y is not None:
            image_points.append((img_x, img_y))
        else:
            out_of_view_points += 1

    # Draw the line if there are enough points
    if len(image_points) > 1:
        pts = np.array(image_points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(
            annotated_img,
            [pts],
            isClosed=False,
            color=COLORS["line_green"],
            thickness=3,
        )

    if out_of_view_points > 0:
        print(
            f"WARNING: {out_of_view_points} navigation points for the line are outside the camera field of view"
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
    map_array,
    robot_pos,
    navigation_points,
    map_info,
    scale_factor=4,
    invalid_points=None,
):
    """
    Create a visualization of the map with robot and navigation points.

    Args:
        map_array: 2D numpy array of map data
        robot_pos: (x, y, yaw) tuple in grid coordinates
        navigation_points: List of (x, y, theta, point_id) tuples for valid points in grid coordinates
        map_info: Dictionary with map metadata
        scale_factor: Factor to scale up the final visualization
        invalid_points: Optional list of (x, y, theta, point_id) tuples for invalid points in grid coordinates
    """
    # Create RGB visualization
    vis_map = np.full((map_array.shape[0], map_array.shape[1], 3), [255, 192, 203], dtype=np.uint8)
    vis_map[map_array < 0] = [0, 0, 128]  # Unknown = blue
    
    # For values between 0 and 100, create gradient proportional to the value
    # mask for values that are not <0
    other_values_mask = (map_array >= 0)
    if np.any(other_values_mask):
        normalized = ((100 - map_array[other_values_mask]) / 100.0 * 255).astype(np.uint8)
        vis_map[other_values_mask] = np.stack([normalized, normalized, normalized], axis=-1)
    
    

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

    # Draw valid navigation points on scaled map
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

    # Draw invalid navigation points on scaled map
    if invalid_points:
        for point_data in invalid_points:
            if len(point_data) == 4:  # Using (x, y, theta, point_id) format
                point_x, point_y, point_theta, point_id = point_data
            else:  # Should not happen
                point_x, point_y, point_theta = point_data
                point_id = 0

            # Scale coordinates for the enlarged map
            point_x = int(point_x * scale_factor)
            point_y = int(point_y * scale_factor)

            # Draw point (circles appropriate for the scaled map) - use "invalid" color
            cv2.circle(
                vis_map_large, (point_x, point_y), 8, COLORS["invalid"], -1
            )  # Red color
            cv2.circle(
                vis_map_large, (point_x, point_y), 10, COLORS["text"], 1
            )  # Black outline

            # Draw orientation (lines appropriate for the scaled map) - use "invalid" color
            orientation_length = 20
            endpoint_x = int(point_x + orientation_length * np.cos(point_theta))
            endpoint_y = int(point_y + orientation_length * np.sin(point_theta))
            cv2.line(
                vis_map_large,
                (point_x, point_y),
                (endpoint_x, endpoint_y),
                COLORS["invalid"],  # Red color
                2,
            )

            # Add point ID (text size appropriate for the scaled map)
            cv2.putText(
                vis_map_large,
                str(point_id),
                (point_x + 12, point_y - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                COLORS["text"],
                2,
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
        ("Invalid Points", COLORS["invalid"]),  # Added legend for invalid points
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


def save_navigation_visualizations(camera_image, map_vis, timestamp=None, prefix=""):
    """Save both camera and map visualizations with timestamp."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs("navigation_visualizations", exist_ok=True)

    camera_path = (
        f"navigation_visualizations/{prefix}_camera_with_points_{timestamp}.jpg"
    )
    map_path = f"navigation_visualizations/{prefix}_map_with_points_{timestamp}.jpg"

    cv2.imwrite(camera_path, camera_image)
    cv2.imwrite(map_path, map_vis)

    print(f"Saved camera visualization to {camera_path}")
    print(f"Saved map visualization to {map_path}")

    return camera_path, map_path


def annotate_camera_view_with_orientation_lines(
    image, orientation_angles, point_converter, camera_info
):
    """
    Annotate camera view with vertical lines representing orientation angles.

    Args:
        image: Original camera image
        orientation_angles: List of angles in degrees relative to robot's forward direction
        point_converter: Function that converts (angle, distance) to image coordinates
        camera_info: Camera information dictionary
    """
    annotated_img = image.copy()
    image_height, image_width = annotated_img.shape[:2]

    # Define distances to create vertical lines (from ground to horizon)
    min_distance = 0.5  # Start close to robot
    max_distance = 5.0  # Extend to horizon
    distance_steps = 20  # Number of points to create smooth vertical lines

    for angle_deg in orientation_angles:
        angle_rad = np.deg2rad(angle_deg)
        line_points = []

        # Create points along the vertical line at this angle
        for distance in np.linspace(min_distance, max_distance, distance_steps):
            # IN THE SIM WE HAVE TO INVERT THE ANGLE AND I DONT KNOW IF THIS
            # WILL BE THE SAME FOR THE REAL ROBOT
            img_x, img_y = point_converter(-angle_rad, distance)

            if img_x is not None and img_y is not None:
                line_points.append((img_x, img_y))

        # Draw the vertical line if we have enough points
        if len(line_points) > 1:
            # Sort points by y-coordinate to ensure proper line drawing
            line_points.sort(key=lambda p: p[1])

            # Draw the line
            pts = np.array(line_points, np.int32)
            pts = pts.reshape((-1, 1, 2))

            # Use blue color for orientation lines
            line_color = (255, 0, 0)  # Blue in BGR
            cv2.polylines(
                annotated_img,
                [pts],
                isClosed=False,
                color=line_color,
                thickness=2,
            )

            # Add angle label at the bottom of the line if there are points
            if line_points:
                bottom_point = max(
                    line_points, key=lambda p: p[1]
                )  # Point with highest y value
                label = f"{angle_deg:+.0f}°"

                # Add text background for better visibility
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)

                text_x = max(
                    0,
                    min(
                        image_width - text_size[0], bottom_point[0] - text_size[0] // 2
                    ),
                )
                text_y = min(image_height - 5, bottom_point[1] + 20)

                # Draw background rectangle
                cv2.rectangle(
                    annotated_img,
                    (text_x - 2, text_y - text_size[1] - 2),
                    (text_x + text_size[0] + 2, text_y + 2),
                    (255, 255, 255),  # White background
                    -1,
                )

                # Draw text
                cv2.putText(
                    annotated_img,
                    label,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (0, 0, 0),  # Black text
                    font_thickness,
                )

    return annotated_img


def annotate_camera_view_with_corridors(
    image, horizontal_fov_deg, point_converter, corridor_width, camera_info
):
    """
    Annotate camera view with corridors delimited by vertical lines.
    Each corridor is 20 degrees wide, centered on 0, with mean angle displayed at bottom.

    Args:
        image: Original camera image
        horizontal_fov_deg: Horizontal field of view in degrees
        point_converter: Function that converts (angle, distance) to image coordinates
        camera_info: Camera information dictionary
    """
    annotated_img = image.copy()
    image_height, image_width = annotated_img.shape[:2]

    # Define corridor parameters
    corridor_half_width = corridor_width / 2.0

    # Calculate the range we can cover
    max_angle = horizontal_fov_deg / 2.0

    # Generate corridors centered on 0
    corridors = []

    # Start with center corridor
    center_corridor = (-corridor_half_width, corridor_half_width)
    corridors.append(center_corridor)

    # Add corridors to the left and right alternately
    left_start = -corridor_half_width
    right_start = corridor_half_width

    while True:
        added_corridor = False

        # Try to add corridor to the left
        left_end = left_start - corridor_width
        if left_end >= -max_angle:
            corridors.insert(
                0, (left_end, left_start)
            )  # Insert at beginning to maintain order
            left_start = left_end
            added_corridor = True
        elif left_start > -max_angle:
            # Add partial corridor on the left
            corridors.insert(0, (-max_angle, left_start))
            left_start = -max_angle  # Update to prevent further left attempts
            added_corridor = True

        # Try to add corridor to the right
        right_end = right_start + corridor_width
        if right_end <= max_angle:
            corridors.append((right_start, right_end))
            right_start = right_end
            added_corridor = True
        elif right_start < max_angle:
            # Add partial corridor on the right
            corridors.append((right_start, max_angle))
            right_start = max_angle  # Update to prevent further right attempts
            added_corridor = True

        if not added_corridor:
            break

    print(
        f"Generated {len(corridors)} corridors for hfov={horizontal_fov_deg}°: {corridors}"
    )

    # Define distances to create vertical lines (from ground to horizon)
    min_distance = 0.3  # Start close to robot
    max_distance = 5.0  # Extend to horizon
    distance_steps = 20  # Number of points to create smooth vertical lines

    # Colors for corridors (alternating for better visibility)
    corridor_colors = [
        (255, 0, 0),  # Blue
        (0, 255, 0),  # Green
        (0, 0, 255),  # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]

    # Draw corridor delimiter lines and labels
    for i, (start_angle, end_angle) in enumerate(corridors):
        mean_angle = (start_angle + end_angle) / 2.0
        color = corridor_colors[i % len(corridor_colors)]

        # Draw left boundary line (except for leftmost corridor)
        if i > 0 or start_angle > -max_angle:
            left_line_points = []
            for distance in np.linspace(min_distance, max_distance, distance_steps):
                # IN THE SIM WE HAVE TO INVERT THE ANGLE
                img_x, img_y = point_converter(-np.deg2rad(start_angle), distance)
                if img_x is not None and img_y is not None:
                    left_line_points.append((img_x, img_y))

            if len(left_line_points) > 1:
                left_line_points.sort(key=lambda p: p[1])
                pts = np.array(left_line_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    annotated_img, [pts], isClosed=False, color=color, thickness=2
                )

        # Draw right boundary line (except for rightmost corridor)
        if i < len(corridors) - 1 or end_angle < max_angle:
            right_line_points = []
            for distance in np.linspace(min_distance, max_distance, distance_steps):
                # IN THE SIM WE HAVE TO INVERT THE ANGLE
                img_x, img_y = point_converter(-np.deg2rad(end_angle), distance)
                if img_x is not None and img_y is not None:
                    right_line_points.append((img_x, img_y))

            if len(right_line_points) > 1:
                right_line_points.sort(key=lambda p: p[1])
                pts = np.array(right_line_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    annotated_img, [pts], isClosed=False, color=color, thickness=2
                )

        # Add mean angle label at the bottom of the corridor
        # Find a point at the mean angle to position the label
        label_distance = min_distance + 0.2  # Slightly above ground
        img_x, img_y = point_converter(-np.deg2rad(mean_angle), label_distance)

        if img_x is not None and img_y is not None:
            label = f"{mean_angle}"

            # Add text background for better visibility
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)

            text_x = max(0, min(image_width - text_size[0], img_x - text_size[0] // 2))
            text_y = min(image_height - 10, img_y + 40)  # Position near bottom

            # Draw background rectangle
            cv2.rectangle(
                annotated_img,
                (text_x - 3, text_y - text_size[1] - 3),
                (text_x + text_size[0] + 3, text_y + 3),
                (255, 255, 255),  # White background
                -1,
            )

            # Draw text
            cv2.putText(
                annotated_img,
                label,
                (text_x, text_y),
                font,
                font_scale,
                (0, 0, 0),  # Black text
                font_thickness,
            )

    return annotated_img, corridors
