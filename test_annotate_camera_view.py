import cv2
import numpy as np
from datetime import datetime
import os

# Assuming the script is run from the workspace root
from src.primitives.visualization_utils import annotate_camera_view
from src.primitives.projection_utils import (
    angle_distance_to_image_coordinates,
    deg2rad,
)


def main():
    # --- Configuration ---
    # Replace with the actual path to your test image
    image_path = "/Users/axelpeytavin/Projects/innate-repos/innate-cloud-agent/navigation_points/camera_with_points_20250513_010240.jpg"
    output_dir = "test_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Image and Camera Parameters (replace with actual values if needed)
    image_width = 1280
    image_height = 800
    horizontal_fov = 128  # degrees
    vertical_fov = 80  # degrees

    # Sample valid navigation points: (angle_degrees, distance_meters, point_id)
    # Angles are relative to the camera's center view (0 degrees)
    # Positive angle is right, negative angle is left
    valid_navigation_points = [
        (0, 1.0, "a"),  # Straight ahead, 1 meter
        (20, 1.0, "b"),  # 20 degrees right, 1.5 meters
        (40, 1.0, "c"),  # 30 degrees left, 0.8 meters
        (0, 1.5, "d"),  # Straight ahead, 2 meters
        (-40, 1.5, "e"),  # 40 degrees right, 2.5 meters
    ]

    # Convert angles to radians
    valid_navigation_points = [
        (deg2rad(point[0]), point[1], point[2]) for point in valid_navigation_points
    ]

    # --- Image Loading ---
    if not os.path.exists(image_path):
        print(f"Warning: Test image '{image_path}' not found. Creating a blank image.")
        # Create a blank black image as a placeholder
        cv_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    else:
        cv_image = cv2.imread(image_path)
        if cv_image is None:
            print(f"Error: Could not load image from '{image_path}'.")
            return
        # Resize image if it doesn't match parameters
        if cv_image.shape[0] != image_height or cv_image.shape[1] != image_width:
            print(f"Resizing image to {image_width}x{image_height}")
            cv_image = cv2.resize(cv_image, (image_width, image_height))

    # --- Coordinate Conversion Setup ---
    camera_params = {
        "width": image_width,
        "height": image_height,
        "horizontal_fov": horizontal_fov,
        "vertical_fov": vertical_fov,
        "pitch_deg": -10,
        "x_cam": 0.0197,
        "height_cam": 0.19663,
    }

    # Wrapper function matching the expected signature
    def convert_to_image_coords(angle, distance):
        return angle_distance_to_image_coordinates(angle, distance, camera_params)

    # --- Annotation ---
    print("Annotating camera view...")
    annotated_image = annotate_camera_view(
        cv_image.copy(),  # Pass a copy to avoid modifying the original
        valid_navigation_points,
        convert_to_image_coords,
    )

    # --- Saving Output ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(output_dir, f"annotated_view_{timestamp}.jpg")

    try:
        cv2.imwrite(output_filename, annotated_image)
        print(f"Successfully saved annotated image to '{output_filename}'")
    except Exception as e:
        print(f"Error saving image: {e}")


if __name__ == "__main__":
    main()
