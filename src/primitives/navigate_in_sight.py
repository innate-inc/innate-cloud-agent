from datetime import datetime
import traceback
from src.primitives.types import Primitive
import asyncio
import base64
import cv2
import numpy as np
from ultralytics import YOLO, SAM
import google.generativeai as genai
import os
import math


# Import our Groq helpers (assumed to be defined elsewhere)
# from orchestrator.agent.groq_instant_use import query_groq_classifier, query_groq_vlm

# Utility to decode depth payload (assumed defined in src/utils.py)
from src.utils import decode_depth_payload, decode_map_payload


class NavigateInSight(Primitive):
    def __init__(self):
        """
        Load the YOLO and SAM models for segmentation.
        Using the ultralytics implementations, these models are loaded on instantiation.
        """

        # YOLO model for object detection. Adjust the weights file as needed.
        self.yolo_model = YOLO("yolov8m-worldv2.pt")
        # SAM model for segmentation. Adjust the weights file as needed.
        self.sam_model = SAM("sam2_t.pt")

    @property
    def name(self):
        return "navigate_in_sight"

    def guidelines(self):
        return "To use to navigate to an object or target in sight. Is a much better primitive than navigate_to_position to use when it's to navigate to a target in sight. Provide a target object name, such as 'shelf', 'table', 'chair', etc."

    def update_current_vars(
        self,
        current_x: float,
        current_y: float,
        current_yaw: float,
        image_b64: str,
        depth_payload: dict,
        horizontal_fov: float = 60.0,
        vertical_fov: float = 40.0,
    ):
        self.current_x = current_x
        self.current_y = current_y
        self.current_yaw = current_yaw
        self.image_b64 = image_b64
        self.depth_payload = depth_payload
        self.horizontal_fov = horizontal_fov
        self.vertical_fov = vertical_fov

    async def execute(
        self,
        target_object: str = None,
        target_description: str = None,
        map_payload: dict = None,
        use_point_selection: bool = False,
    ):
        """
        Execute the navigate_in_sight primitive.

        Args:
            target_object (str, optional): The target object to navigate to
            target_description (str, optional): Description of where to navigate
            map_payload (dict, optional): Map payload from the robot
            use_point_selection (bool): Whether to use the point selection approach

        Returns:
            tuple: (message, success, navigation_command)
        """
        # Use the point selection approach if requested and we have a map
        if use_point_selection and map_payload:
            return await self.execute_with_point_selection(
                target_description or target_object, map_payload
            )

        # Otherwise, use the original object-based approach
        print(
            f"NavigateInSight: Starting navigation from ({self.current_x}, {self.current_y}) towards '{target_object}'"
        )

        # Decode the provided image from base64 into a cv2 image.
        try:
            image_bytes = base64.b64decode(self.image_b64)
            image_array = np.frombuffer(image_bytes, np.uint8)
            cv_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if cv_image is None:
                print("Failed to decode image into cv_image.")
                return "Failed to decode image", False, None
        except Exception as e:
            print(f"Exception decoding image: {e}")
            return "Image decode error", False, None

        # Process the depth payload.
        # Assume depth_data is a JSON string with keys like "height", "width", "encoding", "data", [and optionally "is_bigendian"].
        depth_array = None
        try:
            depth_array = decode_depth_payload(self.depth_payload)
        except Exception as e:
            print(
                f"Failed to decode depth payload: {e}. Traceback: {traceback.format_exc()}"
            )

        # Refine the target object using a Groq classifier.
        # For now we don't do that.
        # user_prompt = f"Extract from the following command the most appropriate name for the object to be segmented: '{target_object}'"
        # system_prompt = (
        #     "You are an AI assistant refining object names for image segmentation. Provide a short, clear, "
        #     "and specific name for the object desired. RETURN A JSON OBJECT with keys 'name_to_segment' and 'reasoning'."
        # )
        # try:
        #     refined_response = query_groq_classifier(
        #         user_prompt, system_prompt, "in_sight_navigator"
        #     )
        #     refined_json = json.loads(refined_response)
        #     refined_target = refined_json.get("name_to_segment", target_object)
        # except Exception as e:
        #     print(f"Error refining target object: {e}")
        #     refined_target = target_object

        # print(f"Refined target object: {refined_target}")

        refined_target = target_object

        # Attempt segmentation using YOLO + SAM.
        segmentation_masks = self.attempt_segmentation(
            cv_image,
            refined_target,
            depth_array,
            self.horizontal_fov,
            self.vertical_fov,
        )
        if segmentation_masks is None or len(segmentation_masks) == 0:
            print(
                f"\033[33mSegmentation failed for target object '{refined_target}'.\033[0m"
            )
            # Save image with segmentation masks.
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(
                f"segmentation_failed_{refined_target}_{timestamp}.jpg", cv_image
            )
            return "Segmentation failed", False, None

        # Select the segmentation mask with highest confidence.
        target_info = segmentation_masks.get("1")
        if not target_info:
            print(
                f"No valid segmentation mask found for target object '{refined_target}'."
            )
            # Save image with segmentation masks.
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(
                f"segmentation_failed_{refined_target}_{timestamp}.jpg", cv_image
            )
            return "Segmentation failed", False, None

        print(f"Segmentation result: {target_info}")

        # --------------------------------------------------------------------
        # Compute navigation setpoint using the segmentation result.
        #
        # For simplicity, assume:
        #   - The current orientation (yaw) is 0 (facing east).
        #   - next_yaw is adjusted by the horizontal angle from the segmentation.
        #
        # This calculation resembles the original logic:
        #   target_x = current_x + (corrected_distance * cos(next_yaw))
        #   target_y = current_y + (corrected_distance * sin(next_yaw))
        # --------------------------------------------------------------------
        next_yaw = self.current_yaw - np.radians(target_info["angle_horizontal"])

        max_travelling_distance = 2.0  # meters
        closest_distance = 0.2  # meters

        depth_of_target = target_info["depth"]
        if depth_of_target == 0:
            depth_of_target = target_info["depth_percentiles"].get("75th", 1.0)
            if depth_of_target == 0:
                depth_of_target = target_info["depth_percentiles"].get("90th", 1.0)

        corrected_distance = min(
            max_travelling_distance,
            depth_of_target * np.cos(np.radians(target_info["angle_vertical"])),
        )
        corrected_distance = max(0, corrected_distance - closest_distance)

        target_x = self.current_x + corrected_distance * np.cos(next_yaw)
        target_y = self.current_y + corrected_distance * np.sin(next_yaw)

        navigation_command = {
            "x": target_x,
            "y": target_y,
            "theta": next_yaw,
        }
        print(f"Computed navigation command: {navigation_command}")

        # Simulate delay for navigation initiation.
        await asyncio.sleep(1)
        return (
            f"Navigation towards {refined_target} initiated with command: {navigation_command}",
            True,
            navigation_command,
        )

    def attempt_segmentation(
        self,
        cv_image,
        target_class,
        depth_array,
        horizontal_fov: float = 60.0,
        vertical_fov: float = 40.0,
    ):
        """
        Use YOLO to detect the target object and SAM to segment the image.
        Then compute spatial information based on the segmentation mask and depth data.
        """
        # Create a copy of the image for visualization
        vis_image = cv_image.copy()

        # Set YOLO to detect only the target class.
        try:
            self.yolo_model.set_classes([target_class])
        except Exception as e:
            print(f"Failed to set YOLO classes: {e}")

        # Run YOLO on the image.
        yolo_results = self.yolo_model(cv_image, conf=0.01)
        if (
            len(yolo_results) == 0
            or not hasattr(yolo_results[0], "boxes")
            or len(yolo_results[0].boxes) == 0
        ):
            print(f"No detections found via YOLO for target class '{target_class}'.")
            return {}

        image_height, image_width = cv_image.shape[:2]
        sam_bboxes = []
        # Draw YOLO bounding boxes
        for bbox in yolo_results[0].boxes.xyxyn:
            x1 = int(bbox[0] * image_width)
            y1 = int(bbox[1] * image_height)
            x2 = int(bbox[2] * image_width)
            y2 = int(bbox[3] * image_height)
            sam_bboxes.append([x1, y1, x2, y2])
            # Draw rectangle for YOLO detection
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Get confidences from YOLO detections.
        confidences = list(yolo_results[0].boxes.conf.cpu().numpy())

        # Run SAM on the input image with the computed bounding boxes.
        sam_results = self.sam_model(cv_image, bboxes=sam_bboxes)
        if not sam_results or not hasattr(sam_results[0], "masks"):
            print("SAM segmentation failed.")
            return {}

        segmentation_masks = {}

        # Resize the image to match depth dimensions if needed.
        if depth_array is not None:
            depth_height, depth_width = depth_array.shape[:2]
        else:
            depth_height, depth_width = image_height, image_width

        if (cv_image.shape[0] != depth_height) or (cv_image.shape[1] != depth_width):
            cv_image_resized = cv2.resize(cv_image, (depth_width, depth_height))
        else:
            cv_image_resized = cv_image.copy()

        # Create a separate image for all segmentation masks
        all_masks = np.zeros_like(cv_image)

        # Iterate over each segmentation mask
        for i, (mask_tensor, conf) in enumerate(
            zip(sam_results[0].masks.data, confidences)
        ):
            try:
                mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
            except Exception as e:
                print(f"Error processing SAM mask tensor: {e}")
                continue

            # Resize the mask if needed
            if mask_np.shape[:2] != (depth_height, depth_width):
                mask_np_resized = cv2.resize(
                    mask_np,
                    (depth_width, depth_height),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                mask_np_resized = mask_np

            # Add colored mask to visualization
            color = np.random.randint(0, 255, 3).tolist()
            mask_colored = np.zeros_like(cv_image)
            mask_colored[mask_np_resized > 0] = color
            all_masks = cv2.addWeighted(all_masks, 1.0, mask_colored, 0.5, 0)

            # Find centroid and draw it
            contours, _ = cv2.findContours(
                mask_np_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours and len(contours) > 0:
                M = cv2.moments(contours[0])
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                    # Draw centroid
                    cv2.circle(vis_image, (center_x, center_y), 5, (0, 0, 255), -1)
                    # Draw mask ID
                    cv2.putText(
                        vis_image,
                        f"Mask {i+1}",
                        (center_x + 10, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )
                else:
                    print("No contours found for mask.")
                    continue
            else:
                print("No contours found for mask.")
                continue

            # Extract depth values corresponding to the mask.
            if depth_array is not None:
                mask_depth_values = depth_array[mask_np_resized > 0]
                if mask_depth_values.size > 0:
                    median_depth = float(np.median(mask_depth_values))
                    p10 = float(np.percentile(mask_depth_values, 10))
                    p25 = float(np.percentile(mask_depth_values, 25))
                    p75 = float(np.percentile(mask_depth_values, 75))
                    p90 = float(np.percentile(mask_depth_values, 90))
                else:
                    median_depth = 1.0
                    p10 = p25 = p75 = p90 = 1.0
            else:
                median_depth = 1.0
                p10 = p25 = p75 = p90 = 1.0

            # Calculate the horizontal and vertical angles based on the mask centroid.
            angle_horizontal, angle_vertical = self._calculate_angles(
                center_x,
                center_y,
                (depth_height, depth_width),
                horizontal_fov,
                vertical_fov,
            )

            segmentation_masks[str(i + 1)] = {
                "angle_horizontal": angle_horizontal,
                "angle_vertical": angle_vertical,
                "depth": median_depth,
                "depth_percentiles": {
                    "10th": p10,
                    "25th": p25,
                    "75th": p75,
                    "90th": p90,
                },
                "confidence": float(conf),
            }

        # Combine original image with masks
        final_vis = cv2.addWeighted(vis_image, 0.7, all_masks, 0.3, 0)

        # Save the visualization
        cv2.imwrite(f"segmentation_result.jpg", final_vis)
        print("Saved visualization to segmentation_result.jpg")

        return segmentation_masks

    def _calculate_angles(
        self,
        center_x,
        center_y,
        image_shape,
        horizontal_fov: float = 60.0,
        vertical_fov: float = 40.0,
    ):
        """
        Compute horizontal and vertical angles from the camera center to the given point.

        Args:
            center_x (int): x-coordinate of the point (in pixels).
            center_y (int): y-coordinate of the point (in pixels).
            image_shape (tuple): The (height, width) of the image.
            horizontal_fov (float): Horizontal field of view in degrees
            vertical_fov (float): Vertical field of view in degrees

        Returns:
            tuple: (horizontal_angle, vertical_angle) in degrees.
        """
        height, width = image_shape
        center_img_x = width / 2.0
        center_img_y = height / 2.0

        # Compute differences between the point and image center.
        dx = center_x - center_img_x
        dy = center_y - center_img_y

        # Normalize the offsets.
        x_norm = dx / (width / 2.0)
        y_norm = dy / (height / 2.0)

        # Use the provided FOV values
        horizontal_angle = x_norm * (horizontal_fov / 2.0)
        vertical_angle = y_norm * (vertical_fov / 2.0)
        return horizontal_angle, vertical_angle

    def sample_valid_navigation_points(
        self,
        current_x,
        current_y,
        current_yaw,
        map_array,
        map_info,
        min_distance=0.5,
        max_distance=2.5,
        min_obstacle_distance=0.25,
        num_samples=8,
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
            min_obstacle_distance (float): Minimum allowable distance from obstacles (meters)
            num_samples (int): Number of points to sample

        Returns:
            list: List of valid navigation points as (x, y, theta) tuples in world coordinates
        """
        import numpy as np
        import math

        # Get map metadata
        resolution = map_info["resolution"]
        origin_x = map_info["origin_x"]
        origin_y = map_info["origin_y"]

        # Sample points in a sector in front of the robot
        valid_points = []

        # Calculate obstacle radius in grid cells
        obstacle_radius = int(min_obstacle_distance / resolution)

        # Sample angles in front of the robot (±60 degrees from current orientation)
        angle_range = 120 * (np.pi / 180)  # 120 degrees in radians

        # Distribute angles with more density in the middle of the view
        half_samples = num_samples // 2

        # Create two sets of angles: one focused in the center, one spread out
        center_angles = np.linspace(
            current_yaw - angle_range / 4, current_yaw + angle_range / 4, half_samples
        )

        wide_angles = np.linspace(
            current_yaw - angle_range / 2, current_yaw + angle_range / 2, half_samples
        )

        # Combine both sets
        angles = np.concatenate([center_angles, wide_angles])

        # Sample distances
        distances = np.linspace(min_distance, max_distance, 3)  # 3 distances per angle

        # For each combination of angle and distance
        for angle in angles:
            for distance in distances:
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
                    valid_points.append((point_x, point_y, target_theta))

                    # Limit the total number of points
                    if len(valid_points) >= num_samples:
                        return valid_points

        return valid_points

    def annotate_navigation_points(self, image, navigation_points, world_to_image_func):
        """
        Annotate an image with navigation points.

        Args:
            image (numpy.ndarray): The image to annotate
            navigation_points (list): List of (x, y, theta) tuples in world coordinates
            world_to_image_func (callable): Function to convert world coordinates to image coordinates

        Returns:
            numpy.ndarray: The annotated image
            dict: Mapping from point IDs to world coordinates
        """
        # Create a copy of the image for annotation
        annotated_img = image.copy()

        # Create a mapping from point IDs to world coordinates
        point_mapping = {}

        for i, (point_x, point_y, point_theta) in enumerate(navigation_points):
            # Convert world coordinates to image coordinates
            img_x, img_y = world_to_image_func(point_x, point_y)

            # Skip points outside the image
            if (
                img_x < 0
                or img_x >= image.shape[1]
                or img_y < 0
                or img_y >= image.shape[0]
            ):
                continue

            # Generate point ID (1-based)
            point_id = i + 1

            # Store point in mapping
            point_mapping[str(point_id)] = {
                "x": point_x,
                "y": point_y,
                "theta": point_theta,
            }

            # Draw point with number - larger and more visible
            # First draw a thick black circle for contrast
            circle_radius = 20
            cv2.circle(
                annotated_img,
                (int(img_x), int(img_y)),
                circle_radius + 2,
                (0, 0, 0),
                -1,
            )

            # Then draw the green circle
            cv2.circle(
                annotated_img, (int(img_x), int(img_y)), circle_radius, (0, 255, 0), -1
            )

            # Add number text with better visibility
            font_size = 1.0
            font_thickness = 2
            text = str(point_id)
            text_size, _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thickness
            )
            text_x = int(img_x - text_size[0] / 2)
            text_y = int(img_y + text_size[1] / 2)

            # Draw text with black outline for better visibility
            cv2.putText(
                annotated_img,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                (0, 0, 0),
                font_thickness + 2,
            )

            cv2.putText(
                annotated_img,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                (255, 255, 255),
                font_thickness,
            )

        # Add a title with the target description
        cv2.putText(
            annotated_img,
            "Navigation Points (Camera View)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )

        return annotated_img, point_mapping

    def world_to_image_coordinates(
        self,
        world_x,
        world_y,
        depth_array,
        current_x,
        current_y,
        current_yaw,
        horizontal_fov,
        vertical_fov,
    ):
        """
        Convert world coordinates to image coordinates.

        Args:
            world_x, world_y: World coordinates to convert
            depth_array: The depth array from the camera
            current_x, current_y, current_yaw: Robot position and orientation
            horizontal_fov, vertical_fov: Camera field of view in degrees

        Returns:
            tuple: (image_x, image_y) coordinates or (-1, -1) if not in view
        """
        # Get image dimensions
        if depth_array is not None:
            height, width = depth_array.shape[:2]
        else:
            # Default fallback dimensions
            height, width = 480, 640

        # Calculate relative position from robot
        dx = world_x - current_x
        dy = world_y - current_y

        # Calculate distance and angle from robot to point
        distance = math.sqrt(dx * dx + dy * dy)
        point_angle = math.atan2(dy, dx)

        # Calculate angle relative to robot's heading
        relative_angle = point_angle - current_yaw

        # Normalize to [-pi, pi]
        while relative_angle > math.pi:
            relative_angle -= 2 * math.pi
        while relative_angle < -math.pi:
            relative_angle += 2 * math.pi

        # Check if point is in the field of view
        if abs(relative_angle) > math.radians(horizontal_fov / 2):
            return -1, -1  # Point not in view

        # Calculate image coordinates
        # X coordinate: map from [-hfov/2, hfov/2] to [0, width]
        img_x = (
            (relative_angle + math.radians(horizontal_fov / 2))
            / math.radians(horizontal_fov)
            * width
        )

        # Rough approximation for Y coordinate based on distance
        # Further objects appear higher in image
        vertical_angle_range = math.radians(vertical_fov)
        # Adjust this factor as needed based on testing
        distance_factor = 0.8
        img_y = height * (1 - distance_factor * distance / 5.0)

        return img_x, img_y

    def visualize_map_with_navigation_points(
        self, map_array, map_info, current_x, current_y, current_yaw, navigation_points
    ):
        """
        Create a visualization of the map with navigation points.

        Args:
            map_array (np.ndarray): Map occupancy grid data
            map_info (dict): Map metadata
            current_x, current_y, current_yaw: Robot position and orientation
            navigation_points (list): List of (x, y, theta) tuples in world coordinates

        Returns:
            np.ndarray: The map visualization image
        """
        import cv2
        import numpy as np
        import math

        # Create a colored map visualization (grayscale to RGB)
        # 0 = free space (white), 100 = obstacle (black), -1 = unknown (gray)
        vis_map = np.zeros((map_array.shape[0], map_array.shape[1], 3), dtype=np.uint8)

        # Fill with appropriate colors
        vis_map[map_array == 0] = [255, 255, 255]  # Free space = white
        vis_map[map_array == 100] = [0, 0, 0]  # Obstacles = black
        vis_map[map_array == -1] = [128, 128, 128]  # Unknown = gray

        # Get map metadata
        resolution = map_info["resolution"]
        origin_x = map_info["origin_x"]
        origin_y = map_info["origin_y"]

        # Convert robot position to pixel coordinates on the map
        robot_pixel_x = int((current_x - origin_x) / resolution)
        robot_pixel_y = int((current_y - origin_y) / resolution)

        # Draw robot position (blue circle with bigger radius)
        cv2.circle(vis_map, (robot_pixel_x, robot_pixel_y), 8, (255, 0, 0), -1)

        # Draw robot orientation (blue line, thicker)
        orientation_length = 20
        endpoint_x = int(robot_pixel_x + orientation_length * math.cos(current_yaw))
        endpoint_y = int(robot_pixel_y + orientation_length * math.sin(current_yaw))
        cv2.line(
            vis_map,
            (robot_pixel_x, robot_pixel_y),
            (endpoint_x, endpoint_y),
            (255, 0, 0),
            3,
        )

        # Add robot position text
        cv2.putText(
            vis_map,
            f"Robot",
            (robot_pixel_x + 10, robot_pixel_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )

        # Draw navigation points (green circles with numbers)
        for i, (point_x, point_y, point_theta) in enumerate(navigation_points):
            # Convert world coordinates to pixel coordinates
            pixel_x = int((point_x - origin_x) / resolution)
            pixel_y = int((point_y - origin_y) / resolution)

            # Draw point (green circle, bigger)
            cv2.circle(vis_map, (pixel_x, pixel_y), 6, (0, 255, 0), -1)

            # Draw a thick black outline around the circle for better visibility
            cv2.circle(vis_map, (pixel_x, pixel_y), 7, (0, 0, 0), 1)

            # Draw ID number (larger, with background for better visibility)
            # Draw text background
            text = str(i + 1)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(
                vis_map,
                (pixel_x + 5, pixel_y - text_size[1] - 5),
                (pixel_x + text_size[0] + 10, pixel_y + 5),
                (255, 255, 255),
                -1,
            )

            # Draw text
            cv2.putText(
                vis_map,
                text,
                (pixel_x + 7, pixel_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )

            # Draw orientation (green line, thicker)
            orientation_length = 15
            endpoint_x = int(pixel_x + orientation_length * math.cos(point_theta))
            endpoint_y = int(pixel_y + orientation_length * math.sin(point_theta))
            cv2.line(
                vis_map, (pixel_x, pixel_y), (endpoint_x, endpoint_y), (0, 255, 0), 2
            )

        # Scale up the map for better visibility (2x larger)
        scale_factor = 2
        vis_map_large = cv2.resize(
            vis_map,
            (vis_map.shape[1] * scale_factor, vis_map.shape[0] * scale_factor),
            interpolation=cv2.INTER_NEAREST,
        )

        # Add a border around the map
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
        )  # Light gray border

        # Place the map in the center of the border
        map_with_border[
            border_size : border_size + vis_map_large.shape[0],
            border_size : border_size + vis_map_large.shape[1],
        ] = vis_map_large

        # Add title with useful information
        title = f"Navigation Map - Robot at ({current_x:.2f}, {current_y:.2f})"
        cv2.putText(
            map_with_border,
            title,
            (border_size, int(border_size / 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )

        # Add legend
        y_pos = border_size + vis_map_large.shape[0] + 15

        # Robot legend
        cv2.circle(map_with_border, (border_size + 15, y_pos), 8, (255, 0, 0), -1)
        cv2.putText(
            map_with_border,
            "Robot",
            (border_size + 30, y_pos + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
        )

        # Navigation point legend
        cv2.circle(map_with_border, (border_size + 150, y_pos), 6, (0, 255, 0), -1)
        cv2.circle(map_with_border, (border_size + 150, y_pos), 7, (0, 0, 0), 1)
        cv2.putText(
            map_with_border,
            "Navigation Points",
            (border_size + 165, y_pos + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
        )

        # Obstacle legend
        cv2.rectangle(
            map_with_border,
            (border_size + 350, y_pos - 8),
            (border_size + 366, y_pos + 8),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            map_with_border,
            "Obstacles",
            (border_size + 375, y_pos + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
        )

        # Add min obstacle distance info
        min_obstacle_distance = map_info.get("min_obstacle_distance", 0.25)
        cv2.putText(
            map_with_border,
            f"Min obstacle distance: {min_obstacle_distance} m",
            (border_size + 500, y_pos + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
        )

        return map_with_border

    async def execute_with_point_selection(
        self, target_description: str, map_payload: dict
    ):
        """
        Execute the navigate_in_sight primitive using point selection.

        Args:
            target_description: Description of where to navigate
            map_payload: Map payload from the robot

        Returns:
            tuple: (message, success, navigation_command)
        """
        print(
            f"NavigateInSight: Starting navigation from ({self.current_x}, {self.current_y})"
        )

        # Decode the map payload
        try:
            map_array, map_info = decode_map_payload(map_payload)
        except Exception as e:
            error_msg = f"Failed to decode map payload: {e}"
            print(error_msg)
            return error_msg, False, None

        # Decode the provided image from base64 into a cv2 image.
        try:
            image_bytes = base64.b64decode(self.image_b64)
            image_array = np.frombuffer(image_bytes, np.uint8)
            cv_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if cv_image is None:
                print("Failed to decode image into cv_image.")
                return "Failed to decode image", False, None
        except Exception as e:
            print(f"Exception decoding image: {e}")
            return "Image decode error", False, None

        # Process the depth payload.
        depth_array = None
        try:
            depth_array = decode_depth_payload(self.depth_payload)
        except Exception as e:
            print(f"Failed to decode depth payload: {e}")

        # Sample valid navigation points
        navigation_points = self.sample_valid_navigation_points(
            self.current_x,
            self.current_y,
            self.current_yaw,
            map_array,
            map_info,
            min_distance=0.5,
            max_distance=2.5,
            min_obstacle_distance=0.25,
            num_samples=8,
        )

        if not navigation_points:
            return "Could not find any valid navigation points", False, None

        # Create directory for navigation images if it doesn't exist
        os.makedirs("navigation_points", exist_ok=True)

        # Create and save map visualization with navigation points
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        map_vis = self.visualize_map_with_navigation_points(
            map_array,
            map_info,
            self.current_x,
            self.current_y,
            self.current_yaw,
            navigation_points,
        )
        map_vis_path = f"navigation_points/map_with_points_{timestamp}.jpg"
        cv2.imwrite(map_vis_path, map_vis)
        print(f"Saved map visualization to {map_vis_path}")

        # Annotate the image with navigation points
        annotated_image, point_mapping = self.annotate_navigation_points(
            cv_image,
            navigation_points,
            lambda x, y: self.world_to_image_coordinates(
                x,
                y,
                depth_array,
                self.current_x,
                self.current_y,
                self.current_yaw,
                self.horizontal_fov,
                self.vertical_fov,
            ),
        )

        # Save the annotated image
        annotated_image_path = f"navigation_points/camera_with_points_{timestamp}.jpg"
        cv2.imwrite(annotated_image_path, annotated_image)
        print(f"Saved annotated camera image to {annotated_image_path}")

        # If there's only one valid point, just use that
        if len(point_mapping) == 1:
            selected_point_id = list(point_mapping.keys())[0]
            selected_point = point_mapping[selected_point_id]

            navigation_command = {
                "x": selected_point["x"],
                "y": selected_point["y"],
                "theta": selected_point["theta"],
            }

            print(
                f"Only one valid point found, automatically selecting point {selected_point_id}"
            )
            return (
                f"Navigation to point {selected_point_id} initiated",
                True,
                navigation_command,
            )

        # Create prompt for Gemini to select a navigation point
        user_prompt = f"""
I need to navigate to: {target_description}

The image shows several numbered green circles. Each circle represents a safe location I can navigate to.
Which numbered point should I navigate to based on the description?

Please respond with ONLY the number of the best point (1, 2, 3, etc).
"""

        # Use the GenerativeAI package directly, like in navigate_through_memory.py
        try:
            # Check if API key is available
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print(
                    "Warning: GEMINI_API_KEY not found in environment. Using default point 1."
                )
                selected_point_id = "1"
            else:
                print("Calling Gemini to select a navigation point...")

                # Configure the genai library
                genai.configure(api_key=api_key)

                # Create model instance
                model = genai.GenerativeModel(
                    model_name="gemini-2.0-flash",
                    generation_config={
                        "temperature": 0,
                        "top_p": 0.95,
                        "top_k": 64,
                        "max_output_tokens": 1024,
                    },
                )

                # Prepare image for Gemini
                # Convert the annotated image to bytes for the model
                _, img_encoded = cv2.imencode(".jpg", annotated_image)
                img_bytes = img_encoded.tobytes()

                # Create content parts
                message_parts = [
                    {"text": user_prompt},
                    {"mime_type": "image/jpeg", "data": img_bytes},
                ]

                # Call Gemini model
                response = model.generate_content(message_parts)
                response_text = response.text

                print(f"Gemini response: {response_text}")

                # Extract the number from the response
                import re

                numbers = re.findall(r"\d+", response_text)
                if numbers:
                    selected_point_id = numbers[0]
                    print(f"Extracted selected point ID: {selected_point_id}")
                else:
                    # Default to the first point if no number found
                    selected_point_id = "1"
                    print(
                        f"No point number found in response, defaulting to point {selected_point_id}"
                    )
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            # Default to the first point
            selected_point_id = "1"
            print(f"Error with Gemini API, defaulting to point {selected_point_id}")

        # Get the selected point
        if selected_point_id in point_mapping:
            selected_point = point_mapping[selected_point_id]

            navigation_command = {
                "x": selected_point["x"],
                "y": selected_point["y"],
                "theta": selected_point["theta"],
            }

            print(
                f"Selected navigation point {selected_point_id}: {navigation_command}"
            )
            return (
                f"Navigation to point {selected_point_id} initiated",
                True,
                navigation_command,
            )
        else:
            # If the selected point is not valid, use the first available one
            if point_mapping:
                fallback_id = list(point_mapping.keys())[0]
                fallback_point = point_mapping[fallback_id]

                navigation_command = {
                    "x": fallback_point["x"],
                    "y": fallback_point["y"],
                    "theta": fallback_point["theta"],
                }

                print(
                    f"Selected point {selected_point_id} not found, using fallback point {fallback_id}"
                )
                return (
                    f"Navigation to point {fallback_id} initiated (fallback)",
                    True,
                    navigation_command,
                )
            else:
                return "No valid navigation points found", False, None
