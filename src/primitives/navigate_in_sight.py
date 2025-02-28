import traceback
from src.primitives.types import Primitive
import asyncio
import base64
import json
import cv2
import numpy as np
import time
from ultralytics import YOLO, SAM

# Import our Groq helpers (assumed to be defined elsewhere)
# from orchestrator.agent.groq_instant_use import query_groq_classifier, query_groq_vlm

# Utility to decode depth payload (assumed defined in src/utils.py)
from src.utils import decode_depth_payload


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
        return "To use to navigate to an object or target in sight. Provide a target object name, such as 'shelf', 'table', 'chair', etc."

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
        target_object: str,
    ):
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
            print("Segmentation failed. Aborting navigation.")
            return "Segmentation failed", False, None

        # Select the segmentation mask with highest confidence.
        target_info = segmentation_masks.get("1")
        if not target_info:
            print("No valid segmentation mask found.")
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
