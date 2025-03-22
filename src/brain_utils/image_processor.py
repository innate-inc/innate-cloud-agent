import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import base64
from io import BytesIO
import imghdr  # For detecting image format from binary data
import math


from src.utils import decode_depth_payload, decode_map_payload


class ImageProcessor:
    def __init__(self, logger):
        self.logger = logger

    def extract_image_data(self, payload):
        """
        Extract and validate image data from the payload.

        Args:
            payload (dict): Dictionary containing image data and metadata

        Returns:
            tuple: (base64_img, depth_payload, robot_coords)

        Raises:
            ValueError: If required data is missing or malformed
        """
        # Validate payload exists
        if not payload:
            raise ValueError("Empty payload received")

        # Check for required image_b64 field
        if "image_b64" not in payload:
            raise ValueError("Missing required 'image_b64' in payload")

        base64_img = payload["image_b64"]

        # Validate base64_img is not empty
        if not base64_img:
            raise ValueError("Empty image data received")

        # Validate required depth payload
        if "depth" not in payload:
            raise ValueError("Missing required 'depth' in payload")

        if "map" not in payload:
            raise ValueError("Missing required 'map' in payload")

        map_payload = payload["map"]

        depth_payload = payload["depth"]

        # Verify depth payload contains required fields
        required_depth_fields = ["height", "width", "encoding", "data"]
        missing_fields = [
            field for field in required_depth_fields if field not in depth_payload
        ]
        if missing_fields:
            raise ValueError(
                f"Depth payload missing required fields: {', '.join(missing_fields)}"
            )

        # Validate required robot coordinates
        if "robot_coords" not in payload:
            raise ValueError("Missing required 'robot_coords' in payload")

        robot_coords = payload["robot_coords"]

        # Verify robot_coords contains required fields
        required_coord_fields = ["x", "y", "theta"]
        missing_fields = [
            field for field in required_coord_fields if field not in robot_coords
        ]
        if missing_fields:
            raise ValueError(
                f"Robot coordinates missing required fields: {', '.join(missing_fields)}"
            )

        return base64_img, depth_payload, robot_coords, map_payload

    def process_depth_map(self, depth_payload, map_payload=None, robot_coords=None):
        """
        Process depth map data and optionally map data with robot position.

        Args:
            depth_payload (dict): Dictionary containing depth map data and metadata
            map_payload (dict, optional): Dictionary containing map data and metadata
            robot_coords (dict, optional): Dictionary containing robot coordinates

        Returns:
            depth_map (np.ndarray): Processed depth map as a numpy array
        """
        depth_map = decode_depth_payload(depth_payload)
        # Compute min and max values from the depth map.
        d_min = depth_map.min()
        d_max = depth_map.max()

        # Normalize the depth map so that the maximum value becomes 255.
        if d_max > d_min:
            normalized_depth = ((depth_map - d_min) / (d_max - d_min) * 255).astype(
                np.uint8
            )
        else:
            normalized_depth = np.zeros_like(depth_map, dtype=np.uint8)

        # Create a PIL image (L mode for grayscale) and convert it to RGB so we can add colored text.
        img = Image.fromarray(normalized_depth, mode="L").convert("RGB")

        # Prepare debug text showing the min and max values.
        debug_text = f"Min: {d_min} Max: {d_max}"
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        # Draw the text at position (10, 10) with a contrasting color (red in this case).
        draw.text((10, 10), debug_text, font=font, fill=(255, 0, 0))

        # Save the annotated depth map as a PNG file.
        os.makedirs("depth_maps", exist_ok=True)
        img.save("depth_maps/depth_map.png")

        # Process map data if available
        if map_payload and robot_coords:
            self.process_map_with_robot(map_payload, robot_coords)

        return depth_map

    def process_map_with_robot(self, map_payload, robot_coords):
        """
        Process map data and draw the robot position on it.

        Args:
            map_payload (dict): Dictionary containing map data and metadata
            robot_coords (dict): Dictionary containing robot coordinates

        Returns:
            map_array (np.ndarray): Processed map as a numpy array
        """
        try:
            # Decode the map payload
            map_array, map_info = decode_map_payload(map_payload)

            # Convert occupancy grid (-1, 0, 100) to an RGB image
            # -1: Unknown (gray), 0: Free (white), 100: Occupied (black)
            rgb_map = np.zeros(
                (map_array.shape[0], map_array.shape[1], 3), dtype=np.uint8
            )

            # Unknown space (gray)
            rgb_map[map_array == -1] = [128, 128, 128]

            # Free space (white)
            rgb_map[map_array == 0] = [255, 255, 255]

            # Occupied space (black)
            rgb_map[map_array == 100] = [0, 0, 0]

            # Create PIL image from numpy array
            map_img = Image.fromarray(rgb_map)
            draw = ImageDraw.Draw(map_img)

            # Calculate robot position on the map
            # Convert robot position from world coordinates to map coordinates
            resolution = map_info["resolution"]
            origin_x = map_info["origin_x"]
            origin_y = map_info["origin_y"]

            # Calculate pixel position
            robot_x = int((robot_coords["x"] - origin_x) / resolution)
            robot_y = int((robot_coords["y"] - origin_y) / resolution)

            # In many 2D maps, Y is pointing down in image coordinates
            robot_y = map_info["height"] - robot_y

            # Draw robot as a red circle with a line indicating orientation
            robot_radius = 5
            robot_theta = robot_coords["theta"]
            line_length = 15

            # Draw the robot circle
            draw.ellipse(
                [
                    robot_x - robot_radius,
                    robot_y - robot_radius,
                    robot_x + robot_radius,
                    robot_y + robot_radius,
                ],
                fill=(255, 0, 0),  # Red
                outline=(0, 0, 0),  # Black outline
            )

            # Draw a line indicating the robot's orientation
            end_x = robot_x + int(line_length * math.cos(robot_theta))
            end_y = robot_y - int(line_length * math.sin(robot_theta))
            draw.line(
                [robot_x, robot_y, end_x, end_y], fill=(0, 0, 255), width=2  # Blue
            )

            # Add map metadata as text
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()

            metadata_text = (
                f"Resolution: {resolution:.3f}m/pixel, "
                f"Size: {map_info['width']}x{map_info['height']}, "
                f"Robot: ({robot_coords['x']:.2f}, {robot_coords['y']:.2f}, "
                f"{robot_coords['theta']:.2f})"
            )

            draw.text((10, 10), metadata_text, font=font, fill=(255, 0, 0))

            # Save the map with robot position
            os.makedirs("maps", exist_ok=True)
            map_img.save("maps/map_with_robot.png")

            self.logger.info("Saved map with robot position to maps/map_with_robot.png")
            return map_array

        except Exception as e:
            self.logger.error(f"Error processing map: {e}")
            return None

    def ensure_jpeg_format(self, base64_img):
        """
        Ensures the base64 encoded image is a JPEG.
        Converts PNGs to JPEGs and rejects transparent images.

        Args:
            base64_img (str): Base64 encoded image

        Returns:
            str: Base64 encoded JPEG image

        Raises:
            ValueError: If the image has transparency which is not supported
        """
        # Decode base64 image
        try:
            img_data = base64.b64decode(base64_img)

            # Detect the image format from the binary data
            img_format = imghdr.what(None, h=img_data)

            img = Image.open(BytesIO(img_data))

            # Check if image has actual transparent pixels, not just an alpha channel
            has_transparency = False

            if img.mode == "RGBA":
                # Check if any alpha values are less than 255 (not fully opaque)
                alpha = np.array(img.split()[3])
                if np.any(alpha < 255):
                    has_transparency = True
            elif img.mode == "P" and "transparency" in img.info:
                # For palette mode with transparency
                has_transparency = True

            if has_transparency:
                raise ValueError("Images with transparent pixels are not supported")

            # If already JPEG, return as is
            if img_format == "jpeg":
                return base64_img

            # Convert to RGB (in case of palette mode or other modes)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Convert to JPEG
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=95)
            jpeg_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            self.logger.info(f"Converted image from {img_format or 'unknown'} to JPEG")
            return jpeg_base64

        except Exception as e:
            self.logger.error(f"Error processing image format: {e}")
            raise ValueError(f"Invalid image format: {e}")
