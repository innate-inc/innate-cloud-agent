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
            tuple: (base64_img, depth_payload, robot_coords, map_payload, additional_image_data, camera_info)

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

        # Validate required map payload
        if "map" not in payload:
            raise ValueError("Missing required 'map' in payload")
        map_payload = payload["map"]

        # Depth payload is now optional
        depth_payload = payload.get("depth")

        additional_image_data = {}

        # Check if we received an additional camera image
        if "additional_camera" in payload:
            additional_image_data["camera_type"] = payload["additional_camera"][
                "camera_type"
            ]
            additional_image_data["image_b64"] = payload["additional_camera"][
                "image_b64"
            ]

        # Extract camera info from payload (required)
        if "camera_info" not in payload:
            raise ValueError("Missing required 'camera_info' in payload")

        camera_info = payload["camera_info"]

        # Validate required camera info fields
        required_camera_fields = [
            "horizontal_fov",
            "vertical_fov",
            "pitch_deg",
            "x_cam",
            "height_cam",
        ]
        missing_fields = [
            field for field in required_camera_fields if field not in camera_info
        ]
        if missing_fields:
            raise ValueError(
                f"Camera info missing required fields: " f"{', '.join(missing_fields)}"
            )

        # Validate required depth payload if present
        if depth_payload is not None:
            required_depth_fields = ["height", "width", "encoding", "data"]
            missing_fields = [
                field for field in required_depth_fields if field not in depth_payload
            ]
            if missing_fields:
                raise ValueError(
                    f"Depth payload missing required fields: "
                    f"{', '.join(missing_fields)}"
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
                f"Robot coordinates missing required fields: "
                f"{', '.join(missing_fields)}"
            )

        return (
            base64_img,
            depth_payload,
            robot_coords,
            map_payload,
            additional_image_data,
            camera_info,
        )

    def process_depth(self, depth_payload):
        """
        Process depth map data.

        Args:
            depth_payload (dict): Dictionary containing depth map data and metadata

        Returns:
            depth_map (np.ndarray or None): Processed depth map as a numpy array,
                                          or None if no depth_payload.
        """
        if depth_payload is None:
            self.logger.info("No depth payload provided, skipping depth processing.")
            return None

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

        # Create a PIL image (L mode for grayscale) and convert it to RGB so we can add
        # colored text
        img = Image.fromarray(normalized_depth, mode="L").convert("RGB")

        # Prepare debug text showing the min and max values.
        debug_text = f"Min: {d_min} Max: {d_max}"
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        # Draw the text at position (10, 10) with a contrasting color (red)
        draw.text((10, 10), debug_text, font=font, fill=(255, 0, 0))

        # Save the annotated depth map as a PNG file.
        os.makedirs("depth_maps", exist_ok=True)
        img.save("depth_maps/depth_map.png")

        return depth_map

    def process_map_with_robot(self, map_payload, robot_coords):
        """
        Process map data and draw the robot position on it.

        HOLD ON IS THAT JUST A HELPER FUNCTION WHAT THE HELL

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

            # Flip the map vertically to correct the orientation
            rgb_map = np.flipud(rgb_map)

            # Create PIL image from numpy array
            map_img = Image.fromarray(rgb_map)
            draw = ImageDraw.Draw(map_img)

            # Add grid lines to the map (every 20 pixels)
            grid_spacing = 20
            grid_color = (200, 200, 200)  # Light gray

            # Draw horizontal grid lines
            for y in range(0, map_info["height"], grid_spacing):
                draw.line([(0, y), (map_info["width"], y)], fill=grid_color, width=1)

            # Draw vertical grid lines
            for x in range(0, map_info["width"], grid_spacing):
                draw.line([(x, 0), (x, map_info["height"])], fill=grid_color, width=1)

            # Add coordinate labels at major grid points (every 100 pixels)
            major_grid_spacing = 100
            label_color = (100, 100, 100)  # Darker gray
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except IOError:
                font = ImageFont.load_default()

            for x in range(0, map_info["width"], major_grid_spacing):
                for y in range(0, map_info["height"], major_grid_spacing):
                    # Calculate world coordinates
                    world_x = x * map_info["resolution"] + map_info["origin_x"]
                    # Adjust Y coordinate calculation for flipped map
                    flipped_y = map_info["height"] - y
                    world_y = flipped_y * map_info["resolution"] + map_info["origin_y"]
                    label = f"({world_x:.1f}, {world_y:.1f})"
                    draw.text((x + 2, y + 2), label, font=font, fill=label_color)

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
            # Use negative sin because Y increases downward in image coordinates
            end_y = robot_y - int(line_length * math.sin(robot_theta))

            # Draw the orientation line
            draw.line(
                [robot_x, robot_y, end_x, end_y], fill=(0, 0, 255), width=2  # Blue
            )

            # Add a small triangle at the end of the orientation line to show direction
            arrow_size = 4
            draw.polygon(
                [
                    (end_x, end_y),
                    (
                        end_x - int(arrow_size * math.cos(robot_theta - math.pi / 6)),
                        end_y + int(arrow_size * math.sin(robot_theta - math.pi / 6)),
                    ),
                    (
                        end_x - int(arrow_size * math.cos(robot_theta + math.pi / 6)),
                        end_y + int(arrow_size * math.sin(robot_theta + math.pi / 6)),
                    ),
                ],
                fill=(0, 0, 255),  # Blue
            )

            # Add map metadata as text
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()

            theta_deg = robot_coords['theta'] * 180.0 / math.pi  # Convert radians to degrees
            metadata_text = (
                f"Resolution: {resolution:.3f}m/pixel, "
                f"Size: {map_info['width']}x{map_info['height']}, "
                f"Robot: ({robot_coords['x']:.2f}, {robot_coords['y']:.2f}, "
                f"{theta_deg:.1f}°)"
            )

            draw.text((10, 10), metadata_text, font=font, fill=(255, 0, 0))

            # Save the map with robot position
            os.makedirs("maps", exist_ok=True)
            map_img.save("maps/map_with_robot.png")

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
