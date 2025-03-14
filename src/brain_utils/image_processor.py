import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import imghdr  # For detecting image format from binary data


from src.utils import decode_depth_payload


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

        return base64_img, depth_payload, robot_coords

    def process_depth_map(self, depth_payload):
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

        return depth_map

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
