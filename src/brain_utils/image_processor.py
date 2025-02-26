import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.utils import decode_depth_payload


class ImageProcessor:
    def __init__(self, logger):
        self.logger = logger

    def extract_image_data(self, payload):
        base64_img = payload["image_b64"]
        depth_payload = payload.get("depth")
        robot_coords = payload.get("robot_coords")
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
        self.logger.info(
            f"Depth map saved as depth_map.png with debug info: {debug_text}"
        )

        return depth_map
