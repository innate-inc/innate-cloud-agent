# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Innate Inc

import base64
import numpy as np


def decode_depth_payload(depth_payload):
    # Retrieve metadata from the depth payload.
    height = depth_payload["height"]
    width = depth_payload["width"]
    encoding = depth_payload["encoding"]
    is_bigendian = depth_payload.get("is_bigendian", 0)
    data_b64 = depth_payload["data"]

    # Decode the base64 data to raw bytes.
    depth_bytes = base64.b64decode(data_b64)

    # Select the numpy dtype based on encoding.
    if encoding == "16UC1":
        dtype = np.uint16
    elif encoding == "32FC1":
        dtype = np.float32
    elif encoding == "8UC1":
        dtype = np.uint8
    else:
        raise ValueError(f"Unrecognized encoding type: {encoding}")

    # Set the byte order based on the 'is_bigendian' flag.
    np_dtype = np.dtype(dtype)
    np_dtype = np_dtype.newbyteorder(">" if is_bigendian else "<")

    # Create the numpy array from the decoded bytes.
    depth_array = np.frombuffer(depth_bytes, dtype=np_dtype)

    # Ensure the size is consistent with the provided dimensions.
    if depth_array.size != height * width:
        raise ValueError("Mismatch between depth array size and provided dimensions.")

    depth_array = depth_array.reshape((height, width))
    return depth_array


def decode_map_payload(map_payload):
    """
    Decode map payload from base64 to numpy array.

    Args:
        map_payload (dict): Dictionary containing map data and metadata

    Returns:
        tuple: (map_array, map_info) where map_array is a numpy array and map_info contains metadata
    """
    # Retrieve metadata from the map payload
    width = map_payload["width"]
    height = map_payload["height"]
    resolution = map_payload["resolution"]
    origin_x = map_payload["origin_x"]
    origin_y = map_payload["origin_y"]
    origin_z = map_payload["origin_z"]
    origin_yaw = map_payload["origin_yaw"]
    frame_id = map_payload["frame_id"]
    data_b64 = map_payload["data"]

    # Decode the base64 data to raw bytes
    map_bytes = base64.b64decode(data_b64)

    # Create the numpy array from the decoded bytes (maps are typically int8)
    map_array = np.frombuffer(map_bytes, dtype=np.int8)

    # Ensure the size is consistent with the provided dimensions
    if map_array.size != height * width:
        raise ValueError("Mismatch between map array size and provided dimensions.")

    # Reshape the array to 2D
    map_array = map_array.reshape((height, width))

    # Create a metadata dictionary
    map_info = {
        "resolution": resolution,
        "width": width,
        "height": height,
        "origin_x": origin_x,
        "origin_y": origin_y,
        "origin_z": origin_z,
        "origin_yaw": origin_yaw,
        "frame_id": frame_id,
    }

    return map_array, map_info
