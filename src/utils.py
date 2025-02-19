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
