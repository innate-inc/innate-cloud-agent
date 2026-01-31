import os
from dotenv import load_dotenv

load_dotenv()

# Minimum required client version (semver)
MIN_CLIENT_VERSION = "0.3.0"

ROBOT_PARAMS_SIM = {
    "vertical_fov": 80.0,
    "horizontal_fov": 128.0,
    "camera_resolution": (640, 480),
    "min_obstacle_distance": 0.10,
    "camera_info": {
        "pitch_deg": 0,
        "x_cam": 0.0,
        "height_cam": 0.2,
    },
    "average_pos_cov_threshold": 0.05,
    "average_yaw_cov_threshold": 0.04,  # Could be lower?
    "enable_visualizations": True,
}

ROBOT_PARAMS_MAURICE_OAK_D = {
    "vertical_fov": 80.0,
    "horizontal_fov": 128.0,
    "camera_resolution": (1280, 800),
    "min_obstacle_distance": 0.00,  # our costmap is already inflated
    "camera_info": {
        "pitch_deg": -10,
        "x_cam": 0.0197,
        "height_cam": 0.19663,
    },
    "average_pos_cov_threshold": 0.25,
    "average_yaw_cov_threshold": 0.18,
    "enable_visualizations": True,
}

ROBOTS_PARAMS = {
    "maurice_oak_d": ROBOT_PARAMS_MAURICE_OAK_D,
    "sim": ROBOT_PARAMS_SIM,
}

# Get robot type from environment variable, default to maurice_oak_d
ROBOT_TYPE = os.environ.get("ROBOT_TYPE", "undef")

# Validate robot type
if ROBOT_TYPE not in ROBOTS_PARAMS:
    raise ValueError(
        f"Invalid ROBOT_TYPE '{ROBOT_TYPE}'. Must be one of: {list(ROBOTS_PARAMS.keys())}"
    )

ROBOT_PARAMS_TO_USE = ROBOTS_PARAMS[ROBOT_TYPE]
