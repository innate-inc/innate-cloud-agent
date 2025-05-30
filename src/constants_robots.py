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
}

ROBOT_PARAMS_MAURICE_OAK_D = {
    "vertical_fov": 80.0,
    "horizontal_fov": 128.0,
    "camera_resolution": (1280, 800),
    "min_obstacle_distance": 0.10,
    "camera_info": {
        "pitch_deg": -10,
        "x_cam": 0.0197,
        "height_cam": 0.19663,
    },
    "average_pos_cov_threshold": 0.05,
    "average_yaw_cov_threshold": 0.04,
}

ROBOT_PARAMS_TO_USE = ROBOT_PARAMS_MAURICE_OAK_D
