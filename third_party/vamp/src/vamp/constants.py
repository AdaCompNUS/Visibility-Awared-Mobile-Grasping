DEFAULT_ITERATIONS = 100000

ROBOT_JOINTS = {
    "fetch": [
        "torso_lift_joint",
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "upperarm_roll_joint",
        "elbow_flex_joint",
        "forearm_roll_joint",
        "wrist_flex_joint",
        "wrist_roll_joint",
    ],
}

ROBOT_RRT_RANGES = {
    "fetch": 1.0,
}

ROBOT_RADII_RANGES = {
    "fetch": (0.012, 0.055),
}

ROBOT_FIRST_JOINT_LOCATIONS = {
    "fetch": [0.0, 0.0, 0.4],
}

ROBOT_MAX_RADII = {
    "fetch": 1.5,
}

POINT_RADIUS = 0.0025
