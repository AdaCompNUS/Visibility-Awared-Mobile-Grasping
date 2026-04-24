import time

import numpy as np

from grasp_anywhere.utils.logger import log
from grasp_anywhere.utils.perception_utils import depth2pc


def get_current_pcd(robot):
    snapshot = robot.robot_env.get_sensor_snapshot()
    if snapshot is None:
        return None
    depth = snapshot["depth"]

    # Get intrinsics
    K = snapshot["intrinsics"]

    # Calculate T_wc (Camera to World)
    joint_names, joint_positions = snapshot["joint_states"]
    joint_dict = dict(zip(joint_names, joint_positions))
    T_wc = robot.compute_camera_pose_from_joints(joint_dict)

    # Get PCD in camera frame
    # Use voxel_size=0.05 for navigation map resolution
    pcd_cam, _ = depth2pc(depth, K, voxel_size=0.05)

    if len(pcd_cam) == 0:
        return np.empty((0, 3))

    # Transform to world
    pcd_cam_h = np.hstack((pcd_cam, np.ones((len(pcd_cam), 1))))
    pcd_world = (T_wc @ pcd_cam_h.T).T[:, :3]

    # Filter ground
    ground_thresh = 0.1
    if hasattr(robot, "env_config"):
        ground_thresh = robot.env_config.get("ground_z_threshold", 0.1)

    if len(pcd_world) > 0:
        pcd_world = pcd_world[pcd_world[:, 2] > ground_thresh]

    return pcd_world


def _find_current_waypoint_index(base_configs, current_base_config):
    """Finds the closest waypoint in the path to the robot's current base config."""
    current_pos = np.array(current_base_config[:2])
    path_positions = np.array([conf[:2] for conf in base_configs])
    distances = np.linalg.norm(path_positions - current_pos, axis=1)
    return np.argmin(distances)


def navigate(
    robot,
    target_base_pose,
    fixed_arm_joints,
    enable_replanning=True,
    max_replan_attempts=5,
    goal_tolerance_xy=0.1,
    goal_tolerance_theta=0.1,
):
    """
    Navigate to target base pose using A* (VAMP Hybrid A*) and local costmap replanning.
    Strictly Base Motion only (Arm fixed).
    """

    log.info(f"Starting navigation to {target_base_pose}...")

    # --- Initial Plan ---
    pcd_world = get_current_pcd(robot)
    if pcd_world is None:
        return False, "SENSOR_FAILURE"

    # Update Local Map (No Fusion)
    robot.clear_pointclouds()
    robot.add_pointcloud(pcd_world, filter_robot=True)

    current_base = robot.get_base_params()

    # Plan
    plan_result = robot.plan_base_motion(
        start_base=current_base, goal_base=target_base_pose, start_arm=fixed_arm_joints
    )

    if not plan_result or not plan_result["success"]:
        return False, "Initial planning failed."

    base_configs = plan_result["base_configs"]
    # Reconstruct whole body path with fixed arm
    arm_path = [fixed_arm_joints for _ in range(len(base_configs))]

    if not robot.start_whole_body_motion(arm_path, base_configs):
        return False, "Failed to start motion."

    start_time = time.time()
    # Estimate timeout (0.5s per waypoint is generous)
    timeout = len(base_configs) * 0.5 + 5.0

    # --- Monitoring Loop ---
    while True:
        # Check if complete
        if robot.is_motion_done():
            # Verify goal
            curr = robot.get_base_params()
            dist_xy = np.linalg.norm(
                np.array(curr[:2]) - np.array(target_base_pose[:2])
            )
            dist_th = abs(curr[2] - target_base_pose[2])
            # Normalize angle diff
            dist_th = (dist_th + np.pi) % (2 * np.pi) - np.pi
            dist_th = abs(dist_th)

            if dist_xy < goal_tolerance_xy and dist_th < goal_tolerance_theta:
                log.info("Navigation Success.")
                return True, "SUCCESS"
            else:
                log.warning(
                    f"Motion done but goal tolerance missed? Dist: {dist_xy:.3f}, {dist_th:.3f}"
                )
                # Ideally we might retry fine positioning but for now return Success if close enough
                # or rely on result of action
                result = robot.get_motion_result()
                if result:
                    return True, "SUCCESS"
                else:
                    return False, "EXECUTION_FAILURE"

        # Check Timeout
        if time.time() - start_time > timeout:
            robot.stop_whole_body_motion()
            return False, "TIMEOUT"

        # --- Replanning Check ---
        if enable_replanning:
            # Update Map
            pcd_world_new = get_current_pcd(robot)
            if pcd_world_new is not None:
                robot.clear_pointclouds()
                robot.add_pointcloud(pcd_world_new, filter_robot=True)

            # Check Collision
            current_base = robot.get_base_params()
            idx = _find_current_waypoint_index(base_configs, current_base)

            # Look ahead slightly? check_plan_for_collisions handles it.
            # We assume arm_path is fixed, but check_plan uses it.
            # Optimization: Downsample checking?
            is_collision = robot.check_plan_for_collisions(arm_path, base_configs, idx)

            if is_collision:
                log.warning("Obstacle detected! Replanning...")
                robot.stop_whole_body_motion()

                replan_success = False
                for attempt in range(max_replan_attempts):
                    pcd_world_new = get_current_pcd(robot)
                    robot.clear_pointclouds()
                    robot.add_pointcloud(pcd_world_new, filter_robot=True)

                    current_base = robot.get_base_params()
                    plan_result = robot.plan_base_motion(
                        start_base=current_base,
                        goal_base=target_base_pose,
                        start_arm=fixed_arm_joints,
                    )

                    if plan_result and plan_result["success"]:
                        log.info("Replanning successful.")
                        base_configs = plan_result["base_configs"]
                        arm_path = [fixed_arm_joints for _ in range(len(base_configs))]

                        if robot.start_whole_body_motion(arm_path, base_configs):
                            replan_success = True
                            break

                    log.warning(f"Replan attempt {attempt+1} failed.")
                    time.sleep(0.1)  # Wait a bit ?

                if not replan_success:
                    return False, "REPLANNING_FAILURE"

        time.sleep(0.1)
