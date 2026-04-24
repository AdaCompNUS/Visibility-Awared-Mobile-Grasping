import os
import time

import numpy as np
import rospy
from nav_msgs.msg import Path

from grasp_anywhere.data_collector.trajectory_collector import episode_record
from grasp_anywhere.observation import GazeOptimizer
from grasp_anywhere.observation.finean_gaze_optimizer import FineanGazeOptimizer
from grasp_anywhere.utils.logger import log, log_warn_throttle
from grasp_anywhere.utils.rviz_utils import publish_base_path


def save_whole_body_trajectory(
    arm_path, base_configs, filename="debug/whole_body_plan.npy"
):
    """Saves the whole-body trajectory to a .npy file."""
    if len(arm_path) != len(base_configs):
        log.error(
            f"Arm path ({len(arm_path)}) and base configs ({len(base_configs)}) have different lengths, cannot save."
        )
        return

    # The arm_path object from VAMP might not be a simple list of lists.
    # We need to convert it properly.
    arm_path_list = []
    for point in arm_path:
        # This handles different types that VAMP might return
        if hasattr(point, "to_list"):
            arm_path_list.append(point.to_list())
        else:
            arm_path_list.append(list(point))

    base_configs_array = np.array(base_configs)
    arm_path_array = np.array(arm_path_list)

    if base_configs_array.shape[0] != arm_path_array.shape[0]:
        log.error(
            f"Arm path ({arm_path_array.shape[0]}) and base configs "
            f"({base_configs_array.shape[0]}) have different lengths after conversion, cannot save."
        )
        return

    # combined trajectory: [x, y, theta, torso, 7 arm joints] -> 11 columns
    combined_trajectory = np.hstack((base_configs_array, arm_path_array))

    try:
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        np.save(filename, combined_trajectory)
        log.info(
            f"Saved whole-body trajectory with shape {combined_trajectory.shape} to {filename}"
        )
    except Exception as e:
        log.error(f"Failed to save trajectory to {filename}: {e}")


def _find_current_waypoint_index(base_configs, current_base_config):
    """Finds the closest waypoint in the path to the robot's current base config."""
    current_pos = np.array(current_base_config[:2])
    path_positions = np.array([conf[:2] for conf in base_configs])
    distances = np.linalg.norm(path_positions - current_pos, axis=1)
    return np.argmin(distances)


def _maybe_record_episode(robot, snapshot):
    """Helper to record a single timestep if an episode is active."""
    active_cfg = getattr(robot, "_active_episode_cfg", None)
    if active_cfg is not None and snapshot is not None:
        qpos = np.asarray(
            [*robot.get_base_params(), *robot.get_current_planning_joints()],
            dtype=np.float32,
        )
        goal_xyz_world = getattr(robot, "_active_episode_goal_xyz_world", None)
        if goal_xyz_world is None:
            goal_xyz_world = np.zeros((3,), dtype=np.float32)

        # Get head qpos (pan, tilt)
        joint_states_with_head = robot.get_current_planning_joints_with_head()
        if joint_states_with_head is None:
            raise ValueError("Failed to retrieve joint states with head for recording.")
        head_qpos = np.asarray(joint_states_with_head[-2:], dtype=np.float32)

        active_ds = getattr(robot, "_active_episode_ds", None)
        robot._active_episode_ds = episode_record(
            active_ds,
            cfg=active_cfg,
            goal_xyz_world=goal_xyz_world,
            rgb=snapshot["rgb"],
            depth=snapshot["depth"],
            qpos=qpos,
            head_qpos=head_qpos,
        )


def _update_collision_environment(robot, enable_pcd_alignment=False):
    """
    Updates the collision environment using the robot's Scene manager.
    1. Fetches latest sensor data (depth + camera pose).
    2. Calls scene.update() to process and integrate the new point cloud.
       (Scene automatically filters robot body points via its robot_filter callback)
    3. Retrieves the combined environment from the scene.
    4. Updates the planner's collision model.
    """
    # 1. Get latest data from sensors/tf atomically (important for sim)
    snapshot = robot.robot_env.get_sensor_snapshot()

    if snapshot is None:
        log.warning("Could not get sensor snapshot.")
        return

    # Record if an episode is active
    _maybe_record_episode(robot, snapshot)

    depth = snapshot["depth"]
    intrinsics = snapshot["intrinsics"]

    # Compute camera pose from joint states snapshot (uses FK)
    # This ensures camera pose matches the same timestep as depth
    joint_names, joint_positions = snapshot["joint_states"]
    joint_dict = dict(zip(joint_names, joint_positions))
    T_wc = robot.compute_camera_pose_from_joints(joint_dict)

    # 2. Update the scene with synchronized joint_dict for robot filtering
    robot.scene.update(
        depth, intrinsics, T_wc, joint_dict, enable_icp_alignment=enable_pcd_alignment
    )

    # 3. Get the combined environment from the scene
    # Note: Scene already filters robot body points via robot_filter callback
    combined_pcd = robot.scene.current_environment()
    if combined_pcd is not None and combined_pcd.shape[0] > 0:
        robot.clear_pointclouds()
        # Disable filter_robot here because:
        # 1. Scene.update() already filters dynamic points against the robot state at observation time.
        # 2. Filtering the *accumulated* map against the *current* pose causes "ghost" obstacles:
        #    points that were masked at the start (allowing the plan) suddenly appear when the
        #    robot moves away, invalidating the goal.
        robot.add_pointcloud(combined_pcd, filter_robot=True, point_radius=0.03)
        log.info("Collision environment updated from scene manager.")
    return snapshot


def move_to_config_with_replanning(
    robot,
    goal_joints,
    goal_base,
    enable_replanning=True,
    enable_pcd_alignment=False,
    enable_gaze_control=True,
    max_replan_attempts=5,
):
    """
    Moves the robot to a target configuration, with a replanning loop.

    Args:
        robot: The Fetch robot instance (which contains the scene).
        goal_joints: The goal joint configuration (8-dof: [torso, 7 arm]).
        goal_base: The goal base configuration [x, y, theta].
        enable_replanning: If True, enables the 1Hz replanning loop.
        enable_pcd_alignment: If True, enables point cloud alignment during replanning.
        enable_gaze_control: If True, enables gaze optimization during motion.
        max_replan_attempts: Maximum attempts for replanning due to collisions or timeouts.

    Returns:
        (bool, str): Tuple of success status and a message.
    """

    if not enable_replanning:
        log.info("Replanning disabled. Executing standard motion.")
        current_joints = robot.get_current_planning_joints()
        current_base = robot.get_base_params()
        plan_result = robot.plan_whole_body_motion(
            current_joints, goal_joints, list[float](current_base), goal_base
        )
        if not plan_result or not plan_result["success"]:
            return False, "Failed to plan initial motion."

        success = robot.execute_whole_body_motion(
            plan_result["arm_path"], plan_result["base_configs"]
        )
        if success:
            return True, "Motion successful without replanning."
        else:
            return False, "Motion failed without replanning."

    log.info("Moving to config with replanning enabled.")
    path_pub = None
    try:
        path_pub = rospy.Publisher("/planned_base_path", Path, queue_size=1, latch=True)
    except Exception:
        path_pub = None

    # --- Update Collision Environment Before Initial Planning ---
    log.info("Step 0: Updating collision environment with latest observation...")
    _update_collision_environment(robot, enable_pcd_alignment)

    # --- Initial Plan ---
    log.info("Step 1: Planning initial whole-body motion...")
    current_joints = robot.get_current_planning_joints()
    current_base = robot.get_base_params()

    distance = np.linalg.norm(np.array(current_base[:2]) - np.array(goal_base[:2]))
    planner = "fcit_wb" if distance < 0.0 else "rrtc"
    log.info(f"Distance is {distance:.2f}m, using planner: {planner}")

    plan_result = robot.plan_whole_body_motion(
        current_joints, goal_joints, list(current_base), goal_base, planner=planner
    )
    if not plan_result or not plan_result["success"]:
        log.error("Failed to plan initial whole-body motion.")
        return False, "Failed to plan initial whole-body motion."

    arm_path = plan_result["arm_path"]
    base_configs = plan_result["base_configs"]
    try:
        publish_base_path(path_pub, base_configs, "map")
    except Exception:
        pass
    save_whole_body_trajectory(
        arm_path, base_configs, filename="debug/initial_config_plan.npy"
    )

    # --- Initialize Gaze Control ---
    gaze_optimizer = None
    if enable_gaze_control:
        try:
            # Combine base_configs and arm_path into 11-DOF trajectory
            # base_configs: N x 3 [x, y, theta]
            # arm_path: N x 8 [torso, 7 arm]
            # whole_body: N x 11 [x, y, theta, torso, 7 arm]
            whole_body_trajectory = np.hstack(
                (np.array(base_configs), np.array(arm_path))
            )
            gaze_cfg = getattr(robot, "gaze_config", {})
            gaze_type = gaze_cfg.get("type", "default")
            if gaze_type == "finean":
                gaze_optimizer = FineanGazeOptimizer(
                    robot,
                    lookahead_window=gaze_cfg.get("lookahead_window", 40),
                )
            else:
                gaze_optimizer = GazeOptimizer(
                    robot,
                    lookahead_window=gaze_cfg.get("lookahead_window", 40),
                    decay_rate=gaze_cfg.get("decay_rate", 0.999),
                    velocity_weight=gaze_cfg.get("velocity_weight", 1.0),
                    joint_priorities=gaze_cfg.get("joint_priorities", None),
                )
            gaze_optimizer.set_trajectory(whole_body_trajectory)
            log.info("Gaze control enabled for trajectory tracking")
        except Exception as e:
            log.warning(f"Failed to initialize gaze control: {e}")
            gaze_optimizer = None

    # --- Motion Execution and Replanning Loop ---
    if not robot.start_whole_body_motion(arm_path, base_configs):
        log.error("Failed to start initial motion.")
        return False, "Failed to start initial motion."

    # Estimate execution time and set a timeout
    time_per_waypoint = 0.2  # seconds
    estimated_duration = len(arm_path) * time_per_waypoint
    timeout = 2.0 * estimated_duration
    start_time = time.time()

    while True:
        if robot.is_motion_done():
            break
        try:
            if rospy.is_shutdown():
                break
        except Exception:
            pass
        # 1. Update collision environment
        _update_collision_environment(robot, enable_pcd_alignment)

        # Check if the goal is still valid
        # goal_joints: 8-dof, goal_base: 3-dof [x, y, theta]
        # validate_whole_body_config returns True if IN COLLISION
        if robot.validate_whole_body_config(goal_joints, goal_base):
            log.warning("Target configuration is now in collision. Aborting.")
            robot.stop_whole_body_motion()
            return False, "TARGET_IN_COLLISION"

        if robot.validate_whole_body_config(
            [0.3, 1.32, 1.4, -0.2, 1.72, 0.0, 1.66, 0.0], goal_base
        ):
            log.warning("Navigation goal configuration is now in collision. Aborting.")
            robot.stop_whole_body_motion()
            return False, "TARGET_IN_COLLISION"

        # 2. Check current plan for collisions
        current_base_config = robot.get_base_params()
        waypoint_index = _find_current_waypoint_index(base_configs, current_base_config)

        # 2.5. Update gaze to look ahead at future trajectory
        if gaze_optimizer is not None:
            try:
                gaze_optimizer.update(waypoint_index)
            except Exception as e:
                log_warn_throttle(5.0, f"Gaze update failed: {e}")

        # Downsample path for verification
        num_points = len(arm_path)
        max_verification_points = 128
        if num_points > max_verification_points:
            step = max(1, num_points // max_verification_points)
            arm_path_for_check = [arm_path[i] for i in range(0, num_points, step)]
            base_configs_for_check = base_configs[::step]
            waypoint_index_for_check = waypoint_index // step
        else:
            arm_path_for_check = arm_path
            base_configs_for_check = base_configs
            waypoint_index_for_check = waypoint_index

        is_collision = robot.check_plan_for_collisions(
            arm_path_for_check, base_configs_for_check, waypoint_index_for_check
        )

        elapsed_time = time.time() - start_time
        timed_out = elapsed_time > timeout
        if timed_out:
            log.warning(
                f"Motion execution timed out after {elapsed_time:.2f}s! Stopping."
            )
            robot.stop_whole_body_motion()
            return False, "Motion timed out."

        if is_collision:
            log.warning("Collision detected! Stopping motion and replanning.")
            robot.stop_whole_body_motion()

            replan_success = False
            for attempt in range(max_replan_attempts):
                log.info(f"Replanning attempt {attempt + 1}/{max_replan_attempts}...")
                current_joints = robot.get_current_planning_joints()
                current_base = robot.get_base_params()

                distance = np.linalg.norm(
                    np.array(current_base[:2]) - np.array(goal_base[:2])
                )
                planner = "fcit_wb" if distance < 0.0 else "rrtc"
                log.info(f"Distance is {distance:.2f}m, using planner: {planner}")

                plan_result = robot.plan_whole_body_motion(
                    current_joints,
                    goal_joints,
                    list(current_base),
                    goal_base,
                    planner=planner,
                )

                if plan_result and plan_result["success"]:
                    log.info("Replanning successful.")
                    arm_path = plan_result["arm_path"]
                    base_configs = plan_result["base_configs"]
                    try:
                        publish_base_path(path_pub, base_configs, "map")
                    except Exception:
                        pass
                    save_whole_body_trajectory(
                        arm_path,
                        base_configs,
                        filename=f"debug/replan_config_{attempt}.npy",
                    )

                    if not robot.start_whole_body_motion(arm_path, base_configs):
                        log.error("Failed to start motion after replanning.")
                        replan_success = False
                        break

                    # Reset timeout for the new path
                    estimated_duration = len(arm_path) * time_per_waypoint
                    timeout = 2.0 * estimated_duration
                    start_time = time.time()

                    # Update gaze optimizer with new trajectory
                    if gaze_optimizer is not None:
                        try:
                            whole_body_trajectory = np.hstack(
                                (np.array(base_configs), np.array(arm_path))
                            )
                            gaze_optimizer.set_trajectory(whole_body_trajectory)
                            log.info("Gaze control updated with replanned trajectory")
                        except Exception as e:
                            log.warning(f"Failed to update gaze control: {e}")

                    replan_success = True
                    break
                else:
                    log.warning(f"Replanning attempt {attempt + 1} failed.")

            if not replan_success:
                log.error(
                    f"Failed to replan after {max_replan_attempts} attempts. Aborting motion."
                )
                return (
                    False,
                    f"Failed to replan after {max_replan_attempts} attempts.",
                )
        else:
            log.info("Path is clear. Continuing motion.")

            # Replanning rate: 1 Hz, but update gaze at 10 Hz
            for _ in range(5):
                if robot.is_motion_done():
                    break
                current_base_config = robot.get_base_params()
                waypoint_index = _find_current_waypoint_index(
                    base_configs, current_base_config
                )
                if gaze_optimizer is not None:
                    gaze_optimizer.update(waypoint_index)

                _maybe_record_episode(robot, robot.robot_env.get_sensor_snapshot())
                time.sleep(0.1)

    # Cancel any pending head movements when body motion completes
    robot.cancel_head_goals()
    time.sleep(0.2)  # Brief pause for cancellation to take effect

    if robot.get_motion_result():
        return True, "Motion successful with replanning."
    else:
        return False, "Motion failed during execution."


def move_arm_with_replanning(
    robot,
    target_joints,
    enable_gaze_control=True,
    enable_replanning=True,
    lookahead_window=20,
    decay_rate=0.999,
    max_replan_attempts=3,
    duration=5.0,
):
    """
    Move arm to target joint configuration with replanning and optional gaze control.

    Args:
        robot: Fetch robot instance
        target_joints: Target 8-DOF configuration [torso + 7 arm joints]
        enable_gaze_control: Whether to enable gaze tracking (default: True)
        enable_replanning: Whether to enable collision monitoring and replanning (default: True)
        lookahead_window: Number of future waypoints for gaze (default: 20)
        decay_rate: Distance decay rate for gaze weighting (default: 0.95)
        max_replan_attempts: Maximum replanning attempts on collision (default: 3)
        duration: Execution duration estimate for timeout (default: 5.0s)

    Returns:
        Result from action client, or None on failure
    """
    assert len(target_joints) == 8, "Expected 8 joint values [torso + 7 arm joints]"

    # If replanning is disabled, just use the standard send_joint_values
    if not enable_replanning and not enable_gaze_control:
        return robot.send_joint_values(target_joints, duration=duration)

    # Track total replan attempts across entire execution (not reset per collision)
    total_replan_attempts = 0

    # Step 1: Get current configuration
    current_joints = robot.get_current_planning_joints()
    if current_joints is None:
        log.error("Failed to get current joint positions")
        return None

    current_base = robot.get_base_params()

    # Check if already at target
    if np.allclose(current_joints, target_joints, atol=0.03):
        log.info("Already at target configuration, skipping motion")
        return True

    # Step 2: Plan initial arm motion
    log.info("Planning initial arm motion...")
    log.info(f"Current config: {[round(v, 3) for v in current_joints]}")
    log.info(f"Target config: {[round(v, 3) for v in target_joints]}")

    # Set VAMP base to current robot base before planning
    robot.set_base_params(current_base[2], current_base[0], current_base[1])

    # Plan with VAMP
    arm_path = robot._plan_arm(current_joints, target_joints)
    if arm_path is None:
        log.error("Failed to plan initial arm motion")
        return None

    log.info(f"Initial plan generated with {len(arm_path)} waypoints")

    # --- Initialize Gaze Control ---
    gaze_optimizer = None
    if enable_gaze_control:
        # Convert 8-DOF arm trajectory to 11-DOF by prepending fixed base pose
        # Format: [x, y, theta, torso, 7 arm joints]
        base_pose = np.array([current_base[0], current_base[1], current_base[2]])
        base_pose_repeated = np.tile(base_pose, (len(arm_path), 1))
        arm_path_array = np.array(arm_path)
        whole_body_trajectory = np.hstack((base_pose_repeated, arm_path_array))

        gaze_cfg = getattr(robot, "gaze_config", {})
        # Prefer config values over function arguments if present
        la_window = gaze_cfg.get("lookahead_window", lookahead_window)
        d_rate = gaze_cfg.get("decay_rate", decay_rate)
        vel_weight = gaze_cfg.get("velocity_weight", 1.0)
        joint_p = gaze_cfg.get("joint_priorities", None)

        gaze_type = gaze_cfg.get("type", "default")

        if gaze_type == "finean":
            gaze_optimizer = FineanGazeOptimizer(
                robot,
                lookahead_window=la_window,
            )
        else:
            gaze_optimizer = GazeOptimizer(
                robot,
                lookahead_window=la_window,
                decay_rate=d_rate,
                velocity_weight=vel_weight,
                joint_priorities=joint_p,
            )
        gaze_optimizer.set_trajectory(whole_body_trajectory)

    # --- Execution with Replanning Loop ---
    log.info("Starting arm motion with collision monitoring...")

    # Start executing the initial trajectory
    robot.start_arm_trajectory_async(arm_path, duration)

    # Calculate timeout based on number of waypoints
    time_per_waypoint = 0.4  # seconds
    estimated_duration = len(arm_path) * time_per_waypoint
    timeout = 2.0 * estimated_duration
    start_time = time.time()

    while True:
        try:
            if rospy.is_shutdown():
                break
        except Exception:
            pass
        # 1. Update collision environment
        if enable_replanning:
            _update_collision_environment(robot, enable_pcd_alignment=False)

        # 2. Get current waypoint index (based on current joint positions)
        current_joints = robot.get_current_planning_joints()
        # Find closest waypoint to current configuration
        distances = [
            np.linalg.norm(np.array(current_joints) - np.array(wp)) for wp in arm_path
        ]
        current_waypoint_index = np.argmin(distances)

        # 3. Update gaze to look ahead
        if gaze_optimizer is not None:
            gaze_optimizer.update(current_waypoint_index)

        # 4. Check remaining path for collisions
        base_configs = [list[float](current_base)] * len(arm_path)

        # Downsample for faster verification
        num_points = len(arm_path)
        max_verification_points = 128
        if num_points > max_verification_points:
            step = max(1, num_points // max_verification_points)
            arm_path_for_check = [arm_path[i] for i in range(0, num_points, step)]
            base_configs_for_check = base_configs[::step]
            waypoint_index_for_check = current_waypoint_index // step
        else:
            arm_path_for_check = arm_path
            base_configs_for_check = base_configs
            waypoint_index_for_check = current_waypoint_index

        is_collision = robot.check_plan_for_collisions(
            arm_path_for_check, base_configs_for_check, waypoint_index_for_check
        )

        # 4b. Check for physics collision from simulator monitoring
        if hasattr(robot, "robot_env") and hasattr(
            robot.robot_env, "get_monitoring_results"
        ):
            physics_collision, _ = robot.robot_env.get_monitoring_results()
            if physics_collision:
                log.error("Physics collision detected by simulator! Aborting motion.")
                robot.cancel_arm_goals()
                robot.cancel_torso_goals()
                robot.cancel_head_goals()
                return None

        # 5. Check for timeout
        elapsed = time.time() - start_time
        if elapsed > timeout:
            log.warning(f"Arm motion timed out after {elapsed:.2f}s")
            robot.cancel_arm_goals()
            robot.cancel_torso_goals()
            robot.cancel_head_goals()
            return None

        # 6. Handle collision detection and replanning
        if is_collision:
            log.warning("Collision detected in arm path! Stopping and replanning...")
            # Stop current motion
            robot.cancel_arm_goals()
            robot.cancel_torso_goals()

            # Check if we've exhausted all replan attempts
            if total_replan_attempts >= max_replan_attempts:
                log.error(
                    f"Failed to replan after {max_replan_attempts} total attempts"
                )
                return None

            replan_success = False
            while total_replan_attempts < max_replan_attempts:
                total_replan_attempts += 1
                log.info(
                    f"Replanning attempt {total_replan_attempts}/{max_replan_attempts}"
                )

                # Get fresh collision environment
                _update_collision_environment(robot, enable_pcd_alignment=False)

                # Replan from current position to goal
                current_joints = robot.get_current_planning_joints()

                arm_path = robot._plan_arm(current_joints, target_joints)
                if arm_path is None:
                    log.warning(f"Arm replan attempt {total_replan_attempts} failed")
                    continue

                log.info(f"Arm replan attempt {total_replan_attempts} successful")

                # Update gaze optimizer with new trajectory
                base_pose_repeated = np.tile(
                    np.array([current_base[0], current_base[1], current_base[2]]),
                    (len(arm_path), 1),
                )
                whole_body_trajectory = np.hstack(
                    (base_pose_repeated, np.array(arm_path))
                )
                if gaze_optimizer is not None:
                    gaze_optimizer.set_trajectory(whole_body_trajectory)

                # Execute replanned trajectory
                robot.start_arm_trajectory_async(arm_path, duration)

                # Reset timeout for the new path
                estimated_duration = len(arm_path) * time_per_waypoint
                timeout = 2.0 * estimated_duration
                start_time = time.time()
                replan_success = True
                break

            if not replan_success:
                log.error(f"Failed to replan after {max_replan_attempts} attempts")
                return None

        # 7. Check if execution is complete
        # Use numeric constants (compatible with actionlib.GoalStatus):
        # 2 = PREEMPTED, 3 = SUCCEEDED, 4 = ABORTED
        PREEMPTED, SUCCEEDED, ABORTED = 2, 3, 4

        arm_state = robot.get_arm_action_state()

        if arm_state in [SUCCEEDED, ABORTED, PREEMPTED]:
            # Cancel any pending head movements when arm motion completes
            robot.cancel_head_goals()
            time.sleep(0.2)  # Brief pause for cancellation to take effect

            if arm_state == SUCCEEDED:
                # Double check we actually reached the goal
                current_joints = robot.get_current_planning_joints()
                log.info("Successfully reached target configuration")
                return robot.get_arm_action_result()
            elif arm_state in [ABORTED, PREEMPTED]:
                log.warning(f"Motion was aborted/preempted (state: {arm_state})")
                return None
        # 8. Continue monitoring, replanning rate: 1 Hz, but update gaze at 10 Hz
        for _ in range(5):
            if robot.get_arm_action_state() in [SUCCEEDED, ABORTED, PREEMPTED]:
                break
            current_joints = robot.get_current_planning_joints()
            if current_joints is not None:
                distances = [
                    np.linalg.norm(np.array(current_joints) - np.array(wp))
                    for wp in arm_path
                ]
                current_waypoint_index = np.argmin(distances)
                if gaze_optimizer is not None:
                    gaze_optimizer.update(current_waypoint_index)

            _maybe_record_episode(robot, robot.robot_env.get_sensor_snapshot())
            time.sleep(0.1)
