#!/usr/bin/env python3
import time

import numpy as np

from grasp_anywhere.robot import Fetch
from grasp_anywhere.robot.ik import ikfast_compute_ik, trac_solve_arm_only
from grasp_anywhere.robot.utils.transform_utils import (
    extract_position_orientation,
    transform_pose_to_base,
    transform_pose_to_world,
)
from grasp_anywhere.utils import reachability_utils
from grasp_anywhere.utils.logger import log
from grasp_anywhere.utils.visualization_utils import visualize_grasp_pose_world


class GraspPlanner:
    def __init__(
        self,
        robot: Fetch,
        grasp_depth_offset=0.11,  # meters
        max_ik_attempts=20,
        max_replan_attempts=5,
        enable_visualization=False,  # Control grasping visualization
    ):
        self.robot = robot
        self.grasp_depth_offset = grasp_depth_offset
        self.max_ik_attempts = max_ik_attempts
        self.max_replan_attempts = max_replan_attempts
        self.enable_visualization = enable_visualization

        if self.grasp_depth_offset != 0:
            log.info(f"Applying a grasp depth offset of {self.grasp_depth_offset}m.")
        if not self.enable_visualization:
            log.info("Grasping visualization disabled.")

    def filter_reachable_grasps(self, grasp_poses_world, threshold=0.04):
        """
        Filter grasp poses based on the robot's capability map (reachability).
        Args:
            grasp_poses_world: List of 4x4 numpy arrays (poses in world frame)
            threshold: Minimum score to consider reachable
        Returns:
            List of filtered grasp poses
        """
        if (
            not hasattr(self.robot, "capability_map")
            or self.robot.capability_map is None
        ):
            log.warning("Capability map not loaded/available. Skipping filter.")
            return grasp_poses_world

        base_config = self.robot.get_base_params()
        filtered_poses = []
        total = len(grasp_poses_world)

        for pose in grasp_poses_world:
            # pose is 4x4 matrix
            position, orientation = extract_position_orientation(pose)
            # [x, y, z, qx, qy, qz, qw]
            target_pose_7d = position + orientation
            score = reachability_utils.query_capability_score(
                self.robot.capability_map, base_config, target_pose_7d
            )
            if score > threshold:
                filtered_poses.append(pose)

        log.info(
            f"Capability Filter: Kept {len(filtered_poses)}/{total} grasps (threshold={threshold})."
        )
        return filtered_poses

    def execute_grasp(
        self,
        top_grasp_world_matrix,
        camera_extrinsic,
        collision_points,
        use_active_perception=True,
    ):
        # Open gripper before grasping
        log.info("Opening gripper before grasp execution.")
        self.robot.control_gripper(1.0)
        time.sleep(1.0)

        # Visualize grasp pose in pointcloud if visualization is enabled
        if self.enable_visualization:
            log.info("Visualizing grasp pose in pointcloud...")
            try:
                # Get current camera data for visualization
                rgb = self.robot.get_rgb()
                depth = self.robot.get_depth()
                K = self.robot.get_camera_intrinsic()

                if rgb is not None and depth is not None and K is not None:
                    visualize_grasp_pose_world(
                        grasp_pose_world=top_grasp_world_matrix,
                        rgb=rgb,
                        depth=depth,
                        K=K,
                        camera_pose=camera_extrinsic,
                    )
                    log.info("Grasp pose visualization completed.")
                else:
                    log.warning("Warning: Could not get camera data for visualization.")
            except Exception as e:
                log.warning(f"Warning: Failed to visualize grasp pose: {e}")

        # Motion to pre-grasp

        # pre pose 1
        pre_grasp_matrix = top_grasp_world_matrix.copy()
        approach_vector = pre_grasp_matrix[:3, 0]
        pre_grasp_matrix[:3, 3] -= 0.10 * approach_vector
        position, orientation = extract_position_orientation(pre_grasp_matrix)

        # pre pose 2 - top down pose
        # Transform from end-effector frame back to grasp frame
        T_ee_to_grasp = np.array(
            [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
        )
        grasp_frame_matrix = top_grasp_world_matrix @ T_ee_to_grasp

        # In grasp frame: x=approach, y=closing, z=normal
        # Create top-down pose in grasp frame
        pre_grasp_grasp_frame = grasp_frame_matrix.copy()
        # Keep approach direction (x) and closing direction (y) from original grasp
        approach = pre_grasp_grasp_frame[:3, 0]  # x in grasp frame
        closing = pre_grasp_grasp_frame[:3, 1]  # y in grasp frame
        # Force normal to be straight down
        normal = np.array([0, 0, -1])  # z in grasp frame

        # Ensure orthogonality
        closing = np.cross(normal, approach)
        closing = closing / np.linalg.norm(closing)
        approach = np.cross(closing, normal)
        approach = approach / np.linalg.norm(approach)

        # Update rotation matrix in grasp frame
        pre_grasp_grasp_frame[:3, 0] = approach
        pre_grasp_grasp_frame[:3, 1] = closing
        pre_grasp_grasp_frame[:3, 2] = normal

        # Move up by 0.10 meters in grasp frame (along normal direction)
        pre_grasp_grasp_frame[:3, 3] -= 0.10 * normal

        # Transform back to end-effector frame
        T_grasp_to_ee = np.array(
            [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
        )
        pre_grasp_matrix2 = pre_grasp_grasp_frame @ T_grasp_to_ee

        position2, orientation2 = extract_position_orientation(pre_grasp_matrix2)

        base_config = self.robot.get_base_config_from_camera_pose(camera_extrinsic)
        if base_config is None:
            log.warning(
                "WARNING: Could not update base config from camera pose. Using default from TF."
            )
            base_config = self.robot.get_base_params()

        torso_pos = self.robot.get_torso_position()
        arm_seed = self.robot.get_arm_joint_values()

        # Visualize pre grasp pose in pointcloud if visualization is enabled
        if self.enable_visualization:
            log.info("Visualizing pre grasp pose in pointcloud...")
            try:
                # Get current camera data for visualization
                rgb = self.robot.get_rgb()
                depth = self.robot.get_depth()
                K = self.robot.get_camera_intrinsic()

                if rgb is not None and depth is not None and K is not None:
                    visualize_grasp_pose_world(
                        grasp_pose_world=pre_grasp_matrix2,
                        rgb=rgb,
                        depth=depth,
                        K=K,
                        camera_pose=camera_extrinsic,
                    )
                    log.info("Pre grasp pose visualization completed.")
                else:
                    log.warning("Warning: Could not get camera data for visualization.")
            except Exception as e:
                log.warning(f"Warning: Failed to visualize pre grasp pose: {e}")

        # Try pre pose 1 first, if it fails, try pre pose 2
        pre_grasp_joints = trac_solve_arm_only(
            arm_seed, base_config, torso_pos, position, orientation
        )
        if pre_grasp_joints is None:
            pre_grasp_joints = trac_solve_arm_only(
                arm_seed, base_config, torso_pos, position2, orientation2
            )
        if pre_grasp_joints:
            log.info(
                "Found IK solution for pre-grasp with fixed base. Executing arm motion."
            )
            result = self.robot.send_joint_values_with_replanning(
                [torso_pos] + pre_grasp_joints,
                duration=7.0,
                enable_replanning=use_active_perception,
                enable_gaze_control=use_active_perception,
            )
            time.sleep(2.0)
            if result is None:
                log.info("Failed to execute pre-grasp motion.")
                return (False, "Failed to execute pre-grasp motion.")
        else:
            log.info(
                "Fixed-base IK failed; attempting IKFast with collision checking..."
            )
            self.robot.set_base_params(base_config[2], base_config[0], base_config[1])

            cand_pos_ori = [(position, orientation), (position2, orientation2)]
            base_pos = [base_config[0], base_config[1], 0]
            base_yaw = base_config[2]

            valid_solution = None
            for _ in range(20):
                for pos_w, quat_w in cand_pos_ori:
                    ee_pos, ee_quat = transform_pose_to_base(
                        pos_w, quat_w, base_pos, base_yaw
                    )
                    from scipy.spatial.transform import Rotation as R

                    rot3 = R.from_quat(ee_quat).as_matrix()
                    sols = ikfast_compute_ik([ee_pos, rot3]) or []
                    for sol in sols:
                        if sol is None or len(sol) != 8:
                            continue
                        if not self.robot.vamp_module.validate(
                            sol, self.robot.planning_env
                        ):
                            continue
                        valid_solution = sol
                        break
                    if valid_solution is not None:
                        break
                if valid_solution is not None:
                    break

            if valid_solution is None:
                return (
                    False,
                    "Failed to find IK solution for pre-grasp with IKFast.",
                )

            result = self.robot.send_joint_values_with_replanning(
                valid_solution,
                duration=3.0,
                enable_replanning=use_active_perception,
                enable_gaze_control=use_active_perception,
            )
            time.sleep(2.0)
            if result is None:
                return (False, "Failed to execute pre-grasp motion with IKFast.")

        pre_grasp_joints = self.robot.get_arm_joint_values()

        # Reset collision scene
        log.info("Resetting collision scene to initial static environment.")
        self.robot.clear_pointclouds()
        self.robot.add_pointcloud(collision_points, point_radius=0.01)

        # Motion to final grasp
        grasp_position, grasp_orientation = extract_position_orientation(
            top_grasp_world_matrix
        )

        # Use Cartesian interpolation for smooth straight-line end-effector motion
        log.info("Executing Cartesian interpolation to final grasp pose.")
        result = self.robot.send_cartesian_interpolated_motion(
            grasp_position, grasp_orientation, duration=3.0, num_waypoints=20
        )
        if result is None:
            return False, "Failed to execute Cartesian motion to grasp pose."
        time.sleep(0.5)

        # Close gripper
        self.robot.control_gripper(0.0)
        time.sleep(0.5)

        # Return to pre-grasp pose to lift the object
        log.info("Lifting object with Cartesian interpolation.")

        # Calculate lift target (Cartesian) from pre_grasp_joints (which are valid and reachable)
        pre_grasp_full_config = [torso_pos] + pre_grasp_joints
        pg_ee_pos_b, pg_ee_quat_b = self.robot.vamp_module.eefk(pre_grasp_full_config)
        target_pre_pos, target_pre_quat = transform_pose_to_world(
            [base_config[0], base_config[1], 0],
            base_config[2],
            pg_ee_pos_b,
            pg_ee_quat_b,
        )

        result = self.robot.send_cartesian_interpolated_motion(
            target_pre_pos, target_pre_quat, duration=3.0, num_waypoints=20
        )
        if result is None:
            log.warning(
                "Warning: Failed to execute smooth lift motion, but object may be grasped."
            )
        time.sleep(0.5)

        # Reset collision scene and add grasped object attachment
        log.info("Resetting collision scene and adding grasped object attachment.")
        self.robot.clear_pointclouds()
        self.robot.add_pointcloud(collision_points, point_radius=0.1)

        # Attach sphere to end-effector to represent grasped object
        sphere_params = [
            {"position": [0.05, 0, 0], "radius": 0.05}
        ]  # 5cm forward, 5cm radius
        if self.robot.attach_objects_to_eef(sphere_params):
            log.info("Successfully attached grasped object sphere to end-effector.")
        else:
            log.warning("WARNING: Failed to attach grasped object sphere.")

        return True, "Direct grasp successful."

    def run(
        self,
        grasp_pose_world,
        camera_pose,
        collision_points,
        use_active_perception=True,
    ):
        print("=== Stage 2: Grasping started ===")

        # Visualize original grasp pose in pointcloud if visualization is enabled
        if self.enable_visualization:
            print("Visualizing original grasp pose in pointcloud...")
            try:
                # Get current camera data for visualization
                rgb = self.robot.get_rgb()
                depth = self.robot.get_depth()
                K = self.robot.get_camera_intrinsic()

                if rgb is not None and depth is not None and K is not None:
                    visualize_grasp_pose_world(
                        grasp_pose_world=grasp_pose_world,
                        rgb=rgb,
                        depth=depth,
                        K=K,
                        camera_pose=camera_pose,
                    )
                    print("Original grasp pose visualization completed.")
                else:
                    print("Warning: Could not get camera data for visualization.")
            except Exception as e:
                print(f"Warning: Failed to visualize original grasp pose: {e}")

        if self.grasp_depth_offset != 0:
            print(f"Adjusting grasp pose forward by {self.grasp_depth_offset}m.")
            approach_vector = grasp_pose_world[:3, 2]
            grasp_pose_world[:3, 3] += approach_vector * self.grasp_depth_offset

        # Transformation from grasp frame to end-effector frame
        T_grasp_to_ee = np.array(
            [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
        )
        top_grasp_ee_world = grasp_pose_world @ T_grasp_to_ee

        return self.execute_grasp(
            top_grasp_ee_world,
            camera_pose,
            collision_points,
            use_active_perception=use_active_perception,
        )
