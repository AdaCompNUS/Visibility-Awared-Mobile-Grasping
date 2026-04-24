from __future__ import annotations

import os
from time import time

import numpy as np
import open3d as o3d
import rospy
import vamp
import yaml
from scipy.spatial.transform import Rotation as R
from sensor_msgs import point_cloud2
from sensor_msgs.msg import (
    PointCloud2,
    PointField,
)
from std_msgs.msg import Header

import grasp_anywhere.robot.utils.replanning_utils as replanning_utils
import grasp_anywhere.robot.utils.transform_utils as transform_utils
from grasp_anywhere.dataclass.datacollector.config import DataCollectionConfig
from grasp_anywhere.envs.base import RobotEnv
from grasp_anywhere.envs.fetch_realrobot.fetch_real_env import FetchRealEnv
from grasp_anywhere.observation.scene import Scene
from grasp_anywhere.robot.ik import (
    ikfast_compute_ik,
)
from grasp_anywhere.robot.ik.ikfast_api import JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER
from grasp_anywhere.robot.kinematics import (
    _create_transform_matrix,
    forward_kinematics,
)
from grasp_anywhere.robot.utils.whole_body_planners import (
    plan_base_only,
    plan_fcit_wb_whole_body,
    plan_rrtc_whole_body,
)
from grasp_anywhere.samplers.base_sampler import BaseSampler
from grasp_anywhere.utils.logger import log
from grasp_anywhere.utils.visualization_utils import (
    init_pyrender_viewer,
    pyrender_is_available,
    pyrender_set_static_pointcloud,
    pyrender_update_pointcloud,
    show_costmap,
)


class Fetch:
    """
    Core class for controlling the Fetch robot.
    """

    def __init__(
        self,
        config_path="grasp_anywhere/configs/real_fetch.yaml",
        scene_mode: str = "combine",
        robot_env: RobotEnv = None,
        costmap_path: str = None,
        static_pcd_paths: list = None,
    ):
        """
        Initialize the Fetch robot interface.

        Args:
            config_path: Path to the config file
            scene_mode: Scene observation mode ("accumulated", "combine", "static", "latest" or "ray_casting")
            robot_env: Robot environment interface (if None, uses FetchRealEnv)
            costmap_path: Path to the costmap .npz file (if None, uses config)
            static_pcd_paths: List of paths to static .ply files (if None, uses config)
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        robot_config = config.get("robot", {})
        env_config = config.get("environment", {})
        debug_config = config.get("debug", {})
        planning_config = config.get("planning", {})
        self.gaze_config = config.get("gaze", {})

        # Use parameter if provided, otherwise fallback to config
        costmap_path = robot_config.get("costmap_path", None)

        # Use parameter if provided, otherwise fallback to config
        if static_pcd_paths is None:
            pcd_path = env_config.get("pcd_path", "")
            pcd_files = env_config.get("pcd_files", [])
            static_collision_pcd_paths = [os.path.join(pcd_path, f) for f in pcd_files]
        else:
            static_collision_pcd_paths = static_pcd_paths

        enable_debug = debug_config.get("enable_debug", False)
        self.enable_visualization = debug_config.get("enable_visualization", False)

        self.planning_joint_names = [
            "torso_lift_joint",  # First joint is torso
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "upperarm_roll_joint",
            "elbow_flex_joint",
            "forearm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
        ]
        if robot_env is None:
            self.robot_env = FetchRealEnv(self.planning_joint_names)
        else:
            self.robot_env = robot_env

        self.is_sim = not isinstance(self.robot_env, FetchRealEnv)

        # Load raw static collision PCDs
        static_pcd_combined = np.empty((0, 3), dtype=np.float32)
        if static_collision_pcd_paths:
            for pcd_path in static_collision_pcd_paths:
                try:
                    pcd = o3d.io.read_point_cloud(pcd_path)
                    if len(pcd.points) > 0:
                        points = np.asarray(pcd.points, dtype=np.float32)
                        static_pcd_combined = np.vstack((static_pcd_combined, points))
                except Exception as e:
                    log.warning(f"Could not load {pcd_path}: {e}")

        # Initialize the scene manager, which will handle downsampling
        # Scene uses synchronized filtering method that requires joint_dict
        # Allow config to override default scene_mode
        scene_mode = env_config.get("scene_mode", scene_mode)
        self.scene = Scene(
            static_map_pcd=static_pcd_combined,
            downsample_voxel_size=0.05,  # Voxel size from action_interface.py
            mode=scene_mode,
            robot_filter=self.filter_points_on_robot_with_state,
            ground_z_threshold=env_config.get("ground_z_threshold", 0.3),
        )
        log.info(f"Scene manager initialized in '{scene_mode}' mode.")

        # Optional: active episode recording state (used by motion_utils + scheduler)
        self._active_episode_cfg: DataCollectionConfig | None = None
        self._active_episode_ds = None
        self._active_episode_goal_xyz_world = None

        # Control parameters
        self.control_rate = 10
        if not self.is_sim:
            self.rate = rospy.Rate(self.control_rate)

        # Movement limits
        self.max_linear_speed = 0.5  # m/s
        self.max_angular_speed = 1.0  # rad/s

        self.planner = "rrtc"  # ["rrtc", "fcit", "prm"]
        # Whole-body planner option: "rrtc" (default) or "fcit_wb"
        self.whole_body_planner = "rrtc"
        # Bounds over XY for FCIT*, updated when adding pointclouds
        self._pc_bounds_xy = None  # tuple (x_min, x_max, y_min, y_max)
        self._bounds_padding = 0.1

        # Initialize VAMP planner
        self._init_vamp_planner()

        # Initialize attachment tracking
        self._current_attachment = None

        # Initialize base sampler (TRAC-IK auto-initializes internally)
        dynamic_costmap = bool(planning_config.get("dynamic_costmap", False))
        self.base_sampler = BaseSampler(
            costmap_path=costmap_path, dynamic_costmap=dynamic_costmap
        )
        if self.enable_visualization and self.base_sampler.costmap is not None:
            show_costmap(self.base_sampler.costmap)

        # Store debug flag
        self.enable_debug = enable_debug

        # Publisher for debugging point clouds (only if debug is enabled)
        if self.enable_debug and not self.is_sim:
            self.pointcloud_publisher = rospy.Publisher(
                "/debug_pointcloud", PointCloud2, queue_size=1
            )
        else:
            self.pointcloud_publisher = None
        # Initialize Pyrender viewer (ROS-independent) when in debug mode
        if self.enable_debug:
            if init_pyrender_viewer():
                if (
                    isinstance(static_pcd_combined, np.ndarray)
                    and static_pcd_combined.size > 0
                    and static_pcd_combined.shape[1] == 3
                ):
                    pyrender_set_static_pointcloud(static_pcd_combined)
        # Approx. forward from head
        self.camera_mount_offset = np.array([0.0443, 0.0, 0.0394])

    def sample_wb_ik(
        self,
        position,
        orientation,
        max_attempts=20,
        manipulation_radius=0.5,
    ):
        """
        Solve whole-body inverse kinematics for a given target pose.

        This method attempts to find a valid (collision-free) configuration for the whole robot
        (base + arm) that places the end effector at the desired pose.

        Args:
            position: [x, y, z] position as list/numpy array
            orientation: [x, y, z, w] quaternion as list/numpy array
            max_attempts: Maximum number of sampling attempts
            manipulation_radius: Radius for base position sampling

        Returns:
            dict: Solution containing base and arm configuration, or None if no solution found
        """
        # Fixed-base strategy using IKFast with internal free-param handling
        attempts = 0
        while attempts < max_attempts and not rospy.is_shutdown():
            attempts += 1
            target_point = [float(position[0]), float(position[1]), float(position[2])]
            if self.base_sampler.dynamic_costmap:
                env_pcd = self.scene.current_environment()
                self.base_sampler.update_from_pointcloud(env_pcd)
                if self.enable_visualization and self.base_sampler.costmap is not None:
                    show_costmap(self.base_sampler.costmap)
            # base_config is [x, y, theta] (3-DoF)
            base_config = self.base_sampler.sample_base_pose(
                target_point, manipulation_radius=manipulation_radius
            )

            # check None
            if base_config is None:
                log.warning("Failed to sample base pose")
                continue

            # World -> base frame
            base_pos = [base_config[0], base_config[1], 0.0]
            base_yaw = base_config[2]
            ee_pos_b, ee_quat_b = transform_utils.transform_pose_to_base(
                list(position), list(orientation), base_pos, base_yaw
            )
            # Convert quaternion to rotation matrix (base frame)
            rot_mat_b = R.from_quat(ee_quat_b).as_matrix()

            # IKFast returns full 8-DOF solutions (torso + 7 joints)
            # solutions is list of [torso, shoulder_pan, shoulder_lift, ..., wrist_roll]
            solutions = ikfast_compute_ik([ee_pos_b, rot_mat_b])
            if not solutions:
                continue

            # VAMP set_base_params takes (theta, x, y)
            self.vamp_module.set_base_params(
                base_config[2], base_config[0], base_config[1]
            )
            for sol in solutions:
                if sol is None or len(sol) != 8:
                    continue
                # if not self.vamp_module.validate(sol, self.planning_env):
                #     continue
                # if not self.vamp_module.validate(
                #     [0.3, 1.32, 1.4, -0.2, 1.72, 0.0, 1.66, 0.0], self.planning_env
                # ):
                #     continue

                # Validate whole body config: arm_config=sol (8-dof), base_config=[x, y, theta]
                if not self.vamp_module.validate_whole_body_config(
                    sol, base_config, self.planning_env
                ):
                    continue
                if not self.vamp_module.validate_whole_body_config(
                    [0.3, 1.32, 1.4, -0.2, 1.72, 0.0, 1.66, 0.0],
                    base_config,
                    self.planning_env,
                ):
                    continue
                full_solution = list[float](base_config) + list[float](sol)
                return {
                    "base_config": list[float](base_config),  # [x, y, theta]
                    "arm_config": list[float](sol),  # [torso, 7 arm joints]
                    "full_solution": full_solution,
                    "attempts": attempts,
                }

        return None

    def sample_ik(
        self,
        position,
        orientation,
        base_config,
        torso_lift=None,
    ):
        """
        Solve whole-body inverse kinematics for a given target ee pose and base config.
        (base + arm) that places the end effector at the desired pose.

        Args:
            position: [x, y, z] position as list/numpy array
            orientation: [x, y, z, w] quaternion as list/numpy array
            base_config: [x, y, theta] base configuration as list/numpy array

        Returns:
            list: List of solutions containing base and arm configuration, or None if no solution found
        """
        # World -> base frame
        base_pos = [base_config[0], base_config[1], 0.0]
        base_yaw = base_config[2]
        ee_pos_b, ee_quat_b = transform_utils.transform_pose_to_base(
            list(position), list(orientation), base_pos, base_yaw
        )
        # Convert quaternion to rotation matrix (base frame)
        rot_mat_b = R.from_quat(ee_quat_b).as_matrix()

        # IKFast returns full 8-DOF solutions (torso + 7 joints)
        # solutions is list of [torso, shoulder_pan, shoulder_lift, ..., wrist_roll]
        if torso_lift is None:
            return ikfast_compute_ik([ee_pos_b, rot_mat_b])

        torso_lift = float(torso_lift)
        shoulder_lift = float(
            np.random.uniform(JOINT_LIMITS_LOWER[2], JOINT_LIMITS_UPPER[2])
        )
        return ikfast_compute_ik(
            [ee_pos_b, rot_mat_b], free_params=[torso_lift, shoulder_lift]
        )

    def validate_whole_body_config(self, arm_config, base_config):
        """
        Validate a whole-body configuration for collisions.

        Args:
            arm_config: 8-DoF arm configuration [torso, 7 arm joints]
            base_config: 3-DoF base configuration [x, y, theta]

        Returns:
            bool: True if IN COLLISION (invalid), False if VALID (collision-free)
            Note: This return value convention is confusing!
            It returns `not validate(...)`, and validate() usually returns True for valid.
            So this function returns True if INVALID.
        """
        # self.vamp_module.set_base_params(base_config[2], base_config[0], base_config[1]) # [theta, x, y]
        # return not self.vamp_module.validate(arm_config, self.planning_env)
        return not self.vamp_module.validate_whole_body_config(
            arm_config, base_config, self.planning_env
        )

    def get_end_effector_pose(self, joint_values=None, base_config=None):
        """
        Get the end effector pose for a given configuration.

        If joint_values and base_config are not provided, it uses the current robot state.

        Args:
            joint_values: Optional list of 8 joint values (torso + 7 arm joints)
            base_config: Optional [x, y, theta] base configuration

        Returns:
            pose: End effector pose as 4x4 numpy transformation matrix in world frame
        """
        # Use current configuration if not provided
        if joint_values is None:
            joint_values = self.get_current_planning_joints()
            if joint_values is None:
                log.error("Failed to get current joint positions")
                return None

        if base_config is None:
            base_config = self.get_base_params()

        # Call the eefk function to get the end effector pose in robot frame
        ee_pos, ee_quat = self.vamp_module.eefk(joint_values)

        # Transform to world frame using the utility function
        world_pos, world_quat = transform_utils.transform_pose_to_world(
            [base_config[0], base_config[1], 0], base_config[2], ee_pos, ee_quat
        )

        # Create 4x4 transformation matrix
        pose_matrix = transform_utils.create_pose_matrix(world_pos, world_quat)

        return pose_matrix

    def _execution_finished_callback(self, msg):
        """Callback for the whole body execution finished signal."""
        if self.motion_state == "RUNNING":
            if msg.data:
                self.motion_state = "SUCCEEDED"
            else:
                self.motion_state = "FAILED"
            log.debug(f"Motion execution finished callback: success={msg.data}")

    def _init_vamp_planner(self):
        """
        Initialize VAMP motion planner with 8-DOF configuration and collision settings.

        This function sets up the planning environment and configures the motion planner
        with appropriate parameters for collision avoidance.
        """
        self.planning_env = vamp.Environment()

        # Configure robot and planner with custom settings
        (
            self.vamp_module,
            self.planner_func,
            self.plan_settings,
            self.simp_settings,
        ) = vamp.configure_robot_and_planner_with_kwargs(
            "fetch",  # Robot name
            self.planner,  # Planner algorithm (Rapidly-exploring Random Tree Connect)
            sampler_name="halton",  # Use Halton sampler for better coverage
        )

        # Initialize the sampler
        self.sampler = self.vamp_module.halton()
        self.sampler.skip(0)  # Skip initial samples if needed

        log.info("VAMP planner initialized with collision avoidance settings")

    def set_base_params(self, theta, x, y):
        """
        Set the base parameters for the Fetch robot.

        Args:
            theta (float): Base rotation around z-axis in radians
            x (float): Base x position in meters
            y (float): Base y position in meters
        """
        self.base_theta = theta
        self.base_x = x
        self.base_y = y

        # Update the base parameters in the VAMP module
        self.vamp_module.set_base_params(theta, x, y)

        return True

    def get_base_params(self, world_frame="map", robot_base_frame="base_link"):
        """
        Get the current base parameters from the ROS TF tree.
        This is now the single source of truth for the robot's current pose.
        Args:
            world_frame (str): The name of the fixed world frame (e.g., 'map').
            robot_base_frame (str): The name of the robot's base frame (e.g., 'base_link').
        Returns:
            tuple: (x, y, theta) current base parameters.
        """
        return self.robot_env.get_base_pose(world_frame, robot_base_frame)

    def get_base_config_from_camera_pose(
        self,
        camera_extrinsic,
    ):
        """
        Calculates the robot's base_link pose from a refined camera pose using numpy.
        Args:
            camera_extrinsic (np.ndarray): 4x4 transformation matrix of the camera's pose in the world (T_wc).
        Returns:
            tuple: The new base configuration (x, y, theta) or None if transform lookup fails.
        """
        # Get the transform from the robot base frame to the camera frame using FK.
        # This represents T_bc, the transform from base to camera.
        joint_states = self.get_current_planning_joints_with_head()
        if joint_states is None:
            log.error("Could not get current joint states for FK.")
            return None

        link_poses = forward_kinematics(joint_states)
        # Assuming camera is attached to head_tilt_link with some offset
        T_base_head_tilt = link_poses["head_tilt_link"]
        T_head_tilt_camera = _create_transform_matrix(
            self.camera_mount_offset,
            R.from_euler("xyz", [-np.pi / 2, 0, -np.pi / 2]).as_matrix(),
        )
        T_base_camera = T_base_head_tilt @ T_head_tilt_camera

        # The camera extrinsic is T_world_camera. We need T_world_base.
        # T_world_base = T_world_camera * T_camera_base
        T_camera_base = np.linalg.inv(T_base_camera)
        new_base_pose_matrix = camera_extrinsic @ T_camera_base

        # Extract the new base configuration from the resulting matrix
        x = new_base_pose_matrix[0, 3]
        y = new_base_pose_matrix[1, 3]

        # Get yaw from the rotation matrix
        base_rot = R.from_matrix(new_base_pose_matrix[:3, :3])
        yaw = base_rot.as_euler("xyz", degrees=False)[2]

        return (x, y, yaw)

    def get_current_planning_joints(self):
        """Get current joint positions for planning (8-DOF including torso)."""
        joint_states = self.robot_env.get_joint_states()
        if joint_states is None:
            return None
        joint_names, joint_positions = joint_states
        joint_dict = dict(zip(joint_names, joint_positions))

        positions = []
        for joint_name in self.planning_joint_names:
            if joint_name in joint_dict:
                positions.append(joint_dict[joint_name])
            else:
                log.error(
                    f"Joint {joint_name} not found in joint states, The current joints are: {joint_dict}"
                )
                return None

        return positions

    def get_current_planning_joints_with_head(self):
        """Get current joint positions for planning (10-DOF including torso and head)."""
        joint_states = self.robot_env.get_joint_states()
        if joint_states is None:
            return None
        joint_names, joint_positions = joint_states
        joint_dict = dict(zip(joint_names, joint_positions))

        planning_joint_names_with_head = self.planning_joint_names + [
            "head_pan_joint",
            "head_tilt_joint",
        ]
        positions = []
        for joint_name in planning_joint_names_with_head:
            if joint_name in joint_dict:
                positions.append(joint_dict[joint_name])
            else:
                log.error(
                    f"Joint {joint_name} not found in joint states, The current joints are: {joint_dict}"
                )
                return None

        return positions

    def get_arm_joint_values(self):
        """Get current joint positions for the 7-DOF arm."""
        planning_joints = self.get_current_planning_joints()
        if planning_joints is None:
            return None
        # The first joint is torso, the rest are arm joints
        return planning_joints[1:]

    def get_torso_position(self):
        """Get current torso position."""
        planning_joints = self.get_current_planning_joints()
        if planning_joints is None:
            return None
        # The first joint is torso
        return planning_joints[0]

    def get_camera_pose(self):
        """
        Get the current pose of a specified camera directly in the world frame using FK.
        Returns:
            numpy.ndarray: 4x4 transformation matrix representing the pose of the camera
                        in the world frame.
        """
        # Get T_map_base from TF
        base_x, base_y, base_theta = self.get_base_params()
        T_map_base = _create_transform_matrix(
            [base_x, base_y, 0], R.from_euler("z", base_theta).as_matrix()
        )

        # Get T_base_camera from FK
        joint_states = self.get_current_planning_joints_with_head()
        if joint_states is None:
            log.error("Could not get current joint states for FK.")
            return None

        link_poses = forward_kinematics(joint_states)
        # This transform is an approximation of the optical frame from the URDF.
        T_base_head_tilt = link_poses["head_tilt_link"]
        T_head_tilt_camera = _create_transform_matrix(
            self.camera_mount_offset,
            R.from_euler("xyz", [-np.pi / 2, 0, -np.pi / 2]).as_matrix(),
        )
        T_base_camera = T_base_head_tilt @ T_head_tilt_camera

        # Combine to get T_map_camera
        T_map_camera = T_map_base @ T_base_camera
        return T_map_camera

    def compute_camera_pose_from_joints(self, joint_dict):
        """
        Compute camera pose from a dictionary of joint states.
        Useful for computing pose from a snapshot of joints.

        Args:
            joint_dict: Dictionary mapping joint names to positions

        Returns:
            4x4 transformation matrix for camera pose in world frame
        """
        # Get base pose from joint dict
        base_joint_names = self.robot_env.agent.base_joint_names
        base_x = joint_dict[base_joint_names[0]]
        base_y = joint_dict[base_joint_names[1]]
        base_theta = joint_dict[base_joint_names[2]]

        # Apply base offset for world frame
        if hasattr(self.robot_env, "_qpos_base_offset"):
            offset = self.robot_env._qpos_base_offset
            base_x += offset[0]
            base_y += offset[1]
            base_theta += offset[2]

        T_map_base = _create_transform_matrix(
            [base_x, base_y, 0], R.from_euler("z", base_theta).as_matrix()
        )

        # Get planning joints with head for FK
        planning_joints_with_head = self.planning_joint_names + [
            "head_pan_joint",
            "head_tilt_joint",
        ]
        joint_values = np.array(
            [joint_dict[name] for name in planning_joints_with_head]
        )

        # Compute FK
        link_poses = forward_kinematics(joint_values)
        T_base_head_tilt = link_poses["head_tilt_link"]
        T_head_tilt_camera = _create_transform_matrix(
            self.camera_mount_offset,
            R.from_euler("xyz", [-np.pi / 2, 0, -np.pi / 2]).as_matrix(),
        )
        T_base_camera = T_base_head_tilt @ T_head_tilt_camera

        # Combine
        T_map_camera = T_map_base @ T_base_camera
        return T_map_camera

    def _plan_arm(self, current_joints, target_joints):
        """Plan a path using VAMP motion planner for 8-DOF configuration."""
        current_joints = np.array(current_joints, dtype=np.float64)
        target_joints = np.array(target_joints, dtype=np.float64)

        log.info("Planning with VAMP (8-DOF):")
        log.info(f"Start config values: {current_joints}")
        log.info(f"Goal config values: {target_joints}")

        result = self.planner_func(
            current_joints,
            target_joints,
            self.planning_env,
            self.plan_settings,
            self.sampler,
        )

        if result.solved:
            log.info("Path planning succeeded!")

            # Get planning statistics
            simple = self.vamp_module.simplify(
                result.path, self.planning_env, self.simp_settings, self.sampler
            )

            _ = vamp.results_to_dict(result, simple)

            # Interpolate path
            interpolate = 16
            simple.path.interpolate(interpolate)

            # Convert path to trajectory points
            trajectory_points = []
            for i in range(len(simple.path)):
                point = simple.path[i].to_list()
                trajectory_points.append(point)

            return trajectory_points
        else:
            return None

    def plan_whole_body_motion(
        self,
        start_joints,
        goal_joints,
        start_base,
        goal_base,
        planner: str = None,
        fcit_settings_overrides: dict = None,
    ):
        """
        Plan a whole-body motion.

        By default uses multilayer RRTC; can be switched to FCIT* via the
        `whole_body_planner` attribute or per-call `planner` argument.

        Args:
            start_joints: List of 8 joint positions for start configuration
            goal_joints: List of 8 joint positions for goal configuration
            start_base: List of 3 values [x, y, theta] for start base configuration
            goal_base: List of 3 values [x, y, theta] for goal base configuration
            planner: Optional override for planner selection ("rrtc" or "fcit_wb")
            fcit_settings_overrides: Optional dict of overrides for FCITSettings

        Returns:
            dict: Planning results with keys {success, stats, arm_path, base_configs}
        """
        # Validate input dimensions
        assert (
            len(start_joints) == 8 and len(goal_joints) == 8
        ), "Invalid joint dimensions. Expected 8, got {len(start_joints)} and {len(goal_joints)}"
        assert (
            len(start_base) == 3 and len(goal_base) == 3
        ), "Invalid base dimensions. Expected 3, got {len(start_base)} and {len(goal_base)}"

        start_joints = [round(val, 3) for val in start_joints]
        goal_joints = [round(val, 3) for val in goal_joints]
        start_base = [round(val, 3) for val in start_base]
        goal_base = [round(val, 3) for val in goal_base]

        log.info("Planning whole body motion:")
        log.info(f"Start arm config: {start_joints}")
        log.info(f"Goal arm config: {goal_joints}")
        log.info(f"Start base config: {start_base}")
        log.info(f"Goal base config: {goal_base}")

        # Check if the start and goal whole body configurations are in collision
        if self.validate_whole_body_config(start_joints, start_base):
            log.warning("Start whole body configuration is in collision")
            return False
        if self.validate_whole_body_config(goal_joints, goal_base):
            log.warning("Goal whole body configuration is in collision")
            return False

        planner_to_use = planner if planner is not None else self.whole_body_planner

        if planner_to_use == "fcit_wb":
            if self._pc_bounds_xy is None:
                log.warning(
                    "FCIT* selected but XY bounds are not available. Consider calling add_pointcloud first."
                )
            res = plan_fcit_wb_whole_body(
                start_joints,
                goal_joints,
                start_base,
                goal_base,
                self.planning_env,
                self.vamp_module,
                self._pc_bounds_xy,
                random_generator=self.sampler,
                settings_overrides=fcit_settings_overrides,
                interpolate_density=0.08,
            )
            # Print FCIT* time and stats similar to the example script
            stats = res.get("stats", {})
            if stats:
                time_ms = stats.get("arm_planning_time_ms")
                iters = stats.get("planning_iterations")
                graph = stats.get("planning_graph_size")
                if time_ms is not None:
                    log.info(
                        f"FCIT* Planning Time: {time_ms * 1000:.0f}μs | Iterations: {iters} | Graph size: {graph}"
                    )
            return res

        # Default path: multilayer RRTC
        return plan_rrtc_whole_body(
            start_joints,
            goal_joints,
            start_base,
            goal_base,
            self.planning_env,
            self.vamp_module,
            self.plan_settings,
            self.simp_settings,
            self.sampler,
            interpolate_density=0.03,
        )

    def plan_base_motion(
        self,
        start_base,
        goal_base,
        start_arm=None,
        settings_overrides: dict = None,
    ):
        """
        Plan a base-only motion using Hybrid A*.

        Args:
            start_base: [x, y, theta]
            goal_base: [x, y, theta]
            start_arm: 8-DOF arm config (defaults to current)
            settings_overrides: Optional dict for settings

        Returns:
            dict: Planning results with keys {success, stats, base_configs}
        """
        assert (
            len(start_base) == 3 and len(goal_base) == 3
        ), f"Invalid base dimensions. Expected 3, got {len(start_base)} and {len(goal_base)}"

        start_base = [round(val, 3) for val in start_base]
        goal_base = [round(val, 3) for val in goal_base]

        if start_arm is None:
            start_arm = self.get_current_planning_joints()

        log.info("Planning base-only motion:")
        log.info(f"Start base config: {start_base}")
        log.info(f"Goal base config: {goal_base}")

        return plan_base_only(
            start_base,
            goal_base,
            self.planning_env,
            self.vamp_module,
            self.simp_settings,
            self.sampler,
            start_arm,
            settings_overrides=settings_overrides,
        )

    def execute_whole_body_motion(self, arm_path, base_configs):
        """
        Execute a whole body motion plan with the Fetch robot (blocking).
        This method sends the planned trajectories to the C++ whole body controller
        for coordinated whole-body motion and waits for a completion signal.
        Args:
            arm_path: List of arm joint configurations (8-DOF including torso)
            base_configs: List of base configurations [x, y, theta]
        Returns:
            bool: True if execution succeeded, False otherwise
        """
        return self.robot_env.execute_whole_body_motion(arm_path, base_configs)

    def start_whole_body_motion(self, arm_path, base_configs):
        """
        Starts a whole-body motion plan execution without blocking.
        This method sends the planned trajectories to the C++ whole body controller
        for coordinated whole-body motion and returns immediately.
        Args:
            arm_path: List of arm joint configurations (8-DOF including torso)
            base_configs: List of base configurations [x, y, theta]
        Returns:
            bool: True if execution was started successfully, False otherwise
        """
        return self.robot_env.start_whole_body_motion(arm_path, base_configs)

    def stop_whole_body_motion(self):
        """Stops the current whole-body motion."""
        return self.robot_env.stop_whole_body_motion()

    def is_motion_done(self):
        """Checks if a whole-body motion is currently executing."""
        return self.robot_env.is_motion_done()

    def get_motion_result(self):
        """Returns the success status of the last motion."""
        return self.robot_env.get_motion_result()

    def get_arm_action_state(self):
        """Gets the state of the arm action client."""
        return self.robot_env.get_arm_action_state()

    def cancel_arm_goals(self):
        """Cancels all goals for the arm action client."""
        self.robot_env.cancel_arm_goals()

    def cancel_torso_goals(self):
        """Cancels all goals for the torso action client."""
        self.robot_env.cancel_torso_goals()

    def cancel_head_goals(self):
        """Cancels all pending head movement goals."""
        self.robot_env.cancel_head_goals()

    def get_arm_action_result(self):
        """Gets the result of the arm action client."""
        return self.robot_env.get_arm_action_result()

    def start_arm_trajectory_async(self, arm_path, duration):
        """
        Starts an arm trajectory execution without blocking.
        Args:
            arm_path: List of arm joint configurations (8-DOF including torso)
            duration: The duration of the trajectory.
        Returns:
            bool: True if execution was started successfully, False otherwise
        """
        return self.robot_env.start_joint_trajectory_async(arm_path, duration)

    def add_sphere(self, position, radius, name=None):
        """
        Add a sphere constraint to the environment.

        Args:
            position: [x, y, z] center position
            radius: sphere radius
            name: optional name for the sphere
        """
        position = list(position)  # Convert to list to ensure correct type
        sphere = vamp.Sphere(position, radius)
        if name:
            sphere.name = name
        self.planning_env.add_sphere(sphere)

    def filter_points_on_robot_with_state(self, points, joint_dict, point_radius=0.1):
        """
        Filters a point cloud using synchronized robot state from joint_dict.
        Used for scene updates during replanning to ensure consistency.

        Args:
            points (list or np.ndarray): The input point cloud, as a list of [x, y, z] points
                                         or a NumPy array of shape (N, 3).
            joint_dict (dict): Dictionary of joint states for synchronized filtering.
            point_radius (float): The radius around each point to consider for collision.

        Returns:
            (list, float): A tuple containing:
                - A new list of points with the robot's points filtered out.
                - The time taken for filtering in seconds.
        """
        robot_filter_start_time = time()

        # Get robot configuration from joint_dict
        base_joint_names = self.robot_env.agent.base_joint_names
        current_joints = [joint_dict[name] for name in self.planning_joint_names]

        # Get base from joint_dict
        base_x = joint_dict[base_joint_names[0]]
        base_y = joint_dict[base_joint_names[1]]
        base_theta = joint_dict[base_joint_names[2]]

        # Apply base offset for world frame
        if hasattr(self.robot_env, "_qpos_base_offset"):
            offset = self.robot_env._qpos_base_offset
            base_x += offset[0]
            base_y += offset[1]
            base_theta += offset[2]

        current_base = (base_x, base_y, base_theta)

        # Set base in VAMP
        self.set_base_params(current_base[2], current_base[0], current_base[1])

        # Ensure points are in list format for VAMP
        if isinstance(points, np.ndarray):
            points_list = points.tolist()
        else:
            points_list = points

        # Call VAMP's robot filtering function
        filtered_points = self.vamp_module.filter_fetch_from_pointcloud(
            points_list, current_joints, current_base, self.planning_env, point_radius
        )

        robot_filter_time = time() - robot_filter_start_time
        # filtered_count = len(points) - len(filtered_points)

        return filtered_points, robot_filter_time

    def filter_points_on_robot(self, points, point_radius=0.1):
        """
        Filters a point cloud to remove points that are inside or too close to the robot's body.

        This method uses the robot's current configuration (base and joints) to check for
        collisions between the provided points and the robot model.

        Args:
            points (list or np.ndarray): The input point cloud, as a list of [x, y, z] points
                                         or a NumPy array of shape (N, 3).
            point_radius (float): The radius around each point to consider for collision.
                                  Points within this distance of the robot will be removed.

        Returns:
            (list, float): A tuple containing:
                - A new list of points with the robot's points filtered out.
                - The time taken for filtering in seconds.
              Returns the original list and 0 time if the robot's state cannot be determined.
        """
        robot_filter_start_time = time()

        # Get current robot configuration
        current_joints = self.get_current_planning_joints()
        if current_joints is None:
            log.warning(
                "Could not get current joint positions for robot filtering, returning original points."
            )
            return points, 0

        # Get current base configuration and set it in VAMP
        current_base = self.get_base_params()
        self.set_base_params(current_base[2], current_base[0], current_base[1])

        # Ensure points are in list format for VAMP
        if isinstance(points, np.ndarray):
            points_list = points.tolist()
        else:
            points_list = points

        # Call VAMP's robot filtering function
        filtered_points = self.vamp_module.filter_fetch_from_pointcloud(
            points_list, current_joints, current_base, self.planning_env, point_radius
        )

        robot_filter_time = time() - robot_filter_start_time
        # filtered_count = len(points) - len(filtered_points)

        return filtered_points, robot_filter_time

    def add_pointcloud(
        self,
        points,
        filter_robot=True,
        point_radius=0.03,
    ):
        """
        Add a pointcloud as collision constraint from already loaded point data.

        Args:
            points: List of [x,y,z] points or numpy array of shape (N,3)
            filter_robot: Whether to filter out points that collide with the
                robot's current configuration (default: True)
            point_radius: Radius for each point when checking robot collisions
                (default: 0.1)

        Returns:
            float: Time taken to process and add the point cloud or -1 if error
        """
        from time import time

        import numpy as np

        start_time = time()
        # Convert to numpy array if not already
        if not isinstance(points, np.ndarray):
            points = np.array(points, dtype=np.float64)
        points_to_use = points
        # Filter out points that collide with the robot's current configuration
        if filter_robot and len(points) > 0:
            points_to_use, robot_filter_time = self.filter_points_on_robot(
                points, point_radius=point_radius
            )
        else:
            pass
            # robot_filter_time = 0
            # if not filter_robot:

        # Visualize points_to_use for debugging in RViz (only if debug is enabled)
        if len(points_to_use) > 0 and self.enable_debug:
            # Prefer Pyrender live viewer if available; fallback to ROS PointCloud2
            if pyrender_is_available():
                pyrender_update_pointcloud(points_to_use)
            elif self.pointcloud_publisher is not None and not self.is_sim:
                header = Header()
                header.stamp = rospy.Time.now()
                header.frame_id = "map"
                fields = [
                    PointField("x", 0, PointField.FLOAT32, 1),
                    PointField("y", 4, PointField.FLOAT32, 1),
                    PointField("z", 8, PointField.FLOAT32, 1),
                ]
                # Ensure points_to_use is a list of lists for create_cloud
                if isinstance(points_to_use, np.ndarray):
                    points_for_pcl = points_to_use.tolist()
                else:
                    points_for_pcl = points_to_use
                pc2_msg = point_cloud2.create_cloud(header, fields, points_for_pcl)
                self.pointcloud_publisher.publish(pc2_msg)

        # Define robot-specific radius parameters
        r_min, r_max = vamp.ROBOT_RADII_RANGES[
            "fetch"
        ]  # Min/max sphere radius for Fetch robot

        # Add the filtered point cloud to the environment
        # Ensure points are in list format for vamp
        if isinstance(points_to_use, np.ndarray):
            points_to_use = points_to_use.tolist()

        # add_start_time = time()
        _ = self.planning_env.add_pointcloud(points_to_use, r_min, r_max, point_radius)
        # add_time = time() - add_start_time

        processing_time = time() - start_time


        # Update FCIT* XY bounds from the point cloud for whole-body planning.
        pts_np = np.array(points_to_use, dtype=np.float64)
        if pts_np.size > 0 and pts_np.shape[1] == 3:
            min_xy = pts_np[:, :2].min(axis=0)
            max_xy = pts_np[:, :2].max(axis=0)
            x_min = float(min_xy[0] - self._bounds_padding)
            x_max = float(max_xy[0] + self._bounds_padding)
            y_min = float(min_xy[1] - self._bounds_padding)
            y_max = float(max_xy[1] + self._bounds_padding)
            self._pc_bounds_xy = (x_min, x_max, y_min, y_max)

        return processing_time

    def attach_objects_to_eef(
        self, spheres_params, offset_position=None, offset_orientation_xyzw=None
    ):
        """
        Attaches one or more sphere collision objects to the end-effector for motion planning collision checking.

        This method will NOT replace any existing attachment. If an object is already
        attached, it will return False.
        The offset_position and offset_orientation are relative to the end-effector's frame.
        The individual sphere positions within 'spheres_params' list are relative
        to the attachment's local frame, which is itself offset from the EEF.

        Args:
            spheres_params (list): List of dictionaries, where each dictionary represents a sphere.
                                   Each sphere dictionary must have:
                                   - "position": [x,y,z] relative to the attachment's local frame.
                                   - "radius": float.
            offset_position (list, optional): [x, y, z] offset from end-effector. Defaults to [0,0,0].
            offset_orientation_xyzw (list, optional): [x, y, z, w] quaternion
                offset from end-effector. Defaults to [0, 0, 0, 1].

        Returns:
            bool: True if attachment was successful, False otherwise (e.g., if an object is already attached).
        """
        if offset_position is None:
            offset_position = [0, 0, 0]
        if offset_orientation_xyzw is None:
            offset_orientation_xyzw = [0, 0, 0, 1]

        # Check if an attachment already exists
        if self._current_attachment:
            log.warning(
                "Cannot attach object: An object is already attached to the end-effector. Detach it first."
            )
            return False

        attachment = vamp.Attachment(offset_position, offset_orientation_xyzw)

        if not isinstance(spheres_params, list) or not all(
            isinstance(s, dict) for s in spheres_params
        ):
            raise ValueError("spheres_params must be a list of sphere dictionaries.")

        vamp_spheres = []
        for s_params in spheres_params:
            if "position" not in s_params or "radius" not in s_params:
                raise ValueError(
                    "Each sphere in the list must have 'position' and 'radius'."
                )
            vamp_spheres.append(
                vamp.Sphere(list(s_params["position"]), s_params["radius"])
            )

        attachment.add_spheres(vamp_spheres)

        self.planning_env.attach(attachment)
        self._current_attachment = attachment
        return True

    def detach_objects_from_eef(self):
        """
        Detaches the currently active collision object(s) from the end-effector.
        """
        if self._current_attachment:
            self.planning_env.detach()
            self._current_attachment = None
            return True
        else:
            return False

    def send_joint_values(self, target_joints, duration=5.0):
        """
        Move the arm and torso to specified joint positions using VAMP planning.

        Args:
            target_joints: List of 8 joint positions [torso + 7 arm joints]
            duration: Time to execute trajectory
        """
        assert (
            len(target_joints) == 8
        ), "Expected 8 joint positions [torso + 7 arm joints]"

        # Get current joints
        current_joints = self.get_current_planning_joints()

        if current_joints is None:
            log.warning("Failed to get current joint positions")
            return None

        log.info(f"Planning motion to target configuration: {target_joints}")

        # Set VAMP base to current robot base before planning
        current_base = self.get_base_params()
        self.set_base_params(current_base[2], current_base[0], current_base[1])

        # Check if the current joint values are close to the target joint values
        if np.allclose(current_joints, target_joints, atol=0.03):
            log.info(
                "Current joint values are close to the target joint values, skipping motion planning"
            )
            return current_joints

        # Plan path using VAMP
        trajectory_points = self._plan_arm(current_joints, target_joints)
        if trajectory_points is None:
            log.error("Failed to plan path with VAMP")
            return None

        return self._execute_joint_trajectory(trajectory_points, duration)

    def send_joint_values_with_replanning(
        self,
        target_joints,
        enable_gaze_control=True,
        enable_replanning=True,
        lookahead_window=200,
        decay_rate=0.95,
        max_replan_attempts=3,
        duration=5.0,
    ):
        """
        Move arm to target configuration with replanning and optional gaze control.

        Args:
            target_joints: Target 8-DOF configuration [torso + 7 arm joints]
            enable_gaze_control: Whether to enable gaze tracking (default: True)
            enable_replanning: Whether to enable collision monitoring and replanning (default: True)
            lookahead_window: Number of future waypoints for gaze (default: 20)
            decay_rate: Distance decay rate for gaze weighting (default: 0.95)
            max_replan_attempts: Maximum replanning attempts on collision (default: 3)
            duration: Execution duration estimate (default: 5.0s)

        Returns:
            Result from action client, or None on failure (same as send_joint_values)
        """
        from grasp_anywhere.utils.motion_utils import move_arm_with_replanning

        return move_arm_with_replanning(
            robot=self,
            target_joints=target_joints,
            enable_gaze_control=enable_gaze_control,
            enable_replanning=enable_replanning,
            lookahead_window=lookahead_window,
            decay_rate=decay_rate,
            max_replan_attempts=max_replan_attempts,
            duration=duration,
        )

    def send_cartesian_interpolated_motion(
        self, target_ee_pos, target_ee_quat, duration=3.0, num_waypoints=5
    ):
        """
        Move to target end-effector pose via Cartesian interpolation mapped to joint space.

        This method:
        1. Interpolates in Cartesian space
        2. Uses TRAC-IK with progressive seeding to map each waypoint to joint space
        3. Sends the joint trajectory to the trajectory controller (accurate & fast)

        This avoids joint wrapping issues and ensures smooth Cartesian paths.

        Args:
            target_ee_pos: Target end-effector position [x, y, z] in world frame.
            target_ee_quat: Target end-effector orientation [x, y, z, w] quaternion in world frame.
            duration: Time to execute trajectory (default: 3.0s)
            num_waypoints: Number of Cartesian waypoints to generate (default: 20)

        Returns:
            Result from action client, or None on failure
        """
        from scipy.spatial.transform import Rotation as R
        from scipy.spatial.transform import Slerp

        from grasp_anywhere.robot.ik.trac_api import trac_solve_fixed_base_arm

        # Get current state
        current_torso = self.get_torso_position()
        current_arm_joints = self.get_arm_joint_values()
        base_config = self.get_base_params()

        # Compute current end-effector pose
        current_full_config = [current_torso] + current_arm_joints

        # Get EE pose in robot base frame from FK
        current_ee_pos_base, current_ee_quat_base = self.vamp_module.eefk(
            current_full_config
        )

        # Transform current EE pose to world frame
        (
            current_ee_pos_world,
            current_ee_quat_world,
        ) = transform_utils.transform_pose_to_world(
            [base_config[0], base_config[1], 0],
            base_config[2],
            current_ee_pos_base,
            current_ee_quat_base,
        )

        target_ee_pos_world = target_ee_pos
        target_ee_quat_world = target_ee_quat

        # Create SLERP interpolator for orientation
        key_rots = R.from_quat([current_ee_quat_world, target_ee_quat_world])
        key_times = [0, 1]
        slerp = Slerp(key_times, key_rots)

        # Generate interpolated Cartesian waypoints
        alphas = np.linspace(0, 1, num_waypoints)
        joint_trajectory = []
        seed = current_full_config  # Progressive seeding for smooth IK solutions (includes torso)

        for i, alpha in enumerate(alphas):
            # Interpolate position (linear) and orientation (SLERP)
            interp_pos = (1 - alpha) * np.array(
                current_ee_pos_world
            ) + alpha * np.array(target_ee_pos_world)
            interp_rot = slerp(alpha)
            interp_quat = interp_rot.as_quat()

            # Solve IK with progressive seeding (each solution seeds the next)
            # Use fixed_base_arm solver which includes torso (8-DOF)
            ik_solution = trac_solve_fixed_base_arm(
                seed,
                base_config,
                interp_pos.tolist(),
                interp_quat.tolist(),
            )

            if ik_solution is None:
                log.warning(
                    f"IK failed at waypoint {i}/{num_waypoints} (alpha={alpha:.3f})"
                )
                return None

            # Add the full 8-DOF configuration (torso + arm joints) to trajectory
            joint_trajectory.append(ik_solution)

            # Update seed for next iteration (progressive seeding)
            seed = ik_solution

        # Execute trajectory
        return self._execute_joint_trajectory(joint_trajectory, duration)

    def _execute_joint_trajectory(self, trajectory_points, duration):
        """
        Execute a joint trajectory using the arm and torso controllers.

        Args:
            trajectory_points: List of 8-DOF configurations [torso + 7 arm joints]
            duration: Total duration for trajectory execution

        Returns:
            Result from action client, or None on failure
        """
        return self.robot_env.execute_joint_trajectory(trajectory_points, duration)

    def move_base(self, linear_x, angular_z):
        """
        Move the robot base with specified linear and angular velocities.

        Args:
            linear_x (float): Forward/backward velocity (-1.0 to 1.0)
            angular_z (float): Rotational velocity (-1.0 to 1.0)
        """
        # Clip velocities to safe ranges
        linear_x = np.clip(linear_x, -1.0, 1.0) * self.max_linear_speed
        angular_z = np.clip(angular_z, -1.0, 1.0) * self.max_angular_speed
        self.robot_env.move_base(linear_x, angular_z)

    def stop_base(self):
        """Stop all base movement."""
        self.robot_env.stop_base()

    def navigate_to(self, position, orientation):
        """
        Send the robot base to a target position in the map frame.

        Args:
            position (list): [x, y, z] target position
            orientation (list): [x, y, z, w] target orientation as quaternion
        """
        return self.robot_env.navigate_to(position, orientation)

    def set_torso_height(self, height, duration=5.0):
        """
        Set the torso height to a specific value.

        Args:
            height (float): Target height for the torso lift joint.
            duration (float): Time to execute the movement.
        """
        return self.robot_env.set_torso_height(height, duration)

    def control_gripper(self, position, max_effort=100):
        """
        Control the gripper position.

        Args:
            position (float): 0.0 (closed) to 1.0 (open)
            max_effort (float): Maximum effort to apply
        """
        return self.robot_env.control_gripper(position, max_effort)

    def get_gripper_status(self):
        """
        Get the current gripper status and position.

        Returns:
            dict: Dictionary containing gripper status information
                  {
                      'position': float,  # Current gripper position (0.0=closed, 1.0=open)
                      'effort': float,     # Current effort being applied
                      'stalled': bool,     # Whether gripper is stalled
                      'reached_goal': bool # Whether gripper reached the goal position
                  }
        """
        return self.robot_env.get_gripper_status()

    def move_head(self, pan, tilt, duration=1.0):
        """
        Moves the robot's head to a given pan and tilt position.

        Args:
            pan (float): The target pan position for the head.
            tilt (float): The target tilt position for the head.
            duration (float): The duration of the movement in seconds.
        """
        self.robot_env.move_head(pan, tilt, duration)

    def point_head_at(self, target_point, frame_id="map", duration=1.0):
        """
        Points the robot's head towards a target point in the specified frame.
        """
        # Transform the point from the given frame_id to the base_link frame
        if frame_id != "base_link":
            base_x, base_y, base_theta = self.get_base_params()
            T_map_base = _create_transform_matrix(
                [base_x, base_y, 0], R.from_euler("z", base_theta).as_matrix()
            )
            T_base_map = np.linalg.inv(T_map_base)
            target_point_base = (T_base_map @ np.append(target_point, 1))[:3]
        else:
            target_point_base = np.array(target_point)

        # Get current joint states for FK
        joint_states = self.get_current_planning_joints_with_head()
        if joint_states is None:
            log.error("Could not get current joint states for FK.")
            return

        link_poses = forward_kinematics(joint_states)
        current_head_pan = float(joint_states[-2])
        # current_head_tilt = float(joint_states[-1])

        # Compute RELATIVE pan correction in the current head_pan_link frame
        T_base_head_pan = link_poses["head_pan_link"]
        T_head_pan_base = np.linalg.inv(T_base_head_pan)
        target_point_head_pan = (T_head_pan_base @ np.append(target_point_base, 1))[:3]
        x, y, z = target_point_head_pan
        pan_rel = float(np.arctan2(y, x))

        # Calculate Tilt in the ALIGNED frame (simulating pan completion)
        # Transform Tilt Joint Origin to Pan Frame (it's constant)
        T_base_head_tilt = link_poses["head_tilt_link"]
        T_pan_tilt = T_head_pan_base @ T_base_head_tilt
        tilt_origin_pan = T_pan_tilt[:3, 3]

        # In Aligned Frame: Target is (dist_xy, 0, z), Tilt Origin is same as in Pan Frame
        # Vector from Tilt Joint to Target
        dist_xy = np.sqrt(x**2 + y**2)
        v_tilt_target = np.array([dist_xy, 0, z]) - tilt_origin_pan

        # Calculate absolute tilt angle (pitch down positive)
        tilt_abs = float(np.arctan2(-v_tilt_target[2], v_tilt_target[0]))

        # ManiSkill sim head controller expects ABSOLUTE pan/tilt targets.
        pan = current_head_pan + pan_rel
        tilt = tilt_abs

        # Command the head to the new position
        self.move_head(pan, tilt, duration)

    def clear_pointclouds(self):
        """
        Clears all point cloud collision objects from the VAMP environment.
        This is useful for updating the environment with new sensor data.
        """
        self.planning_env.clear_pointclouds()

    def clear_spheres(self):
        """
        Clears all sphere collision objects from the VAMP environment.
        """
        self.planning_env.clear_spheres()

    def check_plan_for_collisions(self, arm_path, base_configs, current_waypoint_index):
        """
        Checks the remainder of a whole-body plan for collisions against the current
        state of the collision environment.

        Args:
            arm_path (list): The list of joint configurations for the arm.
            base_configs (list): The list of configurations for the base.
            current_waypoint_index (int): The index of the last waypoint the robot
                                          has successfully reached.

        Returns:
            bool: True if a collision is detected, False otherwise.
        """
        return replanning_utils.check_trajectory_for_collisions(
            self.vamp_module,
            self.planning_env,
            arm_path,
            base_configs,
            current_waypoint_index,
        )

    def get_rgb(self):
        """Returns the latest RGB image."""
        return self.robot_env.get_rgb()

    def get_depth(self):
        """Returns the latest depth image."""
        return self.robot_env.get_depth()

    def get_camera_intrinsic(self):
        """Returns the camera intrinsic matrix."""
        return self.robot_env.get_camera_intrinsics()
