"""
Gaze optimizer for the Fetch robot head controller.

This module computes optimal gaze targets based on the robot's future trajectory.
It aggregates positions of ALL key joints (shoulder, elbow, wrist, gripper) weighted by:
  - Distance decay: nearer waypoints have higher base importance
  - Cartesian velocity: joints moving faster in world frame get more attention

The weighting function is: weight = (decay_rate^distance) * velocity_world_frame
Zero velocity = zero weight. This ensures the camera naturally tracks moving parts.
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from grasp_anywhere.robot.kinematics import (
    ELBOW_FLEX_OFFSET,
    FOREARM_ROLL_OFFSET,
    GRIPPER_OFFSET,
    L_GRIPPER_FINGER_OFFSET,
    R_GRIPPER_FINGER_OFFSET,
    SHOULDER_LIFT_OFFSET,
    SHOULDER_PAN_OFFSET,
    TORSO_BASE_OFFSET,
    TORSO_FIXED_OFFSET,
    UPPERARM_ROLL_OFFSET,
    WRIST_FLEX_OFFSET,
    WRIST_ROLL_OFFSET,
    forward_kinematics,
)

# Precomputed collision spheres (from fetch_spherized.urdf)
FETCH_SPHERES = {
    "base_link": [
        [-0.12, 0.0, 0.182],
        [0.225, 0.0, 0.31],
        [0.08, -0.06, 0.16],
        [0.215, -0.07, 0.31],
        [0.185, -0.135, 0.31],
        [0.13, -0.185, 0.31],
        [0.065, -0.2, 0.31],
        [0.01, -0.2, 0.31],
        [0.08, 0.06, 0.16],
        [0.215, 0.07, 0.31],
        [0.185, 0.135, 0.31],
        [0.13, 0.185, 0.31],
        [0.065, 0.2, 0.31],
        [0.01, 0.2, 0.31],
    ],
    "torso_lift_link": [
        [-0.1, -0.05, 0.15],
        [-0.1, 0.05, 0.15],
        [-0.1, 0.05, 0.3],
        [-0.1, 0.05, 0.45],
        [-0.1, -0.05, 0.45],
        [-0.1, -0.05, 0.3],
    ],
    "torso_fixed_link": [
        [-0.1, -0.07, 0.35],
        [-0.1, 0.07, 0.35],
        [-0.1, -0.07, 0.2],
        [-0.1, 0.07, 0.2],
        [-0.1, 0.07, 0.07],
        [-0.1, -0.07, 0.07],
    ],
    "head_pan_link": [
        [0.0, 0.0, 0.06],
        [0.145, 0.0, 0.058],
        [0.145, -0.0425, 0.058],
        [0.145, 0.0425, 0.058],
        [0.145, 0.085, 0.058],
        [0.145, -0.085, 0.058],
        [0.0625, -0.115, 0.03],
        [0.088, -0.115, 0.03],
        [0.1135, -0.115, 0.03],
        [0.139, -0.115, 0.03],
        [0.0625, -0.115, 0.085],
        [0.088, -0.115, 0.085],
        [0.1135, -0.115, 0.085],
        [0.139, -0.115, 0.085],
        [0.16, -0.115, 0.075],
        [0.168, -0.115, 0.0575],
        [0.16, -0.115, 0.04],
        [0.0625, 0.115, 0.03],
        [0.088, 0.115, 0.03],
        [0.1135, 0.115, 0.03],
        [0.139, 0.115, 0.03],
        [0.0625, 0.115, 0.085],
        [0.088, 0.115, 0.085],
        [0.1135, 0.115, 0.085],
        [0.139, 0.115, 0.085],
        [0.16, 0.115, 0.075],
        [0.168, 0.115, 0.0575],
        [0.16, 0.115, 0.04],
    ],
    "shoulder_pan_link": [
        [0.0, 0.0, 0.0],
        [0.025, -0.015, 0.035],
        [0.05, -0.03, 0.06],
        [0.12, -0.03, 0.06],
    ],
    "shoulder_lift_link": [
        [0.025, 0.04, 0.025],
        [-0.025, 0.04, -0.025],
        [0.025, 0.04, -0.025],
        [-0.025, 0.04, 0.025],
        [0.08, 0.0, 0.0],
        [0.11, 0.0, 0.0],
        [0.14, 0.0, 0.0],
    ],
    "upperarm_roll_link": [
        [-0.02, 0.0, 0.0],
        [0.03, 0.0, 0.0],
        [0.08, 0.0, 0.0],
        [0.11, -0.045, 0.02],
        [0.11, -0.045, -0.02],
        [0.155, -0.045, 0.02],
        [0.155, -0.045, -0.02],
        [0.13, 0.0, 0.0],
    ],
    "elbow_flex_link": [
        [0.02, 0.045, 0.02],
        [0.02, 0.045, -0.02],
        [-0.02, 0.045, 0.02],
        [-0.02, 0.045, -0.02],
        [0.08, 0.0, 0.0],
        [0.14, 0.0, 0.0],
    ],
    "forearm_roll_link": [
        [0.0, 0.0, 0.0],
        [0.05, -0.06, 0.02],
        [0.05, -0.06, -0.02],
        [0.1, -0.06, 0.02],
        [0.1, -0.06, -0.02],
        [0.15, -0.06, 0.02],
        [0.15, -0.06, -0.02],
    ],
    "wrist_flex_link": [
        [0.0, 0.0, 0.0],
        [0.06, 0.0, 0.0],
        [0.02, 0.045, 0.02],
        [0.02, 0.045, -0.02],
        [-0.02, 0.045, 0.02],
        [-0.02, 0.045, -0.02],
    ],
    "wrist_roll_link": [
        [-0.03, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ],
    "gripper_link": [
        [-0.07, 0.02, 0.0],
        [-0.07, -0.02, 0.0],
        [-0.1, 0.02, 0.0],
        [-0.1, -0.02, 0.0],
    ],
    "r_gripper_finger_link": [
        [0.017, -0.0085, -0.005],
        [0.017, -0.0085, 0.005],
        [0.0, -0.0085, -0.005],
        [0.0, -0.0085, 0.005],
        [-0.017, -0.0085, -0.005],
        [-0.017, -0.0085, 0.005],
    ],
    "l_gripper_finger_link": [
        [0.017, 0.0085, -0.005],
        [0.017, 0.0085, 0.005],
        [0.0, 0.0085, -0.005],
        [0.0, 0.0085, 0.005],
        [-0.017, 0.0085, -0.005],
        [-0.017, 0.0085, 0.005],
    ],
}


class GazeOptimizer:
    """
    Optimizes robot head gaze based on future trajectory.

    Aggregates ALL key joint positions (shoulder, elbow, wrist, gripper) weighted by
    their Cartesian velocity in world frame. Fast-moving parts get more attention.

    Weighting: weight = (decay_rate^distance) * velocity_world_frame
    - 0 velocity = 0 weight (stationary joints ignored)
    - High velocity = high weight (moving joints tracked)

    Example usage:
        # Initialize with head controller
        gaze_opt = GazeOptimizer(robot, lookahead_window=40, decay_rate=0.99)

        # Set the planned trajectory (Nx11 array)
        gaze_opt.set_trajectory(whole_body_trajectory)

        # During execution, call update() at each waypoint
        for idx in range(len(trajectory)):
            gaze_opt.update(idx)  # Automatically points head at weighted target
            # ... execute motion to waypoint ...
    """

    def __init__(
        self,
        robot,
        lookahead_window=80,
        decay_rate=0.99,
        velocity_weight=1.0,
        joint_priorities=None,
        use_gpu=True,
    ):
        """
        Initialize the gaze optimizer.

        Args:
            robot: The Fetch robot instance.
            lookahead_window: Number of future waypoints to consider (default: 80).
            decay_rate: Exponential decay for weighting future waypoints (0-1).
            velocity_weight: Weight multiplier for velocity awareness (default: 1.0).
            joint_priorities: Dictionary of joint weights (default: None, uses built-in).
            use_gpu: Whether to use GPU for acceleration (default: True).
        """
        self.robot = robot
        self.lookahead_window = lookahead_window
        self.decay_rate = decay_rate
        self.velocity_weight = velocity_weight
        self.joint_priorities = joint_priorities
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")

        self.trajectory = None
        self.joint_velocities = None  # Link origin velocities (numpy, for viz)
        self.sphere_trajectories = None  # (T, N_spheres, 3) torch tensor
        self.sphere_velocities = None  # (T, N_spheres) torch tensor
        self.base_frame = "base_link"

        # Load precomputed spheres onto device
        self.link_spheres = {}
        if self.use_gpu:
            for link_name, spheres in FETCH_SPHERES.items():
                self.link_spheres[link_name] = torch.tensor(
                    spheres, dtype=torch.float32, device=self.device
                )

        # Key joints to track for gaze (body emphasized)
        self.key_joints = [
            "base",
            "torso",
            "shoulder_lift",
            "elbow",
            "wrist_flex",
            "gripper",
        ]

    def _torch_create_transform(self, translation, rotation_matrix):
        """Create Bx4x4 transform batch."""
        B = translation.shape[0]
        T = torch.eye(4, device=self.device).unsqueeze(0).repeat(B, 1, 1)
        T[:, :3, :3] = rotation_matrix
        T[:, :3, 3] = translation
        return T

    def _torch_euler_to_matrix(self, angles, axis):
        """
        Create rotation matrices from euler angles (batch).
        angles: (B,)
        axis: 'x', 'y', or 'z'
        Returns: (B, 3, 3)
        """
        c = torch.cos(angles)
        s = torch.sin(angles)
        o = torch.zeros_like(angles)
        ones = torch.ones_like(angles)

        if axis == "x":
            mat = torch.stack([ones, o, o, o, c, -s, o, s, c], dim=1).reshape(-1, 3, 3)
        elif axis == "y":
            mat = torch.stack([c, o, s, o, ones, o, -s, o, c], dim=1).reshape(-1, 3, 3)
        elif axis == "z":
            mat = torch.stack([c, -s, o, s, c, o, o, o, ones], dim=1).reshape(-1, 3, 3)
        return mat

    def _batch_fk_torch(self, joint_angles_10dof):
        """
        Compute FK for Fetch on GPU (Batch).
        joint_angles: (B, 10)
        Returns: Dict[str, Tensor(B, 4, 4)]
        """
        B = joint_angles_10dof.shape[0]

        # Unpack joints
        torso_lift = joint_angles_10dof[:, 0]
        shoulder_pan = joint_angles_10dof[:, 1]
        shoulder_lift = joint_angles_10dof[:, 2]
        upperarm_roll = joint_angles_10dof[:, 3]
        elbow_flex = joint_angles_10dof[:, 4]
        forearm_roll = joint_angles_10dof[:, 5]
        wrist_flex = joint_angles_10dof[:, 6]
        wrist_roll = joint_angles_10dof[:, 7]
        # Head joints ignored for body spheres

        link_poses = {}

        # Base (Identity)
        T_base = torch.eye(4, device=self.device).unsqueeze(0).repeat(B, 1, 1)
        link_poses["base_link"] = T_base

        # Torso Fixed
        # T_tf = T_base * Offset
        T_tf_trans = self._torch_create_transform(
            torch.tensor(TORSO_FIXED_OFFSET, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .repeat(B, 1),
            torch.eye(3, device=self.device).unsqueeze(0).repeat(B, 1, 1),
        )
        link_poses["torso_fixed_link"] = torch.bmm(T_base, T_tf_trans)

        # Helper for offsets (numpy -> torch)
        def to_torch(arr):
            return (
                torch.tensor(arr, dtype=torch.float32, device=self.device)
                .unsqueeze(0)
                .repeat(B, 1)
            )

        # 1. Base -> Torso
        torso_trans = to_torch(TORSO_BASE_OFFSET)
        torso_trans[:, 2] += torso_lift
        T_torso = self._torch_create_transform(
            torso_trans, torch.eye(3, device=self.device).unsqueeze(0).repeat(B, 1, 1)
        )
        T_torso_accum = torch.bmm(T_base, T_torso)
        link_poses["torso_lift_link"] = T_torso_accum

        # 2. Torso -> Shoulder Pan
        T_span_trans = self._torch_create_transform(
            to_torch(SHOULDER_PAN_OFFSET),
            torch.eye(3, device=self.device).unsqueeze(0).repeat(B, 1, 1),
        )
        T_span_rot = self._torch_create_transform(
            torch.zeros(B, 3, device=self.device),
            self._torch_euler_to_matrix(shoulder_pan, "z"),
        )
        T_span = torch.bmm(T_torso_accum, torch.bmm(T_span_trans, T_span_rot))
        link_poses["shoulder_pan_link"] = T_span

        # 3. Shoulder Pan -> Shoulder Lift
        T_slift_trans = self._torch_create_transform(
            to_torch(SHOULDER_LIFT_OFFSET),
            torch.eye(3, device=self.device).unsqueeze(0).repeat(B, 1, 1),
        )
        T_slift_rot = self._torch_create_transform(
            torch.zeros(B, 3, device=self.device),
            self._torch_euler_to_matrix(shoulder_lift, "y"),
        )
        T_slift = torch.bmm(T_span, torch.bmm(T_slift_trans, T_slift_rot))
        link_poses["shoulder_lift_link"] = T_slift

        # 4. Lift -> Upperarm
        T_uar_trans = self._torch_create_transform(
            to_torch(UPPERARM_ROLL_OFFSET),
            torch.eye(3, device=self.device).unsqueeze(0).repeat(B, 1, 1),
        )
        T_uar_rot = self._torch_create_transform(
            torch.zeros(B, 3, device=self.device),
            self._torch_euler_to_matrix(upperarm_roll, "x"),
        )
        T_uar = torch.bmm(T_slift, torch.bmm(T_uar_trans, T_uar_rot))
        link_poses["upperarm_roll_link"] = T_uar

        # 5. Upperarm -> Elbow
        T_elb_trans = self._torch_create_transform(
            to_torch(ELBOW_FLEX_OFFSET),
            torch.eye(3, device=self.device).unsqueeze(0).repeat(B, 1, 1),
        )
        T_elb_rot = self._torch_create_transform(
            torch.zeros(B, 3, device=self.device),
            self._torch_euler_to_matrix(elbow_flex, "y"),
        )
        T_elb = torch.bmm(T_uar, torch.bmm(T_elb_trans, T_elb_rot))
        link_poses["elbow_flex_link"] = T_elb

        # 6. Elbow -> Forearm
        T_for_trans = self._torch_create_transform(
            to_torch(FOREARM_ROLL_OFFSET),
            torch.eye(3, device=self.device).unsqueeze(0).repeat(B, 1, 1),
        )
        T_for_rot = self._torch_create_transform(
            torch.zeros(B, 3, device=self.device),
            self._torch_euler_to_matrix(forearm_roll, "x"),
        )
        T_for = torch.bmm(T_elb, torch.bmm(T_for_trans, T_for_rot))
        link_poses["forearm_roll_link"] = T_for

        # 7. Forearm -> Wrist Flex
        T_wflex_trans = self._torch_create_transform(
            to_torch(WRIST_FLEX_OFFSET),
            torch.eye(3, device=self.device).unsqueeze(0).repeat(B, 1, 1),
        )
        T_wflex_rot = self._torch_create_transform(
            torch.zeros(B, 3, device=self.device),
            self._torch_euler_to_matrix(wrist_flex, "y"),
        )
        T_wflex = torch.bmm(T_for, torch.bmm(T_wflex_trans, T_wflex_rot))
        link_poses["wrist_flex_link"] = T_wflex

        # 8. Wrist Flex -> Wrist Roll
        T_wroll_trans = self._torch_create_transform(
            to_torch(WRIST_ROLL_OFFSET),
            torch.eye(3, device=self.device).unsqueeze(0).repeat(B, 1, 1),
        )
        T_wroll_rot = self._torch_create_transform(
            torch.zeros(B, 3, device=self.device),
            self._torch_euler_to_matrix(wrist_roll, "x"),
        )
        T_wroll = torch.bmm(T_wflex, torch.bmm(T_wroll_trans, T_wroll_rot))
        link_poses["wrist_roll_link"] = T_wroll

        # 9. Wrist Roll -> Gripper
        T_grip_trans = self._torch_create_transform(
            to_torch(GRIPPER_OFFSET),
            torch.eye(3, device=self.device).unsqueeze(0).repeat(B, 1, 1),
        )
        T_grip = torch.bmm(T_wroll, T_grip_trans)
        link_poses["gripper_link"] = T_grip

        # Fingers
        T_rg_trans = self._torch_create_transform(
            to_torch(R_GRIPPER_FINGER_OFFSET),
            torch.eye(3, device=self.device).unsqueeze(0).repeat(B, 1, 1),
        )
        link_poses["r_gripper_finger_link"] = torch.bmm(T_grip, T_rg_trans)

        T_lg_trans = self._torch_create_transform(
            to_torch(L_GRIPPER_FINGER_OFFSET),
            torch.eye(3, device=self.device).unsqueeze(0).repeat(B, 1, 1),
        )
        link_poses["l_gripper_finger_link"] = torch.bmm(T_grip, T_lg_trans)

        return link_poses

    def _compute_cartesian_velocities(self):
        """
        Compute Cartesian velocities of key joints in world frame.

        Returns:
            dict: {joint_name: np.array of velocities for each waypoint}
                  Each velocity is the magnitude of motion in meters
        """
        if self.trajectory is None or len(self.trajectory) < 2:
            return None

        joint_velocities = {joint: [] for joint in self.key_joints}

        # Compute positions for all waypoints
        all_positions = {joint: [] for joint in self.key_joints}
        for config in self.trajectory:
            link_positions = self._compute_all_link_positions(config)
            for joint in self.key_joints:
                all_positions[joint].append(link_positions[joint])

        # Convert to numpy arrays
        for joint in self.key_joints:
            all_positions[joint] = np.array(all_positions[joint])

        # Compute velocities as positional differences in world frame
        for joint in self.key_joints:
            positions = all_positions[joint]
            diffs = np.diff(positions, axis=0)
            velocities = np.linalg.norm(diffs, axis=1)  # Magnitude in meters

            # Pad with last velocity for final waypoint
            velocities = np.concatenate([velocities, [velocities[-1]]])
            joint_velocities[joint] = velocities

        return joint_velocities

    def set_trajectory(self, trajectory):
        """
        Set the whole-body trajectory to track.

        Args:
            trajectory: Nx11 numpy array or list of 11-DOF configs
                       [x, y, theta, torso, shoulder_pan, shoulder_lift,
                        upperarm_roll, elbow_flex, forearm_roll, wrist_flex, wrist_roll]
        """
        self.trajectory = np.array(trajectory)
        if self.trajectory.ndim == 1:
            self.trajectory = self.trajectory.reshape(1, -1)
        assert self.trajectory.shape[1] == 11, "Trajectory must be 11-DOF (whole body)"

        # 1. Compute Sphere Positions on GPU
        if self.use_gpu:
            traj_tensor = torch.tensor(
                self.trajectory, dtype=torch.float32, device=self.device
            )
            B = traj_tensor.shape[0]

            # Split Base and Joints
            base_pose = traj_tensor[:, :3]  # x,y,theta
            joint_angles_7dof = traj_tensor[:, 3:]  # torso...wrist_roll

            # Prepare 10-dof for FK (append head 0,0)
            zeros = torch.zeros((B, 2), device=self.device)
            joint_angles_10dof = torch.cat([joint_angles_7dof, zeros], dim=1)

            # Running FK
            link_transforms_base = self._batch_fk_torch(joint_angles_10dof)

            # Transforms Base -> Map
            # Create T_map_base
            x = base_pose[:, 0]
            y = base_pose[:, 1]
            theta = base_pose[:, 2]

            T_map_base = torch.eye(4, device=self.device).unsqueeze(0).repeat(B, 1, 1)
            # Rotation Z
            c = torch.cos(theta)
            s = torch.sin(theta)
            T_map_base[:, 0, 0] = c
            T_map_base[:, 0, 1] = -s
            T_map_base[:, 1, 0] = s
            T_map_base[:, 1, 1] = c
            # Translation
            T_map_base[:, 0, 3] = x
            T_map_base[:, 1, 3] = y

            # Collect all sphere positions
            all_spheres = []

            for link_name, spheres_local in self.link_spheres.items():
                if link_name in link_transforms_base:
                    # Get transform for this link: Map -> Link
                    # T_map_link = T_map_base @ T_base_link
                    T_base_link = link_transforms_base[link_name]
                    T_map_link = torch.bmm(T_map_base, T_base_link)

                    # Transform spheres
                    # spheres_local: (N, 3)
                    # We need (B, N, 3)
                    N = spheres_local.shape[0]

                    # Expand spheres to batch: (B, N, 3)
                    spheres_batch = spheres_local.unsqueeze(0).repeat(B, 1, 1)

                    # Apply transform
                    # T: (B, 4, 4), P: (B, N, 3) -> (B, N, 3)
                    # P_h = [x, y, z, 1]
                    ones = torch.ones((B, N, 1), device=self.device)
                    spheres_h = torch.cat([spheres_batch, ones], dim=2)

                    # BMM: (B, 4, 4) x (B, 4, N) -> (B, 4, N)
                    # Transpose spheres to (B, N, 4) -> (B, 4, N)
                    spheres_h_T = spheres_h.transpose(1, 2)

                    transformed_h = torch.bmm(T_map_link, spheres_h_T)

                    # Back to (B, N, 3)
                    transformed = transformed_h.transpose(1, 2)[:, :, :3]

                    all_spheres.append(transformed)

            if all_spheres:
                # Concatenate all spheres: (B, TotalSpheres, 3)
                self.sphere_trajectories = torch.cat(all_spheres, dim=1)

                # Compute Velocities: (B, TotalSpheres)
                # diff[i] = pos[i] - pos[i-1]
                # velocity[i] = |diff[i]|
                if B > 1:
                    diffs = self.sphere_trajectories[1:] - self.sphere_trajectories[:-1]
                    vels = torch.norm(diffs, dim=2)
                    last_vel = vels[-1:]
                    self.sphere_velocities = torch.cat([vels, last_vel], dim=0)
                else:
                    self.sphere_velocities = torch.zeros(
                        (B, self.sphere_trajectories.shape[1]), device=self.device
                    )
            else:
                self.sphere_trajectories = None

        # Precompute Cartesian velocities for all key joints (Legacy)
        self.joint_velocities = self._compute_cartesian_velocities()

    def _compute_all_link_positions(self, config):
        """
        Compute positions of all major arm links in map frame.

        Args:
            config: 11-DOF configuration [x, y, theta, torso, 7 arm joints]

        Returns:
            dict: Dictionary with positions of all major links in map frame
        """
        # Extract base pose and joint angles from the 11-DOF config
        x, y, theta = config[0], config[1], config[2]
        arm_joint_angles = config[3:]  # [torso_lift, 7 arm joints]

        # The forward_kinematics function expects 10 joints, but gaze optimizer doesn't
        # use head joints for its own FK calculations. We'll pass zeros for head joints.
        head_pan, head_tilt = 0.0, 0.0
        full_joint_angles = np.concatenate([arm_joint_angles, [head_pan, head_tilt]])

        # Compute link poses in the base_link frame
        link_poses_base = forward_kinematics(full_joint_angles)

        # Transform all positions to map frame using base pose
        R_map_base = R.from_euler("z", theta).as_matrix()
        t_map_base = np.array([x, y, 0])

        # Transform all link positions to map frame
        link_positions_map = {}
        # Manually create a dictionary of link positions from poses
        link_positions_base = {
            "base": link_poses_base.get("base_link", np.eye(4))[:3, 3],
            "torso": link_poses_base.get("torso_lift_link", np.eye(4))[:3, 3],
            "shoulder_pan": link_poses_base.get("shoulder_pan_link", np.eye(4))[:3, 3],
            "shoulder_lift": link_poses_base.get("shoulder_lift_link", np.eye(4))[
                :3, 3
            ],
            "upperarm": link_poses_base.get("upperarm_roll_link", np.eye(4))[:3, 3],
            "elbow": link_poses_base.get("elbow_flex_link", np.eye(4))[:3, 3],
            "forearm": link_poses_base.get("forearm_roll_link", np.eye(4))[:3, 3],
            "wrist_flex": link_poses_base.get("wrist_flex_link", np.eye(4))[:3, 3],
            "wrist_roll": link_poses_base.get("wrist_roll_link", np.eye(4))[:3, 3],
            "gripper": link_poses_base.get("gripper_link", np.eye(4))[:3, 3],
        }

        for link_name, pos_base in link_positions_base.items():
            link_positions_map[link_name] = R_map_base @ pos_base + t_map_base

        return link_positions_map

    def _compute_weighted_target(self, current_index):
        """
        Compute weighted target point from future trajectory.

        Aggregates ALL key joint positions (shoulder, elbow, wrist, gripper) weighted by:
          weight = (decay_rate^distance) * velocity_in_world_frame

        Zero velocity = zero weight. Fast moving joints get more attention.

        Args:
            current_index: Current waypoint index in trajectory

        Returns:
            np.array: [x, y, z] weighted target position in map frame
        """
        if self.trajectory is None or len(self.trajectory) == 0:
            return None

        # Determine lookahead window
        end_index = min(current_index + self.lookahead_window, len(self.trajectory))

        if current_index >= len(self.trajectory) - 1:
            # At or past the end, aggregate all key joints at final position
            link_positions = self._compute_all_link_positions(self.trajectory[-1])
            positions = [link_positions[joint] for joint in self.key_joints]
            return np.mean(positions, axis=0)

        if self.use_gpu and self.sphere_trajectories is not None:
            # GPU Mode: Use all spheres

            # Slice the window
            # sphere_trajectories: (TotalSteps, NSpheres, 3)
            # sphere_velocities: (TotalSteps, NSpheres)

            traj_slice = self.sphere_trajectories[
                current_index + 1 : end_index
            ]  # (W, N, 3)
            vel_slice = self.sphere_velocities[current_index + 1 : end_index]  # (W, N)

            if traj_slice.shape[0] == 0:
                next_idx = min(current_index + 1, len(self.trajectory) - 1)
                return self.sphere_trajectories[next_idx].mean(dim=0).cpu().numpy()

            # Distance weights (Broadcasting)
            # distance: 1..W
            # Create distance tensor
            W = traj_slice.shape[0]
            steps = torch.arange(1, W + 1, device=self.device).float()

            window_len = end_index - current_index
            mid = 0.5 * window_len
            sigma = max(1.0, 0.25 * window_len)

            # Gaussian distance weight: (W,)
            dist_weights = torch.exp(-0.5 * ((steps - mid) / sigma) ** 2)

            # Expand to (W, N)
            dist_weights = dist_weights.unsqueeze(1).repeat(1, traj_slice.shape[1])

            # Velocity Term
            if self.velocity_weight < 1e-6:
                vel_term = 1.0
            else:
                vel_term = vel_slice**self.velocity_weight

            # Total Weight: (W, N)
            weights = dist_weights * vel_term

            # Weighted Sum
            # weights: (W, N) -> (W, N, 1)
            weighted_pos = traj_slice * weights.unsqueeze(2)

            sum_pos = weighted_pos.sum(dim=(0, 1))  # Sum over time and spheres
            sum_weight = weights.sum()

            if sum_weight > 1e-9:
                target = sum_pos / sum_weight
                return target.cpu().numpy()
            else:
                # Fallback to mean of all spheres in middle of window
                mid_idx = (current_index + end_index) // 2
                return self.sphere_trajectories[mid_idx].mean(dim=0).cpu().numpy()

        if end_index <= current_index + 1:
            # No future waypoints, aggregate all key joints at next position
            link_positions = self._compute_all_link_positions(
                self.trajectory[current_index + 1]
            )
            positions = [link_positions[joint] for joint in self.key_joints]
            return np.mean(positions, axis=0)

        # Aggregate weighted positions from ALL key joints
        weighted_sum = np.zeros(3)
        total_weight = 0.0

        # Prefer mid-lookahead distances (not too near, not too far)
        window_len = end_index - current_index
        mid = 0.5 * window_len
        sigma = max(1.0, 0.25 * window_len)

        # Emphasize body joints
        if self.joint_priorities:
            joint_priority = self.joint_priorities
        else:
            joint_priority = {
                "base": 3.0,
                "torso": 2.0,
                "shoulder_lift": 1.3,
                "elbow": 1.1,
                "wrist_flex": 1.0,
                "gripper": 1.0,
            }

        for i in range(current_index + 1, end_index):
            distance = i - current_index

            # Mid-focused distance weighting (Gaussian around the middle of the window)
            distance_weight = np.exp(-0.5 * ((distance - mid) / sigma) ** 2)

            # Get all link positions for this waypoint
            link_positions = self._compute_all_link_positions(self.trajectory[i])

            # For each key joint, add its contribution weighted by velocity
            for joint_name in self.key_joints:
                joint_pos = link_positions[joint_name]

                # Get Cartesian velocity in world frame
                velocity = 0.0
                if (
                    self.joint_velocities is not None
                    and joint_name in self.joint_velocities
                ):
                    velocity = self.joint_velocities[joint_name][i]

                # Weight: distance term * velocity importance * joint priority
                # Use parameter as exponent to control contrast:
                # - 1.0 = Linear (original behavior, 0 velocity = 0 weight)
                # - 0.0 = Uniform (velocity ignored, all get weight 1.0)
                # - >1.0 = Higher contrast (emphasize fastest parts)
                if self.velocity_weight < 1e-6:
                    velocity_term = 1.0
                else:
                    velocity_term = velocity**self.velocity_weight

                joint_weight = joint_priority.get(joint_name, 1.0)
                weight = distance_weight * velocity_term * joint_weight

                weighted_sum += weight * joint_pos
                total_weight += weight

        if total_weight > 1e-9:  # Avoid division by zero
            target = weighted_sum / total_weight
        else:
            # Fallback: if all velocities are near zero, look at average of all key joints
            # Use the middle of lookahead window
            mid_index = (current_index + end_index) // 2
            link_positions = self._compute_all_link_positions(
                self.trajectory[mid_index]
            )
            positions = [link_positions[joint] for joint in self.key_joints]
            target = np.mean(positions, axis=0)

        return target

    def update(self, current_index):
        """
        Update gaze based on current trajectory progress.

        Computes optimal gaze target from future trajectory waypoints and
        commands the head controller to look at it.

        Args:
            current_index: Current waypoint index in the trajectory
        """
        if self.trajectory is None:
            return

        # Compute weighted target point
        target = self._compute_weighted_target(current_index)

        if target is None:
            return

        # Command head to look at target
        self.robot.point_head_at(
            target_point=target.tolist(),
            frame_id="map",
            duration=0.5,  # Smooth tracking with short duration
        )


if __name__ == "__main__":
    try:
        import matplotlib.pyplot as plt

        HAS_MATPLOTLIB = True
    except ImportError:
        HAS_MATPLOTLIB = False
        print("matplotlib not available, skipping visualization")

    # Simple test without ROS dependencies
    print("GazeOptimizer test with velocity-based weighting")

    # Create mock trajectory (11-DOF whole body)
    # Make it more interesting: slow start, fast middle, slow end
    num_waypoints = 50
    trajectory = np.zeros((num_waypoints, 11))

    for i in range(num_waypoints):
        t = i / num_waypoints
        # Create velocity profile: slow->fast->slow using sine
        velocity_profile = np.sin(t * np.pi)  # 0 at start/end, 1 at middle

        trajectory[i, 0] = t * 2.0  # x moves forward
        trajectory[i, 1] = 0.2 * np.sin(t * 4 * np.pi)  # y oscillates slightly
        trajectory[i, 2] = 0.0  # theta stays constant
        trajectory[i, 3] = 0.3  # torso_lift
        trajectory[i, 4] = 0.0  # shoulder_pan
        # Variable speed arm movement
        trajectory[i, 5] = -0.5 - velocity_profile * 1.0  # shoulder_lift (extends down)
        trajectory[i, 6] = 0.0  # upperarm_roll
        trajectory[i, 7] = 1.5 - velocity_profile * 0.5  # elbow_flex
        trajectory[i, 8] = 0.0  # forearm_roll
        trajectory[i, 9] = 1.0  # wrist_flex
        trajectory[i, 10] = 0.0  # wrist_roll

    # Mock head controller (stores targets for visualization)
    class MockRobot:
        def __init__(self):
            self.targets = []

        def point_head_at(self, target_point, frame_id, duration):
            self.targets.append(target_point)

    # Test gaze optimizer
    mock_robot = MockRobot()

    print("\n=== Gaze Optimizer Test (Cartesian velocity weighting) ===")
    optimizer = GazeOptimizer(mock_robot, lookahead_window=20, decay_rate=0.9)
    optimizer.set_trajectory(trajectory)

    # Print velocity statistics for key joints
    print("\n Cartesian Velocity Statistics (world frame, in meters):")
    for joint_name in optimizer.key_joints:
        vels = optimizer.joint_velocities[joint_name]
        print(
            f"  {joint_name:15s}: min={vels.min():.4f}, max={vels.max():.4f}, mean={vels.mean():.4f}"
        )

    # Compute all link positions along trajectory
    ee_positions = []
    all_link_trajectories = {
        "base": [],
        "torso": [],
        "shoulder_pan": [],
        "shoulder_lift": [],
        "upperarm": [],
        "elbow": [],
        "forearm": [],
        "wrist_flex": [],
        "wrist_roll": [],
        "gripper": [],
    }
    test_indices = range(0, num_waypoints - 5, 2)  # Every 2nd waypoint

    for config in trajectory:
        link_positions = optimizer._compute_all_link_positions(config)
        ee_positions.append(link_positions["gripper"])

        # Store all link positions
        for link_name in all_link_trajectories.keys():
            all_link_trajectories[link_name].append(link_positions[link_name])

    for idx in test_indices:
        optimizer.update(idx)

    ee_positions = np.array(ee_positions)

    # Convert link trajectories to numpy arrays
    for link_name in all_link_trajectories.keys():
        all_link_trajectories[link_name] = np.array(all_link_trajectories[link_name])

    # Print trajectory statistics
    print("\n=== Forward Kinematics Verification ===")
    print(f"Trajectory waypoints: {len(ee_positions)}")
    print("End-effector position range:")
    print(f"  X: [{ee_positions[:, 0].min():.3f}, {ee_positions[:, 0].max():.3f}] m")
    print(f"  Y: [{ee_positions[:, 1].min():.3f}, {ee_positions[:, 1].max():.3f}] m")
    print(f"  Z: [{ee_positions[:, 2].min():.3f}, {ee_positions[:, 2].max():.3f}] m")
    print(
        f"\nStart position: [{ee_positions[0, 0]:.3f}, {ee_positions[0, 1]:.3f}, {ee_positions[0, 2]:.3f}] m"
    )
    print(
        f"End position:   [{ee_positions[-1, 0]:.3f}, {ee_positions[-1, 1]:.3f}, {ee_positions[-1, 2]:.3f}] m"
    )

    # Distance from base
    distances_from_base = np.linalg.norm(ee_positions, axis=1)
    print("\nDistance from base (origin):")
    print(f"  Min: {distances_from_base.min():.3f} m")
    print(f"  Max: {distances_from_base.max():.3f} m")
    print(f"  Mean: {distances_from_base.mean():.3f} m")

    # Height above ground
    print("\nHeight above ground (Z):")
    print(f"  Min: {ee_positions[:, 2].min():.3f} m")
    print(f"  Max: {ee_positions[:, 2].max():.3f} m")
    print(f"  Mean: {ee_positions[:, 2].mean():.3f} m")

    if HAS_MATPLOTLIB:
        # Create visualization with essential plots
        fig = plt.figure(figsize=(16, 10))

        # 1. Full arm skeleton at multiple waypoints (3D)
        ax1 = fig.add_subplot(2, 3, 1, projection="3d")

        # Define arm chain (order of links to connect)
        arm_chain = [
            "base",
            "torso",
            "shoulder_pan",
            "shoulder_lift",
            "upperarm",
            "elbow",
            "forearm",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]

        # Show skeleton at selected waypoints
        skeleton_indices = [
            0,
            len(trajectory) // 4,
            len(trajectory) // 2,
            3 * len(trajectory) // 4,
            len(trajectory) - 1,
        ]
        skeleton_colors = ["green", "cyan", "yellow", "orange", "red"]
        skeleton_labels = ["Start", "25%", "50%", "75%", "End"]

        for idx, color, label in zip(
            skeleton_indices, skeleton_colors, skeleton_labels
        ):
            # Get all joint positions for this waypoint
            joint_pos = np.array(
                [all_link_trajectories[link][idx] for link in arm_chain]
            )

            # Draw skeleton
            ax1.plot(
                joint_pos[:, 0],
                joint_pos[:, 1],
                joint_pos[:, 2],
                color=color,
                linewidth=3,
                marker="o",
                markersize=6,
                label=label,
                alpha=0.7,
            )

        # Add gripper trajectory
        ax1.plot(
            ee_positions[:, 0],
            ee_positions[:, 1],
            ee_positions[:, 2],
            "b--",
            linewidth=1,
            alpha=0.3,
            label="Gripper path",
        )

        ax1.set_xlabel("X (m)", fontsize=10)
        ax1.set_ylabel("Y (m)", fontsize=10)
        ax1.set_zlabel("Z (m)", fontsize=10)
        ax1.set_title(
            "Full Arm Configuration (Multiple Waypoints)",
            fontsize=11,
            fontweight="bold",
        )
        ax1.legend(fontsize=8, loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Set equal aspect ratio
        max_range = 1.5
        ax1.set_xlim([-0.5, max_range])
        ax1.set_ylim([-max_range / 2, max_range / 2])
        ax1.set_zlim([0, max_range])

        # 2. All joint trajectories in 3D
        ax2 = fig.add_subplot(2, 3, 2, projection="3d")

        # Plot trajectory for each major joint
        joint_colors = {
            "shoulder_lift": "red",
            "elbow": "orange",
            "wrist_flex": "purple",
            "gripper": "blue",
        }

        for joint_name, joint_color in joint_colors.items():
            positions = all_link_trajectories[joint_name]
            ax2.plot(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                color=joint_color,
                linewidth=2,
                alpha=0.7,
                label=joint_name.replace("_", " ").title(),
            )

        ax2.scatter(0, 0, 0, c="black", s=200, marker="^", label="Base", zorder=10)
        ax2.set_xlabel("X (m)", fontsize=10)
        ax2.set_ylabel("Y (m)", fontsize=10)
        ax2.set_zlabel("Z (m)", fontsize=10)
        ax2.set_title("Joint Trajectories (3D)", fontsize=11, fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 3. Side view showing all joints (X-Z plane)
        ax3 = fig.add_subplot(2, 3, 3)

        for joint_name, joint_color in joint_colors.items():
            positions = all_link_trajectories[joint_name]
            ax3.plot(
                positions[:, 0],
                positions[:, 2],
                color=joint_color,
                linewidth=2,
                alpha=0.7,
                label=joint_name.replace("_", " ").title(),
            )

        # Mark start and end for gripper
        ax3.scatter(
            ee_positions[0, 0],
            ee_positions[0, 2],
            c="green",
            s=150,
            marker="o",
            edgecolors="black",
            linewidth=2,
            zorder=5,
        )
        ax3.scatter(
            ee_positions[-1, 0],
            ee_positions[-1, 2],
            c="red",
            s=150,
            marker="s",
            edgecolors="black",
            linewidth=2,
            zorder=5,
        )
        ax3.scatter(
            0,
            0,
            c="black",
            s=200,
            marker="^",
            edgecolors="white",
            linewidth=2,
            zorder=5,
        )
        ax3.set_xlabel("X (m)", fontsize=10)
        ax3.set_ylabel("Z (m)", fontsize=10)
        ax3.set_title("Side View - All Joints (X-Z)", fontsize=11, fontweight="bold")
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=7, loc="upper left")
        ax3.axis("equal")

        # 4. Front view showing all joints (Y-Z plane)
        ax4 = fig.add_subplot(2, 3, 4)

        for joint_name, joint_color in joint_colors.items():
            positions = all_link_trajectories[joint_name]
            ax4.plot(
                positions[:, 1],
                positions[:, 2],
                color=joint_color,
                linewidth=2,
                alpha=0.7,
                label=joint_name.replace("_", " ").title(),
            )

        # Mark start and end for gripper
        ax4.scatter(
            ee_positions[0, 1],
            ee_positions[0, 2],
            c="green",
            s=150,
            marker="o",
            edgecolors="black",
            linewidth=2,
            zorder=5,
        )
        ax4.scatter(
            ee_positions[-1, 1],
            ee_positions[-1, 2],
            c="red",
            s=150,
            marker="s",
            edgecolors="black",
            linewidth=2,
            zorder=5,
        )
        ax4.scatter(
            0,
            0,
            c="black",
            s=200,
            marker="^",
            edgecolors="white",
            linewidth=2,
            zorder=5,
        )
        ax4.set_xlabel("Y (m)", fontsize=10)
        ax4.set_ylabel("Z (m)", fontsize=10)
        ax4.set_title("Front View - All Joints (Y-Z)", fontsize=11, fontweight="bold")
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=7, loc="upper right")
        ax4.axis("equal")

        # 5. Joint positions over time (to verify smooth motion)
        ax5 = fig.add_subplot(2, 3, 5)
        for joint_name, joint_color in joint_colors.items():
            positions = all_link_trajectories[joint_name]
            # Plot distance from base over time
            distances = np.linalg.norm(positions, axis=1)
            ax5.plot(
                distances,
                color=joint_color,
                linewidth=2,
                label=joint_name.replace("_", " ").title(),
                alpha=0.7,
            )
        ax5.set_xlabel("Waypoint Index", fontsize=10)
        ax5.set_ylabel("Distance from Base (m)", fontsize=10)
        ax5.set_title("Joint Reach Over Time", fontsize=11, fontweight="bold")
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)

        # 6. Cartesian velocity profiles for key joints
        ax6 = fig.add_subplot(2, 3, 6)
        for joint_name, joint_color in joint_colors.items():
            velocities = optimizer.joint_velocities[joint_name]
            ax6.plot(
                velocities,
                color=joint_color,
                linewidth=2,
                label=joint_name.replace("_", " ").title(),
                alpha=0.7,
            )
        ax6.set_xlabel("Waypoint Index", fontsize=10)
        ax6.set_ylabel("Cartesian Velocity (m)", fontsize=10)
        ax6.set_title(
            "Cartesian Velocities (World Frame)", fontsize=11, fontweight="bold"
        )
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)

        # Add overall title
        fig.suptitle(
            "Gaze Optimizer: Forward Kinematics & Trajectory Analysis",
            fontsize=14,
            fontweight="bold",
            y=0.995,
        )

        plt.tight_layout()
        plt.savefig(
            "debug/gaze_optimizer_visualization.png", dpi=150, bbox_inches="tight"
        )
        print("\n✓ Visualization saved to: gaze_optimizer_visualization.png")
        plt.show()
    else:
        print("\n=== Text-based Cartesian velocity profile ===")
        print("Showing gripper velocity (world frame, in meters):")
        print("Index | Velocity")
        gripper_vels = optimizer.joint_velocities["gripper"]
        max_vel = gripper_vels.max()
        for i in range(0, num_waypoints, 5):
            vel = gripper_vels[i]
            normalized = vel / max_vel if max_vel > 0 else 0
            bar = "#" * int(normalized * 20)
            print(f"  {i:2d}  | {vel:.4f}m {bar}")
