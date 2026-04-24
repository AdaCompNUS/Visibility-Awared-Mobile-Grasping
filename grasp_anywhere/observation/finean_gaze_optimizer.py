"""
Finean Gaze Optimizer for the Fetch robot.
Implementation of 'Where Should I Look? Optimised Gaze Control' by Finean et al.

Paper: https://ora.ox.ac.uk/objects/uuid:4c89f0d1-0bd1-4738-b161-b85efb26deaf

This module computes optimal gaze targets by modeling the robot's future swept volume
as a set of spheres (representing the trajectory) and selecting the best next head movement
from a set of discrete motion primitives to maximize viewing of unseen and important areas.

Adaptations:
- Replaces Voxel Grid with sphere-based trajectory tokens to track visibility and staleness.
- Uses greedy search over discrete motion primitives instead of gradient descent.
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from grasp_anywhere.robot.kinematics import forward_kinematics
from grasp_anywhere.utils.logger import log

# Hardcoded sphere parameters from fetch_spherized.urdf
# Format: 'link_name': [(radius, [x, y, z]), ...]
FETCH_SPHERES = {
    "base_link": [
        (0.24, [-0.12, 0.0, 0.182]),
        (0.066, [0.225, 0.0, 0.31]),
        (0.22, [0.08, -0.06, 0.16]),
        (0.066, [0.215, -0.07, 0.31]),
        (0.066, [0.185, -0.135, 0.31]),
        (0.066, [0.13, -0.185, 0.31]),
        (0.066, [0.065, -0.2, 0.31]),
        (0.066, [0.01, -0.2, 0.31]),
        (0.22, [0.08, 0.06, 0.16]),
        (0.066, [0.215, 0.07, 0.31]),
        (0.066, [0.185, 0.135, 0.31]),
        (0.066, [0.13, 0.185, 0.31]),
        (0.066, [0.065, 0.2, 0.31]),
        (0.066, [0.01, 0.2, 0.31]),
    ],
    "torso_lift_link": [
        (0.15, [-0.1, -0.05, 0.15]),
        (0.15, [-0.1, 0.05, 0.15]),
        (0.15, [-0.1, 0.05, 0.3]),
        (0.15, [-0.1, 0.05, 0.45]),
        (0.15, [-0.1, -0.05, 0.45]),
        (0.15, [-0.1, -0.05, 0.3]),
    ],
    "torso_lift_link_collision_2": [
        (0.07, [0.1, 0.0, 0.24]),
    ],
    "head_pan_link": [
        (0.15, [0.0, 0.0, 0.06]),
        (0.05, [0.145, 0.0, 0.058]),
        (0.05, [0.145, -0.0425, 0.058]),
        (0.05, [0.145, 0.0425, 0.058]),
        (0.05, [0.145, 0.085, 0.058]),
        (0.05, [0.145, -0.085, 0.058]),
        (0.03, [0.0625, -0.115, 0.03]),
        (0.03, [0.088, -0.115, 0.03]),
        (0.03, [0.1135, -0.115, 0.03]),
        (0.03, [0.139, -0.115, 0.03]),
        (0.03, [0.0625, -0.115, 0.085]),
        (0.03, [0.088, -0.115, 0.085]),
        (0.03, [0.1135, -0.115, 0.085]),
        (0.03, [0.139, -0.115, 0.085]),
        (0.03, [0.16, -0.115, 0.075]),
        (0.03, [0.168, -0.115, 0.0575]),
        (0.03, [0.16, -0.115, 0.04]),
        (0.03, [0.0625, 0.115, 0.03]),
        (0.03, [0.088, 0.115, 0.03]),
        (0.03, [0.1135, 0.115, 0.03]),
        (0.03, [0.139, 0.115, 0.03]),
        (0.03, [0.0625, 0.115, 0.085]),
        (0.03, [0.088, 0.115, 0.085]),
        (0.03, [0.1135, 0.115, 0.085]),
        (0.03, [0.139, 0.115, 0.085]),
        (0.03, [0.16, 0.115, 0.075]),
        (0.03, [0.168, 0.115, 0.0575]),
        (0.03, [0.16, 0.115, 0.04]),
    ],
    "shoulder_pan_link": [
        (0.055, [0.0, 0.0, 0.0]),
        (0.055, [0.025, -0.015, 0.035]),
        (0.055, [0.05, -0.03, 0.06]),
        (0.055, [0.12, -0.03, 0.06]),
    ],
    "shoulder_lift_link": [
        (0.04, [0.025, 0.04, 0.025]),
        (0.04, [-0.025, 0.04, -0.025]),
        (0.04, [0.025, 0.04, -0.025]),
        (0.04, [-0.025, 0.04, 0.025]),
        (0.055, [0.08, 0.0, 0.0]),
        (0.055, [0.11, 0.0, 0.0]),
        (0.055, [0.14, 0.0, 0.0]),
    ],
    "upperarm_roll_link": [
        (0.055, [-0.02, 0.0, 0.0]),
        (0.055, [0.03, 0.0, 0.0]),
        (0.055, [0.08, 0.0, 0.0]),
        (0.03, [0.11, -0.045, 0.02]),
        (0.03, [0.11, -0.045, -0.02]),
        (0.03, [0.155, -0.045, 0.02]),
        (0.03, [0.155, -0.045, -0.02]),
        (0.055, [0.13, 0.0, 0.0]),
    ],
    "elbow_flex_link": [
        (0.03, [0.02, 0.045, 0.02]),
        (0.03, [0.02, 0.045, -0.02]),
        (0.03, [-0.02, 0.045, 0.02]),
        (0.03, [-0.02, 0.045, -0.02]),
        (0.055, [0.08, 0.0, 0.0]),
        (0.055, [0.14, 0.0, 0.0]),
    ],
    "forearm_roll_link": [
        (0.055, [0.0, 0.0, 0.0]),
        (0.03, [0.05, -0.06, 0.02]),
        (0.03, [0.05, -0.06, -0.02]),
        (0.03, [0.1, -0.06, 0.02]),
        (0.03, [0.1, -0.06, -0.02]),
        (0.03, [0.15, -0.06, 0.02]),
        (0.03, [0.15, -0.06, -0.02]),
    ],
    "wrist_flex_link": [
        (0.055, [0.0, 0.0, 0.0]),
        (0.055, [0.06, 0.0, 0.0]),
        (0.03, [0.02, 0.045, 0.02]),
        (0.03, [0.02, 0.045, -0.02]),
        (0.03, [-0.02, 0.045, 0.02]),
        (0.03, [-0.02, 0.045, -0.02]),
    ],
    "wrist_roll_link": [
        (0.055, [-0.03, 0.0, 0.0]),
        (0.055, [0.0, 0.0, 0.0]),
    ],
    "gripper_link": [
        (0.05, [-0.07, 0.02, 0.0]),
        (0.05, [-0.07, -0.02, 0.0]),
        (0.05, [-0.1, 0.02, 0.0]),
        (0.05, [-0.1, -0.02, 0.0]),
    ],
    "r_gripper_finger_link": [
        (0.012, [0.017, -0.0085, -0.005]),
        (0.012, [0.017, -0.0085, 0.005]),
        (0.012, [0.0, -0.0085, -0.005]),
        (0.012, [0.0, -0.0085, 0.005]),
        (0.012, [-0.017, -0.0085, -0.005]),
        (0.012, [-0.017, -0.0085, 0.005]),
    ],
    "l_gripper_finger_link": [
        (0.012, [0.017, 0.0085, -0.005]),
        (0.012, [0.017, 0.0085, 0.005]),
        (0.012, [0.0, 0.0085, -0.005]),
        (0.012, [0.0, 0.0085, 0.005]),
        (0.012, [-0.017, 0.0085, -0.005]),
        (0.012, [-0.017, 0.0085, 0.005]),
    ],
    "torso_fixed_link": [
        (0.12, [-0.1, -0.07, 0.35]),
        (0.12, [-0.1, 0.07, 0.35]),
        (0.12, [-0.1, -0.07, 0.2]),
        (0.12, [-0.1, 0.07, 0.2]),
        (0.12, [-0.1, 0.07, 0.07]),
        (0.12, [-0.1, -0.07, 0.07]),
    ],
}


class FineanGazeOptimizer:
    """
    Optimizes robot head gaze based on future trajectory using differentiable rendering (forward pass only)
    and greedy motion primitive selection.
    """

    def __init__(
        self,
        robot,
        lookahead_window=40,
        device=None,
        fov_deg=60.0,
        img_size=(64, 48),  # Low res for fast optimization
    ):
        """
        Initialize the Finean gaze optimizer.

        Args:
            robot: The Fetch robot instance (interface to hardware/sim).
            lookahead_window: Number of future waypoints to consider.
            device: torch.device (cpu or cuda). Auto-detects if None.
            fov_deg: Camera Field of View in degrees.
            img_size: (width, height) for differentiable rendering grid.
        """
        self.robot = robot
        self.lookahead_window = lookahead_window
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.img_W, self.img_H = img_size
        self.fov_deg = fov_deg

        # Camera intrinsics (approximated for optimization)
        # Assuming square pixels
        f = (self.img_W / 2.0) / np.tan(np.deg2rad(fov_deg / 2.0))
        self.K = torch.tensor(
            [[f, 0, self.img_W / 2.0], [0, f, self.img_H / 2.0], [0, 0, 1.0]],
            device=self.device,
            dtype=torch.float32,
        )

        # Head kinematic constants (from Fetch URDF)
        # torso_lift -> head_pan
        self.trans_torso_headpan = torch.tensor(
            [0.053125, 0.0, 0.603001], dtype=torch.float32, device=self.device
        )
        # head_pan -> head_tilt
        self.trans_headpan_headtilt = torch.tensor(
            [0.14253, 0.0, 0.058], dtype=torch.float32, device=self.device
        )
        # head_tilt -> camera (approximate, usually camera is slightly forward/up from tilt axis)
        self.trans_headtilt_camera = torch.tensor(
            [0.1, 0.0, 0.05], dtype=torch.float32, device=self.device
        )

        # Use hardcoded spheres
        self.link_spheres = FETCH_SPHERES
        self.all_links = sorted(list(self.link_spheres.keys()))

        # Fixed transform for torso_fixed_link if missing from FK
        # relative to base_link: xyz="-0.086875 0 0.377425"
        self.T_base_torsofixed = np.eye(4)
        self.T_base_torsofixed[:3, 3] = [-0.086875, 0.0, 0.377425]

        # State tracking
        self.trajectory_sphere_staleness = None  # Tensor: [T, TotalSpheres]

        self.trajectory_spheres = None  # Tensor: [T, TotalSpheres, 3] (x,y,z)
        self.sphere_radii = None
        self.sphere_link_indices = None
        self.trajectory_data = None

        # Warm start state
        self.last_pan = 0.0
        self.last_tilt = 0.5

        # Motion Primitives (angular velocity * dt)
        # Assuming dt=0.2s approx for update loop, max speed ~1.0 rad/s
        step = 0.15
        self.primitives = [
            (0.0, 0.0),  # Stay
            (step, 0.0),  # Pan Left
            (-step, 0.0),  # Pan Right
            (0.0, step),  # Tilt Up
            (0.0, -step),  # Tilt Down
            (step, step),
            (step, -step),
            (-step, step),
            (-step, -step),
        ]

        log.info(
            f"[FineanGazeOptimizer] Initialized on {self.device} with "
            f"{sum(len(s) for s in self.link_spheres.values())} spheres."
        )

    def set_trajectory(self, trajectory):
        """
        Precompute the sphere representations for the entire future trajectory.

        Args:
            trajectory: Nx11 numpy array [x,y,theta, torso, <7 arm joints>]
        """
        if trajectory is None or len(trajectory) == 0:
            return

        self.trajectory_data = np.array(trajectory)
        traj_len = len(trajectory)

        # Flatten spheres structure for batch processing
        flat_sphere_defs = []
        for link_idx, link_name in enumerate(self.all_links):
            for radius, xyz in self.link_spheres[link_name]:
                flat_sphere_defs.append((link_idx, radius, np.array(xyz)))

        batch_positions = []
        all_sphere_radii = []
        all_link_indices = []

        # Helper to extract radii/indices only once
        first_pass = True

        for t in range(traj_len):
            config = trajectory[t]
            # config: [x, y, theta, torso, 7 arm joints]
            x, y, theta = config[0], config[1], config[2]

            # Build full joints for FK [torso, 7 arm, head_pan(0), head_tilt(0)]
            arm_joints = config[3:]  # [torso, 7 arm] (length 8)
            full_joints = np.concatenate([arm_joints, [0.0, 0.0]])

            link_poses = forward_kinematics(full_joints)

            # Handle potentially missing fixed links (like torso_fixed_link)
            if "base_link" in link_poses and "torso_fixed_link" not in link_poses:
                link_poses["torso_fixed_link"] = (
                    link_poses["base_link"] @ self.T_base_torsofixed
                )

            R_map_base = R.from_euler("z", theta).as_matrix()
            t_map_base = np.array([x, y, 0])

            current_t_positions = []

            for link_idx, radius, offset in flat_sphere_defs:
                link_name = self.all_links[link_idx]

                # Default to identity/origin if link not in FK
                T_bl = link_poses.get(link_name, np.eye(4))

                # Sphere pos in base frame
                # T_bl is 4x4
                pos_base = T_bl[:3, 3] + T_bl[:3, :3] @ offset

                # Transform to map frame
                pos_map = t_map_base + R_map_base @ pos_base
                current_t_positions.append(pos_map)

                if first_pass:
                    all_sphere_radii.append(radius)
                    all_link_indices.append(link_idx)

            batch_positions.append(np.array(current_t_positions))  # Optimization
            first_pass = False

        # Convert to tensors (Optimized from numpy array of arrays)
        self.trajectory_spheres = torch.tensor(
            np.array(batch_positions), dtype=torch.float32, device=self.device
        )  # [T, N, 3]
        self.sphere_radii = torch.tensor(
            all_sphere_radii, dtype=torch.float32, device=self.device
        )  # [N]
        self.sphere_link_indices = torch.tensor(
            all_link_indices, dtype=torch.long, device=self.device
        )  # [N]

        # Initialize Staleness [T, N]
        # 1.0 = Freshly Stale (Has not been seen yet)
        self.trajectory_sphere_staleness = torch.ones(
            (traj_len, len(all_sphere_radii)), dtype=torch.float32, device=self.device
        )

    def _calc_loss(
        self,
        head_pan,
        head_tilt,
        base_pose,
        torso_lift,
        spheres,
        radii,
        stale_weights,
    ):
        """
        Calculate differentiable loss (negative score): maximize weighted visibility.
        Used for evaluation in greedy search.
        """
        zero = torch.tensor(0.0, device=self.device)
        one = torch.tensor(1.0, device=self.device)

        # Transform Matrices
        # T_map_base
        theta = base_pose[2]
        c = torch.cos(theta)
        s = torch.sin(theta)
        R_mb = torch.stack(
            [
                torch.stack([c, -s, zero]),
                torch.stack([s, c, zero]),
                torch.stack([zero, zero, one]),
            ]
        )
        t_mb = torch.stack([base_pose[0], base_pose[1], zero])

        # T_base_torso (origin + lift)
        t_bt = torch.tensor(
            [-0.086875, 0.0, 0.37743], device=self.device
        ) + torch.stack([zero, zero, torso_lift])

        # T_torso_headpan
        t_thp = self.trans_torso_headpan
        c_hp = torch.cos(head_pan)
        s_hp = torch.sin(head_pan)
        R_hp = torch.stack(
            [
                torch.stack([c_hp, -s_hp, zero]),
                torch.stack([s_hp, c_hp, zero]),
                torch.stack([zero, zero, one]),
            ]
        )

        # T_headpan_headtilt
        t_hpt = self.trans_headpan_headtilt
        c_ht = torch.cos(head_tilt)
        s_ht = torch.sin(head_tilt)
        R_ht = torch.stack(
            [
                torch.stack([c_ht, zero, s_ht]),
                torch.stack([zero, one, zero]),
                torch.stack([-s_ht, zero, c_ht]),
            ]
        )

        # T_headtilt_camera
        t_htc = self.trans_headtilt_camera
        # Optical frame rotation (Z forward, X right, Y down)
        R_opt = torch.tensor(
            [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], device=self.device
        )

        # Forward Transform Points (Map -> Cam)
        P = spheres.T  # 3 x N

        # Inverse transforms applied sequentially to points
        P = R_mb.T @ (P - t_mb.unsqueeze(1))
        P = P - t_bt.unsqueeze(1)
        P = R_hp.T @ (P - t_thp.unsqueeze(1))
        P = R_ht.T @ (P - t_hpt.unsqueeze(1))
        P = P - t_htc.unsqueeze(1)
        P_cam = R_opt @ P  # 3 x N

        # Projection
        Z = P_cam[2, :]
        mask_front = (Z > 0.1).float()

        X = P_cam[0, :]
        Y = P_cam[1, :]

        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        denom = Z + 1e-6
        u = fx * X / denom + cx
        v = fy * Y / denom + cy

        # Differentiable Visibility (Soft Box)
        # Sigmoid slope determines softness
        def soft_window(vals, limit, slope=0.2):
            return torch.sigmoid(vals * slope) * torch.sigmoid((limit - vals) * slope)

        vis_u = soft_window(u, self.img_W)
        vis_v = soft_window(v, self.img_H)

        # Weights
        w_stale = stale_weights

        # Gaussian Area Weight (projected radius)
        # Larger spheres or closer spheres cover more volume
        r_screen = fx * radii / denom
        w_vol = r_screen

        # Total Weight
        # We want to maximize the sum of weights that are VISIBLE.
        # score = weighting_terms * visibility_terms
        score = (w_stale * w_vol) * (vis_u * vis_v * mask_front)

        # Returns negative score for minimization (or maximization logic)
        loss = -torch.sum(score)
        return loss, u, v, mask_front

    def update(self, current_index):
        """
        Greedy optimization update step.
        """
        if self.trajectory_spheres is None or self.trajectory_data is None:
            return

        # Data Slice
        T_total = self.trajectory_spheres.shape[0]
        if current_index >= T_total:
            current_index = T_total - 1

        # Skip immediate self-occlusion/singularities (Z~0)
        start_idx = min(current_index + 2, T_total - 1)
        end_idx = min(current_index + self.lookahead_window, T_total)

        if start_idx >= end_idx:
            return

        # Prepare batch data for evaluation
        # [Points, 3]
        points = self.trajectory_spheres[start_idx:end_idx].reshape(-1, 3)
        radii = self.sphere_radii.repeat(end_idx - start_idx)

        # Staleness view
        stale_slice = self.trajectory_sphere_staleness[start_idx:end_idx].view(-1)

        # Robot State
        config = self.trajectory_data[current_index]
        current_base_pose = torch.tensor(
            config[:3], dtype=torch.float32, device=self.device
        )
        current_torso_height = torch.tensor(
            config[3], dtype=torch.float32, device=self.device
        )

        # Greedy Search
        best_score = -float("inf")
        best_pan = self.last_pan
        best_tilt = self.last_tilt

        # Evaluate all primitives
        for d_pan, d_tilt in self.primitives:
            cand_pan = self.last_pan + d_pan
            cand_tilt = self.last_tilt + d_tilt

            # Clamp to limits
            cand_pan = np.clip(cand_pan, -1.57, 1.57)
            cand_tilt = np.clip(cand_tilt, -0.76, 1.45)

            # Convert to tensor for calculation
            t_pan = torch.tensor(cand_pan, dtype=torch.float32, device=self.device)
            t_tilt = torch.tensor(cand_tilt, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                # _calc_loss returns negative score
                loss, _, _, _ = self._calc_loss(
                    t_pan,
                    t_tilt,
                    current_base_pose,
                    current_torso_height,
                    points,
                    radii,
                    stale_slice,
                )
                score = -loss.item()

            if score > best_score:
                best_score = score
                best_pan = cand_pan
                best_tilt = cand_tilt

        self.last_pan = best_pan
        self.last_tilt = best_tilt

        # Command Robot
        self.robot.move_head(pan=best_pan, tilt=best_tilt, duration=0.2)
