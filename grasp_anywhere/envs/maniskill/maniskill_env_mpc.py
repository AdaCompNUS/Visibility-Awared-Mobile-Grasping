import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from mani_skill.sensors.camera import Camera
from mani_skill.utils.structs.types import GPUMemoryConfig, SceneConfig, SimConfig

from grasp_anywhere.benchmark.dynamic_benchmark_manager import DynamicBenchmarkManager
from grasp_anywhere.envs.base import RobotEnv
from grasp_anywhere.utils.logger import log
from grasp_anywhere.utils.monitor_core import eval_monitor_contacts, update_hold_state


class ManiSkillEnv(RobotEnv):
    """
    Velocity-control version of ManiSkill Fetch controller using a simple MPC.

    Goals:
    - Preserve the public interface and behavior of the existing ManiSkill.
    - Use a waypoint queue (arm + base) but select the nearest waypoint to the
      current state at each control tick.
    - Compute whole-body velocities (base v,w; torso vel; 7 arm joint vels)
      via a simple quadratic MPC approximation and send them directly to
      ManiSkill velocity control interfaces.
    - Keep the rest (RGBD, joint state accessors, etc.) the same.
    """

    def __init__(
        self,
        env_id: str,
        robot_uids: str = "fetch",
        obs_mode: str = "rgb+depth+state+segmentation",
        control_mode: str = "pd_joint_vel",
        render_mode: Optional[str] = "human",
        camera_width: int = 640,
        camera_height: int = 480,
    ):
        # Force joint velocity control by default (caller may override if needed)
        self.control_mode = control_mode
        self.env = gym.make(
            env_id,
            robot_uids=robot_uids,
            obs_mode=obs_mode,
            control_mode=control_mode,
            render_mode=render_mode,
            sensor_configs={"width": camera_width, "height": camera_height},
            sim_config=SimConfig(
                gpu_memory_config=GPUMemoryConfig(
                    found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
                ),
                scene_config=SceneConfig(contact_offset=0.001),
            ),
        )
        # Assuming the agent is the first one
        self.agent = self.env.unwrapped.agent
        obs, _ = self.env.reset()
        self.obs = obs
        self._obs_logged = False

        # Build action slices in Fetch controller order: arm, gripper, body, base
        self.action_slices = {}
        current_index = 0
        for name in ["arm", "gripper", "body", "base"]:
            controller = self.agent.controller.controllers.get(name)
            if controller is None:
                continue
            action_dim = controller.action_space.shape[0]
            self.action_slices[name] = slice(current_index, current_index + action_dim)
            current_index += action_dim

        # Draw an initial frame if using a human viewer
        if getattr(self.env, "render_mode", None) == "human":
            self.env.render()

        # Background stepping thread state
        self._overlay_once: Dict[str, np.ndarray] = {}
        self._overlay_persistent: Dict[str, np.ndarray] = {}
        self._last_step: Tuple[Any, float, bool, bool, Dict[str, Any]] = (
            self.obs,
            0.0,
            False,
            False,
            {},
        )
        self._lock = threading.RLock()
        self._env_lock = threading.RLock()
        self._running = True
        self._tick_hz = 30.0

        # Waypoint stream: merged whole-body waypoints [x, y, th, torso, 7 arm]
        self._merged_traj: List[np.ndarray] = []
        self._last_waypoint_idx: int = 0

        # Limits for velocity commands (keep simple, tunable)
        self._v_limits = {
            "v_max": 5.0,
            "w_max": 5.0,
            # order: torso, 7 arm joints
            "joint_vel_max": np.array(
                [2.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0], dtype=np.float32
            ),
        }

        # Simple MPC parameters
        self._mpc = {
            "N": 100,  # horizon (used for reference extraction; control is 1-step LQR approx)
            "dt": 0.5,  # fixed MPC discretization for stronger actions
            # state weights for [x, y, th, torso, 7 arm]
            "Q": np.array([20.0, 20.0, 15.0] + [12.0] * 8, dtype=np.float32),
            # control weights for [v, w, torso_vel, 7 arm vels]
            "R": np.array([0.5, 0.8] + [1.0] * 8, dtype=np.float32),
            "gain": 2.5,
        }

        # Gaze (head pan/tilt) absolute target and PD follower (velocity controller)
        # Values are absolute targets; NaN means "no target set/maintain current".
        self._body_abs_target: np.ndarray = np.array(
            [np.nan, np.nan, np.nan], dtype=np.float32
        )
        self._gaze_pd = {
            "kp_pan": 3.0,
            "kp_tilt": 3.0,
            "kp_torso": 10.0,
            "max_pan_vel": 2.0,
            "max_tilt_vel": 2.0,
            "max_torso_vel": float(self._v_limits["joint_vel_max"][0]),
        }

        # Gripper target: 0.0 = closed (0.0m), 1.0 = open (0.05m)
        self._gripper_target: float = 0.0

        # Map qpos base to world by initial offset (consistent with original)
        self._qpos_base_indices: Optional[Tuple[int, int, int]] = None
        self._qpos_base_offset: np.ndarray = np.zeros(3, dtype=np.float32)
        self._init_qpos_world_offset()

        self._monitor_enabled = False
        self._target_object_name = ""
        self._collision_detected = False
        self._collision_pairs: set[tuple[str, str]] = set()
        self._success_detected = False
        self._gripper_keywords = ["gripper", "finger", "wrist"]
        # Monitoring defaults (overwritten by start_monitoring(...), usually from YAML config)
        self._monitor_contact_force_threshold = 0.001

        # Optional "hold stable" monitoring (disabled when _hold_seconds == 0.0)
        self._hold_seconds = 0.0
        # Hold counting is explicitly armed (e.g., after lift) to avoid counting from initial contact.
        self._hold_armed = False
        self._hold_success_detected = False
        self._hold_state = {
            "active": False,
            "t0": 0.0,
        }
        self._hold_target_actor = None
        self._hold_gripper_link = None

        # Initialize Benchmark Manager
        self.benchmark_manager = DynamicBenchmarkManager(self)

        # Trajectory recording (disabled by default)
        self._record_trajectory = False
        self._trajectory_buffer: List[Dict[str, Any]] = []

        self._bg_thread = threading.Thread(target=self._background_loop, daemon=True)
        self._bg_thread.start()

    def close(self):
        self._running = False
        if self._bg_thread.is_alive():
            self._bg_thread.join(timeout=1.0)
        self.env.close()

    def _background_loop(self):
        zero_action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        while self._running:
            # Build an action consisting purely of velocities for each slice.
            action = zero_action.copy()

            # Compute whole-body velocity command via simple MPC if we have waypoints
            u_cmd = None
            with self._lock:
                if len(self._merged_traj) > 0:
                    # Current full state
                    x_state = self._get_current_whole_body_state()
                    # Select nearest waypoint index (from last pointer forward)
                    idx = self._find_nearest_waypoint_index(x_state)
                    self._last_waypoint_idx = idx
                    # Construct a short reference sequence (N+1) starting from idx+1 (look-ahead)
                    ref_seq = self._construct_reference(idx, x_state)
                    # Use a small look-ahead target to be more aggressive
                    k = min(2, int(self._mpc["N"]))
                    x_ref = ref_seq[:, k]
                    # Compute first-step control using linearized one-step QP/LQR
                    u_cmd = self._solve_one_step_mpc(x_state, x_ref)

            # Store MPC torso velocity for use in body controller
            mpc_torso_vel = None

            # If not computed, fall back to persistent overlays (e.g., manual base/gripper)
            if u_cmd is None:
                # Base
                if "base" in self.action_slices:
                    sl = self.action_slices["base"]
                    if "base" in self._overlay_persistent:
                        action[sl] = self._overlay_persistent["base"].astype(np.float32)
                    else:
                        action[sl] = np.zeros((sl.stop - sl.start,), dtype=np.float32)
                # Arm/body/gripper fall back to zeros or one-shot overlays below

            else:
                # Map u_cmd -> action slices
                # u: [v, w, torso_vel, 7 arm vels]
                v = float(u_cmd[0])
                w = float(u_cmd[1])
                mpc_torso_vel = float(u_cmd[2])
                arm_vels = np.array(u_cmd[3:], dtype=np.float32)

                # Base velocities [v, w]
                if "base" in self.action_slices:
                    sl = self.action_slices["base"]
                    action[sl] = np.array([v, w], dtype=np.float32)

                # Arm joint velocities (7)
                if "arm" in self.action_slices:
                    sl = self.action_slices["arm"]
                    out = np.zeros((sl.stop - sl.start,), dtype=np.float32)
                    out[: min(len(out), 7)] = arm_vels[: min(len(out), 7)]
                    action[sl] = out

            # Body (head pan/tilt + torso) handling: simple PD on absolute targets.
            if "body" in self.action_slices:
                sl = self.action_slices["body"]
                # Incorporate one-shot absolute targets for body (persist as target)
                with self._lock:
                    if "body" in self._overlay_once:
                        incoming = np.array(
                            self._overlay_once.pop("body"), dtype=np.float32
                        )
                        mask = ~np.isnan(incoming)
                        self._body_abs_target[mask] = incoming[mask]

                # Read current body joints
                current_pan, current_tilt, current_torso = self.get_joint_positions(
                    self.agent.body_joint_names
                )

                # Desired absolute targets
                targ_pan, targ_tilt, targ_torso = [
                    float(v) for v in self._body_abs_target.tolist()
                ]

                # Gains and limits
                gp = self._gaze_pd
                targ_pan_safe = float(
                    np.where(np.isnan(targ_pan), float(current_pan), float(targ_pan))
                )
                targ_tilt_safe = float(
                    np.where(np.isnan(targ_tilt), float(current_tilt), float(targ_tilt))
                )
                pan_err = float(targ_pan_safe - float(current_pan))
                tilt_err = float(targ_tilt_safe - float(current_tilt))
                pan_vel = float(
                    np.clip(
                        gp["kp_pan"] * pan_err, -gp["max_pan_vel"], gp["max_pan_vel"]
                    )
                )
                tilt_vel = float(
                    np.clip(
                        gp["kp_tilt"] * tilt_err,
                        -gp["max_tilt_vel"],
                        gp["max_tilt_vel"],
                    )
                )

                # Torso velocity: simple PD toward absolute target if set; otherwise zero (uses current as target)
                targ_torso_safe = float(
                    np.where(
                        np.isnan(targ_torso), float(current_torso), float(targ_torso)
                    )
                )
                torso_err = float(targ_torso_safe - float(current_torso))
                torso_vel = float(
                    np.clip(
                        gp["kp_torso"] * torso_err,
                        -gp["max_torso_vel"],
                        gp["max_torso_vel"],
                    )
                )

                # Use MPC torso velocity if available, otherwise use PD
                if mpc_torso_vel is not None:
                    torso_vel = mpc_torso_vel

                # Compose body output [pan_vel, tilt_vel, torso_vel]
                action[sl] = np.array([pan_vel, tilt_vel, torso_vel], dtype=np.float32)

            # Gripper: position control (action space is [-1, 1])
            # 0.0 (closed) -> -1.0, 1.0 (open) -> 1.0
            if "gripper" in self.action_slices:
                sl = self.action_slices["gripper"]
                gripper_action = 2.0 * self._gripper_target - 1.0
                action[sl] = np.array([gripper_action], dtype=np.float32)

            # Apply one-shot overlays (for any remaining controllers)
            with self._lock:
                for name, value in self._overlay_once.items():
                    sl = self.action_slices.get(name)
                    if sl is not None:
                        action[sl] = value
                self._overlay_once.clear()

            # Step environment
            # Safety check: replace any NaN in action before sending
            action = np.nan_to_num(action, nan=0.0)
            with self._env_lock:
                obs, reward, terminated, truncated, info = self.env.step(action)
                dt = self.env.unwrapped.scene.px.timestep

            with self._lock:
                self.obs = obs
                self._last_step = (
                    obs,
                    float(reward),
                    bool(terminated),
                    bool(truncated),
                    info,
                )

                # Update Dynamic Benchmark Manager (moved here to avoid deadlock)
                # If x_state has instantiated
                base_position = self._get_current_whole_body_state()[:3]  # x, y, theta

                # Extract commanded v, w from the action actually sent (handles overlays/MPC both)
                base_v, base_w = 0.0, 0.0
                if "base" in self.action_slices:
                    sl = self.action_slices["base"]
                    # action is numpy array, slice gives us the 2 base elems
                    # v, w corresponds to index 0, 1 in the base slice
                    base_vals = action[sl]
                    base_v, base_w = float(base_vals[0]), float(base_vals[1])

                base_velocity = np.array([base_v, base_w])
                self.benchmark_manager.update(dt, base_position, base_velocity)

                if self._monitor_enabled and self._target_object_name:
                    # robot_link_names = {link.name for link in self.agent.robot.links}
                    contacts = self.env.unwrapped.scene.get_contacts()
                    dt = self.env.unwrapped.scene.px.timestep
                    success_events, collision_events = eval_monitor_contacts(
                        contacts=contacts,
                        target_object_name=self._target_object_name,
                        dt=dt,
                        force_threshold=float(self._monitor_contact_force_threshold),
                        robot_name_substr="fetch",
                        base_link_exclude="base_link",
                    )

                    for robot_actor_name, env_actor_name in success_events:
                        print(
                            f"[MONITOR] SUCCESS DETECTED: {robot_actor_name} contacted {env_actor_name}"
                        )
                        self._success_detected = True

                    for robot_actor_name, env_actor_name in collision_events:
                        print(
                            f"[MONITOR] COLLISION DETECTED: {robot_actor_name} contacted {env_actor_name}"
                        )
                        self._collision_detected = True
                        self._collision_pairs.add((robot_actor_name, env_actor_name))

                    # Optional hold-stability metric: held without slipping for N seconds.
                    if (
                        self._hold_armed
                        and float(self._hold_seconds) > 0.0
                        and self._hold_target_actor is not None
                        and self._hold_gripper_link is not None
                        and not self._hold_success_detected
                    ):
                        has_contact = len(success_events) > 0

                        now_s = time.time()
                        self._hold_state, hold_ok = update_hold_state(
                            state=self._hold_state,
                            now_s=now_s,
                            has_gripper_target_contact=has_contact,
                            hold_seconds=float(self._hold_seconds),
                        )
                        if hold_ok:
                            print(
                                f"[MONITOR] HOLD SUCCESS: held target stably for {float(self._hold_seconds):.2f}s"
                            )
                            self._hold_success_detected = True

                # Record trajectory if enabled
                if self._record_trajectory:
                    state = self._get_current_whole_body_state()
                    self._trajectory_buffer.append(
                        {
                            "timestamp": time.time(),
                            "state": state.tolist(),  # [x, y, th, torso, 7 arm joints]
                            "action": action.tolist(),
                        }
                    )

            if getattr(self.env, "render_mode", None) == "human" and self._running:
                with self._env_lock:
                    self.env.render()
            time.sleep(1.0 / self._tick_hz)

    # ===== Sensors and state access (unchanged APIs) =====
    def _get_head_camera(self) -> Camera:
        sensors = self.env.unwrapped._sensors
        camera = sensors.get("fetch_head")
        if camera is None or not isinstance(camera, Camera):
            raise RuntimeError(
                "Could not find 'fetch_head' camera sensor on the environment."
            )
        return camera

    def reset(self, seed=None):
        with self._env_lock:
            # Reset benchmark manager to clear dynamic obstacles from the scene
            self.benchmark_manager.reset()
            obs, info = self.env.reset(seed=seed)
        with self._lock:
            self.obs = obs
            self._overlay_once.clear()
            self._overlay_persistent.clear()
            self._last_step = (obs, 0.0, False, False, {})
            # Reset MPC trajectory pointers
            self._merged_traj.clear()
            self._last_waypoint_idx = 0
            self._init_qpos_world_offset()
            # Clear stored absolute body targets (gaze)
            self._body_abs_target[:] = np.nan
            # Reset gripper target to current position
            fingers = self.get_joint_positions(
                ["r_gripper_finger_joint", "l_gripper_finger_joint"]
            )
            current_opening = sum(fingers) / 2.0
            self._gripper_target = current_opening / 0.05
        return obs, info

    def get_joint_states(self) -> Tuple[List[str], List[float]]:
        with self._lock:
            qpos = self.obs["state"]
            if isinstance(qpos, torch.Tensor):
                qpos = qpos.detach().cpu().numpy()
            if qpos.ndim == 2:
                qpos = qpos[0]
        with self._env_lock:
            joint_names = [j.name for j in self.agent.robot.active_joints]
        return joint_names, qpos.tolist()

    def get_rgb(self) -> np.ndarray:
        with self._lock:
            rgb = self.obs["sensor_data"]["fetch_head"]["rgb"]
            if isinstance(rgb, torch.Tensor):
                rgb = rgb.detach().cpu().numpy()
            return rgb[0]

    def get_depth(self) -> np.ndarray:
        with self._lock:
            depth = self.obs["sensor_data"]["fetch_head"]["depth"]
            if isinstance(depth, torch.Tensor):
                depth = depth.detach().cpu().numpy()
            depth_mm = depth[0, ..., 0]
            return depth_mm.astype(np.float32) / 1000.0

    def get_camera_intrinsics(self) -> np.ndarray:
        with self._env_lock:
            camera = self._get_head_camera()
            K = np.array(camera.get_params()["intrinsic_cv"])
            if K.ndim == 3:
                return K[0]
            return K

    def _init_qpos_world_offset(self) -> None:
        joint_names = [j.name for j in self.agent.robot.active_joints]
        base_joint_names = self.agent.base_joint_names
        ix = joint_names.index(base_joint_names[0])
        iy = joint_names.index(base_joint_names[1])
        ith = joint_names.index(base_joint_names[2])
        self._qpos_base_indices = (ix, iy, ith)

        qpos = self.obs["state"]
        if isinstance(qpos, torch.Tensor):
            qpos = qpos.detach().cpu().numpy()
        if qpos.ndim == 2:
            qpos = qpos[0]
        qx = float(qpos[ix])
        qy = float(qpos[iy])
        qth = float(qpos[ith])

        base_link = None
        for link in self.agent.robot.links:
            if link.name == "base_link":
                base_link = link
                break
        pose = base_link.pose if base_link is not None else self.agent.robot.pose
        T = pose.to_transformation_matrix()
        if isinstance(T, torch.Tensor):
            T = T.detach().cpu().numpy()
        else:
            T = np.array(T)
        if T.ndim == 3:
            T = T[0]
        wx = float(T[0, 3])
        wy = float(T[1, 3])
        wth = float(np.arctan2(T[1, 0], T[0, 0]))
        dth = float(np.arctan2(np.sin(wth - qth), np.cos(wth - qth)))
        self._qpos_base_offset = np.array([wx - qx, wy - qy, dth], dtype=np.float32)

    def get_base_pose(
        self, world_frame: str = "map", robot_base_frame: str = "base_link"
    ) -> Tuple[float, float, float]:
        with self._lock:
            qpos = self.obs["state"]
            if isinstance(qpos, torch.Tensor):
                qpos = qpos.detach().cpu().numpy()
            if qpos.ndim == 2:
                qpos = qpos[0]
            ix, iy, ith = self._qpos_base_indices
            qx = float(qpos[ix])
            qy = float(qpos[iy])
            qth = float(qpos[ith])
            dx, dy, dth = [float(v) for v in self._qpos_base_offset]
            th = float(np.arctan2(np.sin(qth + dth), np.cos(qth + dth)))
            x = qx + dx
            y = qy + dy
            return (x, y, th)

    def get_camera_pose(
        self, camera_frame: str = "head_camera_rgb_optical_frame"
    ) -> np.ndarray:
        with self._env_lock:
            camera = self._get_head_camera()
            pose = camera.pose.to_transformation_matrix()
            log.debug(f"Camera pose: {pose}")
            return pose

    def get_sensor_snapshot(self) -> Optional[Dict[str, Any]]:
        """
        Atomically get depth, intrinsics, and joint states from same observation.
        This prevents misalignment between sensor data and robot state.
        """
        with self._lock:
            # Read ALL state from current obs atomically
            obs_snapshot = self.obs

            # Extract depth
            depth_data = obs_snapshot["sensor_data"]["fetch_head"]["depth"]
            if isinstance(depth_data, torch.Tensor):
                depth_data = depth_data.detach().cpu().numpy()
            depth = depth_data[0, ..., 0].astype(np.float32) / 1000.0

            # Extract RGB
            rgb_data = obs_snapshot["sensor_data"]["fetch_head"]["rgb"]
            if isinstance(rgb_data, torch.Tensor):
                rgb_data = rgb_data.detach().cpu().numpy()
            rgb = rgb_data[0]

            # Extract joint states
            qpos = obs_snapshot["state"]
            if isinstance(qpos, torch.Tensor):
                qpos = qpos.detach().cpu().numpy()
            if qpos.ndim == 2:
                qpos = qpos[0]

            # Extract segmentation
            segmentation_data = obs_snapshot["sensor_data"]["fetch_head"][
                "segmentation"
            ]
            if isinstance(segmentation_data, torch.Tensor):
                segmentation_data = segmentation_data.detach().cpu().numpy()
            segmentation = segmentation_data[0]

        # Get joint names (doesn't depend on obs)
        with self._env_lock:
            joint_names = [j.name for j in self.agent.robot.active_joints]

        # Get intrinsics (camera parameters are static)
        intrinsics = self.get_camera_intrinsics()

        return {
            "rgb": rgb,
            "depth": depth,
            "segmentation": segmentation,
            "intrinsics": intrinsics,
            "joint_states": (joint_names, qpos.tolist()),
        }

    # ===== Motion APIs (same signatures) =====
    def execute_whole_body_motion(self, arm_path: List, base_configs: List) -> bool:
        if len(arm_path) != len(base_configs):
            raise ValueError("Arm path and base configs must have the same length.")

        merged: List[np.ndarray] = []
        for i in range(len(arm_path)):
            waypoint = list(arm_path[i])
            if len(waypoint) != 8:
                raise ValueError(
                    "Expected 8-DoF waypoint [torso + 7 arm joints] in arm_path"
                )
            base_goal = np.array(base_configs[i], dtype=np.float32)  # [x,y,th]
            whole = np.concatenate(
                [
                    base_goal,
                    np.array([waypoint[0]], dtype=np.float32),
                    np.array(waypoint[1:], dtype=np.float32),
                ]
            ).astype(np.float32)
            merged.append(whole)

        with self._lock:
            # Teleop pattern: single absolute waypoint -> replace queue (latest wins)
            if len(merged) == 1:
                self._merged_traj = [merged[0]]
                self._last_waypoint_idx = 0
            else:
                # Planned path: replace queue and reset index
                self._merged_traj = list(merged)
                self._last_waypoint_idx = 0
        return True

    def start_whole_body_motion(self, arm_path: List, base_configs: List) -> bool:
        return self.execute_whole_body_motion(arm_path, base_configs)

    def stop_whole_body_motion(self):
        with self._lock:
            self._merged_traj.clear()
            self._last_waypoint_idx = 0
            self._body_abs_target[2] = np.nan

    def is_motion_done(self) -> bool:
        with self._lock:
            if len(self._merged_traj) == 0:
                return True
            # Keep it simple: motion is considered done as soon as the controller
            # advances to (i.e., targets) the last waypoint.
            return self._last_waypoint_idx >= (len(self._merged_traj) - 1)

    def get_motion_result(self) -> bool:
        return True

    def cancel_head_goals(self):
        """Cancels any pending head movement commands."""
        with self._lock:
            # Clear any one-shot head commands waiting to be applied
            if "body" in self._overlay_once:
                body_cmd = self._overlay_once["body"]
                # Keep torso command if present, clear pan/tilt only
                self._overlay_once["body"] = np.array(
                    [np.nan, np.nan, body_cmd[2]], dtype=np.float32
                )
            # Clear absolute pan/tilt targets (keep torso target)
            self._body_abs_target[0] = np.nan  # pan
            self._body_abs_target[1] = np.nan  # tilt

    def get_arm_action_state(self) -> Any:
        return 3 if self.is_motion_done() else 1

    def cancel_arm_goals(self):
        with self._lock:
            self._merged_traj.clear()
            self._last_waypoint_idx = 0
            self._overlay_once.pop("arm", None)

    def cancel_torso_goals(self):
        with self._lock:
            self._overlay_once.pop("body", None)
            # Also clear persistent absolute targets for body
            self._body_abs_target[:] = np.nan

    def get_arm_action_result(self) -> Any:
        return True

    def send_joint_values(self, target_joints: List[float], duration: float) -> Any:
        if len(target_joints) != 8:
            raise ValueError("Expected 8 joint values [torso + 7 arm joints]")

        with self._lock:
            # Set absolute torso target; arm commanded via one-shot joint velocities
            self._overlay_once["body"] = np.array(
                [np.nan, np.nan, float(target_joints[0])], dtype=np.float32
            )
            # Compute simple one-shot arm joint velocities toward targets
            current_arm = np.array(self.get_arm_joint_positions(), dtype=np.float32)
            desired_arm = np.array(target_joints[1:], dtype=np.float32)
            gain = 1.0 / max(duration, 1e-3)
            arm_vel = (desired_arm - current_arm) * gain
            limits = self._v_limits["joint_vel_max"][1:]
            arm_vel = np.clip(arm_vel, -limits, limits)
            self._overlay_once["arm"] = arm_vel.astype(np.float32)
            last = self._last_step
        return last

    def execute_joint_trajectory(self, trajectory_points: List, duration: float) -> Any:
        """Execute an arm trajectory by converting it to whole-body waypoints (blocking)."""
        if len(trajectory_points) == 0:
            return True

        self.stop_whole_body_motion()

        # Start the trajectory asynchronously
        self.start_joint_trajectory_async(trajectory_points, duration)

        # Wait for completion (blocking)
        start_time = time.time()
        timeout = duration * 2.0  # Set a reasonable timeout (2x duration)
        while not self.is_motion_done():
            if time.time() - start_time > timeout:
                log.warning(
                    f"Joint trajectory execution timed out after {timeout:.2f}s"
                )
                self.stop_whole_body_motion()
                return False
            time.sleep(0.05)

        return True

    def start_joint_trajectory_async(
        self, trajectory_points: List, duration: float
    ) -> bool:
        """Start an arm trajectory asynchronously (non-blocking)."""
        if len(trajectory_points) == 0:
            return True

        # Get current base pose (stays fixed for arm-only motion)
        x, y, th = self.get_base_pose()

        # Convert arm trajectory to whole-body waypoints [x, y, th, torso, 7 arm joints]
        merged: List[np.ndarray] = []
        for point in trajectory_points:
            if len(point) != 8:
                raise ValueError("Expected 8-DoF waypoint [torso + 7 arm joints]")
            whole = np.concatenate(
                [
                    np.array([x, y, th], dtype=np.float32),
                    np.array([point[0]], dtype=np.float32),  # torso
                    np.array(point[1:], dtype=np.float32),  # 7 arm joints
                ]
            ).astype(np.float32)
            merged.append(whole)

        # Queue the trajectory into _merged_traj
        with self._lock:
            self._merged_traj.extend(merged)
        return True

    def move_base(self, linear_x: float, angular_z: float):
        with self._lock:
            # Manual command overrides MPC base for now
            self._overlay_persistent["base"] = np.array(
                [linear_x, angular_z], dtype=np.float32
            )

    def stop_base(self):
        with self._lock:
            self._overlay_persistent.pop("base", None)

    def navigate_to(self, position: List[float], orientation: List[float]) -> Any:
        # No-op in simulation
        pass

    def set_torso_height(self, height: float, duration: float) -> Any:
        with self._lock:
            # Absolute torso height; pan/tilt unchanged
            self._overlay_once["body"] = np.array(
                [np.nan, np.nan, float(height)], dtype=np.float32
            )
            last = self._last_step
        return last

    def get_joint_positions(self, joint_names: List[str]) -> List[float]:
        all_joint_names, all_joint_pos = self.get_joint_states()
        name_to_pos = dict(zip(all_joint_names, all_joint_pos))
        return [name_to_pos[name] for name in joint_names]

    def get_arm_joint_positions(self) -> List[float]:
        return self.get_joint_positions(self.agent.arm_joint_names)

    def move_arm_to_joint_positions(
        self, target_joints: List[float], duration: float = 2.0
    ) -> Any:
        if len(target_joints) != len(self.agent.arm_joint_names):
            raise ValueError("Incorrect number of target joint positions for the arm.")

        with self._lock:
            # One-shot arm velocity proportional to position error (simple)
            current = np.array(self.get_arm_joint_positions(), dtype=np.float32)
            desired = np.array(target_joints, dtype=np.float32)
            gain = 1.0 / max(duration, 1e-3)
            vel = (desired - current) * gain
            limits = self._v_limits["joint_vel_max"][1:]
            vel = np.clip(vel, -limits, limits)
            self._overlay_once["arm"] = vel.astype(np.float32)
            last = self._last_step
        return last

    def control_gripper(self, position: float, max_effort: float = 50.0) -> Any:
        """
        Control gripper: 0.0 = fully closed, 1.0 = fully open.
        """
        self._gripper_target = float(np.clip(position, 0.0, 1.0))
        return self._last_step

    def get_gripper_status(self) -> Dict[str, Any]:
        gripper_positions = self.get_joint_positions(
            ["r_gripper_finger_joint", "l_gripper_finger_joint"]
        )
        position = sum(gripper_positions) / 2 / 0.05
        return {
            "position": np.clip(position, 0, 1),
            "stalled": False,
            "reached_goal": True,
        }

    def move_head(self, pan: float, tilt: float, duration: float):
        with self._lock:
            # Absolute pan/tilt; keep torso height unchanged
            self._overlay_once["body"] = np.array(
                [float(pan), float(tilt), np.nan], dtype=np.float32
            )
            last = self._last_step
        return last

    # ===== Internal helpers for MPC =====
    def _get_current_whole_body_state(self) -> np.ndarray:
        x, y, th = self.get_base_pose()
        torso = self.get_joint_positions(["torso_lift_joint"])[0]
        arm = self.get_arm_joint_positions()
        state = np.array([x, y, th, torso] + list(arm), dtype=np.float32)
        return state

    def _find_nearest_waypoint_index(self, current_state: np.ndarray) -> int:
        if len(self._merged_traj) == 0:
            return 0
        current_base_pos = current_state[:2]
        current_th = float(current_state[2])
        current_joints = current_state[3:]
        min_cost = float("inf")
        nearest_idx = 0
        # Search across all queued waypoints; include arm/torso error in the distance
        for i in range(0, len(self._merged_traj)):
            wp = self._merged_traj[i]
            pos = wp[:2]
            th = float(wp[2])
            pos_dist = float(np.linalg.norm(pos - current_base_pos))
            dth = abs(current_th - th)
            dth = min(dth, 2 * np.pi - dth)
            joint_err = float(np.linalg.norm(wp[3:] - current_joints))
            # Weighting: prioritize base alignment, then arm
            cost = pos_dist + 0.2 * dth + 0.3 * joint_err
            if cost < min_cost:
                min_cost = cost
                nearest_idx = i
        return nearest_idx

    def _construct_reference(
        self, start_idx: int, current_state: np.ndarray
    ) -> np.ndarray:
        N = int(self._mpc["N"])
        ref = np.zeros((3 + 8, N + 1), dtype=np.float32)

        # Safety check: if trajectory is empty, use current state
        if len(self._merged_traj) == 0:
            for i in range(N + 1):
                ref[:, i] = current_state
            return ref

        start = min(start_idx + 1, len(self._merged_traj) - 1)
        for i in range(N + 1):
            idx = min(start + i, len(self._merged_traj) - 1)
            try:
                ref[:, i] = self._merged_traj[idx]
            except IndexError:
                ref[:, i] = current_state
        # Unwrap heading with respect to current yaw for continuity
        yaw_seq = np.concatenate([[current_state[2]], ref[2, :]])
        yaw_unwrap = np.unwrap(yaw_seq)
        ref[2, :] = yaw_unwrap[1:]
        return ref

    def _solve_one_step_mpc(self, x: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        """
        Simple one-step quadratic control using linearized kinematics around current yaw.

        State x: [x, y, th, torso, 7 arm]
        Control u: [v, w, torso_vel, 7 arm_vel]
        x_next ≈ x + B u * dt, where
            B = [[cos(th), 0, 0, 0...],
                 [sin(th), 0, 0, 0...],
                 [0,       1, 0, 0...],
                 [0,       0, 1, 0...],
                 [0,       0, 0, I_7]]
        Solve: (B^T Q B + R) u = B^T Q (x_ref - x) / dt
        """
        # dt = float(self._mpc["dt"])
        Q = np.diag(self._mpc["Q"])  # (11x11)
        R = np.diag(self._mpc["R"])  # (10x10)

        th = float(x[2])
        cth = float(np.cos(th))
        sth = float(np.sin(th))

        nx = 11
        nu = 10
        B = np.zeros((nx, nu), dtype=np.float32)
        # x, y
        B[0, 0] = cth
        B[1, 0] = sth
        # th
        B[2, 1] = 1.0
        # torso
        B[3, 2] = 1.0
        # arm 7
        for j in range(7):
            B[4 + j, 3 + j] = 1.0

        # Target delta
        dx = (x_ref - x).astype(np.float32)

        # Quadratic solve: (B^T Q B + R) u = (B^T Q dx)
        BtQ = B.T @ Q
        H = (BtQ @ B) + R
        g = BtQ @ dx

        # Solve H u = g
        try:
            u = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            u = np.linalg.lstsq(H, g, rcond=None)[0]

        # Optional aggressiveness gain before clamping
        gain = float(self._mpc.get("gain", 1.0))
        u = (gain * u).astype(np.float32)
        u[0] = float(np.clip(u[0], -self._v_limits["v_max"], self._v_limits["v_max"]))
        u[1] = float(np.clip(u[1], -self._v_limits["w_max"], self._v_limits["w_max"]))
        joint_max = self._v_limits["joint_vel_max"]
        u[2:] = np.clip(u[2:], -joint_max, joint_max)
        return u

    def start_monitoring(
        self,
        target_object_name: str,
        hold_seconds: float = 0.0,
        contact_force_threshold: float = 0.001,
    ):
        # Cache actor + a gripper reference link for hold metric (if enabled).
        # Match existing project behavior: target_object_name is treated as a substring.
        with self._env_lock:
            target_actor = next(
                a
                for a in self.env.scene.get_all_actors()
                if target_object_name in a.name
            )
        gripper_link = next(
            lnk for lnk in self.agent.robot.links if lnk.name == "gripper_link"
        )

        with self._lock:
            self._monitor_enabled = True
            self._target_object_name = target_object_name
            self._collision_detected = False
            self._collision_pairs = set()
            self._success_detected = False
            self._monitor_contact_force_threshold = float(contact_force_threshold)

            # Hold metric config (0.0 disables)
            self._hold_seconds = float(hold_seconds)
            # Do not count immediately from first contact; caller arms after lift.
            self._hold_armed = False
            self._hold_success_detected = False
            self._hold_state = {
                "active": False,
                "t0": 0.0,
            }
            self._hold_target_actor = target_actor
            self._hold_gripper_link = gripper_link

        # Reset dynamic benchmark actors (requires env lock for scene modification)
        with self._env_lock:
            self.benchmark_manager.reset()

    def arm_hold_monitoring(self):
        """
        Arm the hold timer and reset its internal state.
        Intended use: call this right after the grasp stage finishes lifting.
        """
        with self._lock:
            self._hold_armed = True
            self._hold_success_detected = False
            self._hold_state = {"active": False, "t0": 0.0}

    def stop_monitoring(self):
        with self._lock:
            self._monitor_enabled = False
            self._hold_seconds = 0.0

    def get_monitoring_results(self) -> Tuple[bool, bool]:
        with self._lock:
            return self._collision_detected, self._success_detected

    def get_hold_monitoring_results(
        self,
    ) -> Tuple[bool, bool, bool, List[Tuple[str, str]]]:
        with self._lock:
            return (
                self._collision_detected,
                self._success_detected,
                self._hold_success_detected,
                list(self._collision_pairs),
            )

    # ===== Trajectory Recording =====
    def start_trajectory_recording(self) -> None:
        """Start recording the executed trajectory."""
        with self._lock:
            self._trajectory_buffer.clear()
            self._record_trajectory = True

    def stop_trajectory_recording(self) -> List[Dict[str, Any]]:
        """Stop recording and return the recorded trajectory."""
        with self._lock:
            self._record_trajectory = False
            trajectory = list(self._trajectory_buffer)
            return trajectory

    def get_trajectory(self) -> List[Dict[str, Any]]:
        """Get the current trajectory buffer without stopping recording."""
        with self._lock:
            return list(self._trajectory_buffer)

    def clear_trajectory(self) -> None:
        """Clear the trajectory buffer."""
        with self._lock:
            self._trajectory_buffer.clear()
