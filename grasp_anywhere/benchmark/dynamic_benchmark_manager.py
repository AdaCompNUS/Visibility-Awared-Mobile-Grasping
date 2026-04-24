import uuid
from typing import Any, Dict, List

import numpy as np
import sapien.core as sapien
from scipy.spatial.transform import Rotation as R

from grasp_anywhere.robot.kinematics import forward_kinematics


class DynamicBenchmarkManager:
    """
    Simplified Manager for Dynamic Benchmark.
    Injects dynamic obstacles (Boxes only) based on robot state.
    """

    def __init__(self, robot_env: Any) -> None:
        self.env = robot_env

        # Public Tunable Parameters (Directly modified by external runners)
        self.enabled = False
        self.trigger_angle_threshold = np.deg2rad(90)
        self.nav_trigger_distance = 1.7

        # Internal State
        self._triggered_nav = False
        self._triggered_manip = False
        self._dynamic_actors: List[Dict[str, Any]] = []
        self._current_obstacle_config = None

    @property
    def scene(self):
        if hasattr(self.env.env.unwrapped, "scene"):
            return self.env.env.unwrapped.scene
        return None

    def reset(self) -> None:
        self._triggered_nav = False
        self._triggered_manip = False
        self._clear_dynamic_actors()
        # self._current_obstacle_config = None # Handled by external setter

    def set_current_obstacle_config(self, config: Dict[str, Any]):
        """Set the dynamic obstacle configuration for the current task."""
        self._current_obstacle_config = config
        self._triggered_nav = False

    def update(
        self, dt: float, base_position: np.ndarray, base_velocity: np.ndarray
    ) -> None:
        if not self.enabled or self.scene is None:
            return

        # 1. Update Moving Actors
        self._update_moving_actors(dt)

        # 2. Check Triggers
        if not self._triggered_nav:
            self._check_nav_trigger(base_position, base_velocity)

    def _to_flat_numpy(self, arr):
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        arr = np.array(arr)
        if arr.ndim > 1:
            arr = arr.flatten()
        return arr

    def _update_moving_actors(self, dt: float):
        keep_actors = []
        for item in self._dynamic_actors:
            actor = item["actor"]
            if item.get("type") == "moving_pedestrian":
                vel = item["velocity"]
                pose = actor.pose

                # Robustly handle both Tensor and Numpy environments
                p = self._to_flat_numpy(pose.p)
                current_p = p.astype(np.float32)

                new_pos = current_p + vel * dt
                # q should also be safe
                q = self._to_flat_numpy(pose.q)
                actor.set_pose(sapien.Pose(new_pos, q))
            keep_actors.append(item)
        self._dynamic_actors = keep_actors

    def _check_nav_trigger(self, base_position: np.ndarray, base_velocity: np.ndarray):
        """
        Input:
            base_position: (x, y, theta)
            base_velocity: (v, w)
        """
        if not self._current_obstacle_config:
            return

        # 2. Check Conditions
        obstacle_start_pos = np.array(self._current_obstacle_config["start_position"])
        robot_pos_2d = base_position[:2]
        obstacle_pos_2d = obstacle_start_pos[:2]

        dist_to_obstacle = np.linalg.norm(obstacle_pos_2d - robot_pos_2d)

        # Condition 1: Moving near the obstacle check
        if dist_to_obstacle > self.nav_trigger_distance:
            return

        # Condition 3: Facing the obstacle
        theta = base_position[2]
        robot_heading = np.array([np.cos(theta), np.sin(theta)])

        to_obstacle = obstacle_pos_2d - robot_pos_2d
        to_obstacle_norm = np.linalg.norm(to_obstacle)
        if to_obstacle_norm < 1e-3:
            return

        to_obstacle_dir = to_obstacle / to_obstacle_norm
        dot_prod = np.dot(robot_heading, to_obstacle_dir)

        if dot_prod <= 0:
            return

        # 3. Trigger!
        print(
            f"[DynamicBenchmark] Triggering Obstacle! Dist={dist_to_obstacle:.2f}, Dot={dot_prod:.2f}"
        )
        self._spawn_moving_pedestrian(self._current_obstacle_config)
        self._triggered_nav = True

    # --------------------------------------------------------------------------
    # Spawners
    # --------------------------------------------------------------------------

    def _spawn_moving_pedestrian(self, obstacle_config: Dict[str, Any]):
        start_pos = np.array(obstacle_config["start_position"])
        start_rot = np.array(obstacle_config["start_orientation"])
        dims = obstacle_config["dimension"]

        print(f"[DynamicBenchmark] Spawning Triggered Obstacle at {start_pos}")

        actor = self._create_box(
            pose=sapien.Pose(p=start_pos, q=start_rot),
            half_size=[d / 2.0 for d in dims],
            is_static=True,
            name="dynamic_pedestrian",
        )

        if actor:
            self._dynamic_actors.append(
                {"actor": actor, "type": "moving_pedestrian", "velocity": np.zeros(3)}
            )

    def spawn_manipulation_obstacles(self, target_pos: np.ndarray):
        """
        Public triggering method for manipulation obstacles.
        Spawns multiple random boxes around the target, avoiding the robot.
        """
        if not self.enabled:
            return

        if self._triggered_manip:
            return

        print(f"[DynamicBenchmark] Triggering Manipulation Obstacles at {target_pos}")

        # Parameters
        num_obstacles = np.random.randint(3, 6)  # 3 to 5 obstacles
        attempts = 0
        max_attempts = 100
        spawned_count = 0

        # Get robot link positions for collision checking
        robot_link_positions = []

        joint_names, joint_values = self.env.get_joint_states()
        joint_dict = dict(zip(joint_names, joint_values))

        fk_joints = [
            "torso_lift_joint",
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "upperarm_roll_joint",
            "elbow_flex_joint",
            "forearm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
            "head_pan_joint",
            "head_tilt_joint",
        ]

        q = np.array([joint_dict[j] for j in fk_joints])
        link_poses = forward_kinematics(q)

        bx, by, bth = self.env.get_base_pose()
        T_base = np.eye(4)
        T_base[:3, :3] = R.from_euler("z", bth).as_matrix()
        T_base[:3, 3] = [bx, by, 0.0]

        for name, T_local in link_poses.items():
            T_world = T_base @ T_local
            robot_link_positions.append(T_world[:3, 3])

            # Extra points for gripper
            if name == "gripper_link":
                for x_off in [0.05, 0.1, 0.15, 0.2]:
                    p = T_world @ np.array([x_off, 0, 0, 1])
                    robot_link_positions.append(p[:3])

        while spawned_count < num_obstacles and attempts < max_attempts:
            attempts += 1

            # Randomize Size (W, D, H)
            # Dims: 0.04 - 0.1m xy, 0.1 - 0.3m height
            dims = np.random.uniform([0.04, 0.04, 0.1], [0.10, 0.10, 0.3])

            # Randomize Position (Cylinder around target)
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(0.25, 0.70)

            offset = np.array([np.cos(angle) * dist, np.sin(angle) * dist, 0.0])
            spawn_pos = target_pos + offset
            spawn_pos[2] = target_pos[2] + dims[2] / 2.0  # Sit on table

            # 1. Check Collision with Target (Simple distance check)
            if np.linalg.norm(offset) < 0.12:
                continue

            # 2. Check Collision with Robot
            safe_distance = 0.3  # buffer from any link center
            collision = False
            for link_p in robot_link_positions:
                if np.linalg.norm(spawn_pos - link_p) < safe_distance:
                    collision = True
                    break

            if collision:
                continue

            # 3. Check Collision with other spawned obstacles
            for item in self._dynamic_actors:
                if item.get("type", "") == "static_blocker":
                    other_actor = item["actor"]
                    if np.linalg.norm(other_actor.pose.p - spawn_pos) < 0.1:
                        collision = True
                        break

            if collision:
                continue

            # Spawn
            print(
                f"[DynamicBenchmark] Spawning Obstacle {spawned_count+1}/{num_obstacles} at {spawn_pos}"
            )
            actor = self._create_box(
                pose=sapien.Pose(p=spawn_pos),
                half_size=[d / 2 for d in dims],
                is_static=True,
                name="dynamic_blocker",
            )

            if actor:
                self._dynamic_actors.append(
                    {
                        "actor": actor,
                        "type": "static_blocker",
                    }
                )
                spawned_count += 1

        self._triggered_manip = True

    # --------------------------------------------------------------------------
    # Low Level
    # --------------------------------------------------------------------------

    def _create_box(self, pose, half_size, is_static, name):
        if not self.scene:
            return None

        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=half_size)
        builder.add_box_visual(half_size=half_size)

        # Use UUID to ensure global uniqueness across resets/scenes
        unique_name = f"{name}_{uuid.uuid4().hex}"
        if is_static:
            actor = builder.build_static(name=unique_name)
        else:
            actor = builder.build_kinematic(name=unique_name)

        actor.set_pose(pose)
        return actor

    def _clear_dynamic_actors(self):
        if not self.scene:
            self._dynamic_actors = []
            return
        for item in self._dynamic_actors:
            try:
                item["actor"].remove_from_scene()
            except RuntimeError:
                # Actor may already be removed (e.g., scene was reset)
                pass
        self._dynamic_actors = []
