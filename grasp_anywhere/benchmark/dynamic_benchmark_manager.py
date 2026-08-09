import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import sapien.core as sapien
from scipy.ndimage import label
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from grasp_anywhere.robot.kinematics import forward_kinematics


class DynamicBenchmarkManager:
    """
    Manager for the navigation and manipulation dynamic benchmark challenges.

    Navigation obstacles are kinematic actors which follow short arcs on ManiSkill's
    Fetch-specific navigation mesh. Every arc is checked against the active
    robot-to-goal connectivity before it is accepted, so an obstacle can invalidate
    the current plan without sealing the only route to the goal.
    """

    def __init__(self, robot_env: Any) -> None:
        self.env = robot_env

        # Public Tunable Parameters (Directly modified by external runners)
        self.enabled = False
        self.trigger_angle_threshold = np.deg2rad(90)
        self.nav_trigger_distance = 1.7
        self.obstacle_speed = 0.35
        # Circumscribed radius of ManiSkill's Fetch base collision mesh (0.288 m),
        # rounded to the navigation-mesh resolution.
        self.robot_footprint_radius = 0.30

        # Internal State
        self._triggered_nav = False
        self._triggered_manip = False
        self._dynamic_actors: List[Dict[str, Any]] = []
        self._current_obstacle_config = None
        self._navigation_goal: Optional[np.ndarray] = None
        self._spawn_retry_remaining = 0.0

        # Lazily initialized from ReplicaCADSceneBuilder.navigable_positions.
        self._navmesh_initialized = False
        self._nav_vertices: Optional[np.ndarray] = None
        self._nav_vertex_grid: Optional[np.ndarray] = None
        self._nav_mask: Optional[np.ndarray] = None
        self._nav_origin: Optional[np.ndarray] = None
        self._nav_resolution = 0.0
        self._nav_tree: Optional[cKDTree] = None

    @property
    def scene(self):
        if hasattr(self.env.env.unwrapped, "scene"):
            return self.env.env.unwrapped.scene
        return None

    def reset(self) -> None:
        self._triggered_nav = False
        self._triggered_manip = False
        self._navigation_goal = None
        self._spawn_retry_remaining = 0.0
        self._clear_dynamic_actors()
        # The underlying reset may select a different ReplicaCAD scene.
        self._navmesh_initialized = False
        self._nav_vertices = None
        self._nav_vertex_grid = None
        self._nav_mask = None
        self._nav_origin = None
        self._nav_resolution = 0.0
        self._nav_tree = None
        # _current_obstacle_config is set by the benchmark runner after reset.

    def set_current_obstacle_config(self, config: Dict[str, Any]):
        """Set the dynamic obstacle configuration for the current task."""
        self._current_obstacle_config = config
        self._triggered_nav = False

    def set_navigation_goal(self, goal_base_config) -> None:
        """Set the active base goal used by the connectivity invariant."""
        goal = np.asarray(goal_base_config, dtype=np.float32).reshape(-1)
        if goal.size < 2:
            raise ValueError("Navigation goal must contain at least x and y")
        self._navigation_goal = goal[:2].copy()

    def update(
        self, dt: float, base_position: np.ndarray, base_velocity: np.ndarray
    ) -> None:
        if not self.enabled or self.scene is None:
            return

        self._spawn_retry_remaining = max(0.0, self._spawn_retry_remaining - dt)

        # The normal movement tick only performs a few vector operations.
        self._update_moving_actors(dt, base_position)

        if not self._triggered_nav and self._spawn_retry_remaining == 0.0:
            self._check_nav_trigger(base_position, base_velocity)

    def _to_flat_numpy(self, arr):
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        arr = np.array(arr)
        if arr.ndim > 1:
            arr = arr.flatten()
        return arr

    def _update_moving_actors(self, dt: float, base_position: np.ndarray) -> None:
        keep_actors = []
        for item in self._dynamic_actors:
            actor = item["actor"]
            if item.get("type") == "moving_pedestrian":
                robot_xy = np.asarray(base_position[:2], dtype=np.float32)
                pose = actor.pose
                current_p = self._to_flat_numpy(pose.p).astype(np.float32)
                motion_path = item.get("motion_path")
                path_index = int(item.get("path_index", 0))
                item["path_retry_remaining"] = max(
                    0.0, float(item.get("path_retry_remaining", 0.0)) - dt
                )

                if (motion_path is None or path_index >= len(motion_path)) and item[
                    "path_retry_remaining"
                ] == 0.0:
                    motion_path = self._select_motion_path(item, robot_xy)
                    path_index = 0
                    item["motion_path"] = motion_path
                    if motion_path is None:
                        item["path_retry_remaining"] = 0.5

                travel = float(item["speed"]) * max(float(dt), 0.0)
                new_pos = current_p.copy()
                while motion_path is not None and path_index < len(motion_path):
                    target_xy = motion_path[path_index]
                    delta = target_xy - new_pos[:2]
                    distance = float(np.linalg.norm(delta))
                    if distance <= travel + 1e-8:
                        new_pos[:2] = target_xy
                        travel -= distance
                        path_index += 1
                    else:
                        new_pos[:2] += delta * (travel / max(distance, 1e-8))
                        break

                item["path_index"] = path_index
                velocity_xy = new_pos[:2] - current_p[:2]
                if np.linalg.norm(velocity_xy) > 1e-8:
                    yaw = float(np.arctan2(velocity_xy[1], velocity_xy[0]))
                    q = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
                else:
                    q = self._to_flat_numpy(pose.q)
                actor.set_pose(sapien.Pose(new_pos, q))
            keep_actors.append(item)
        self._dynamic_actors = keep_actors

    def _check_nav_trigger(
        self, base_position: np.ndarray, base_velocity: np.ndarray
    ) -> None:
        """Spawn the navigation obstacle when the robot enters its event region."""
        del base_velocity
        if not self._current_obstacle_config or self._navigation_goal is None:
            return

        obstacle_start = np.asarray(
            self._current_obstacle_config["start_position"], dtype=np.float32
        )
        robot_xy = np.asarray(base_position[:2], dtype=np.float32)
        distance = float(np.linalg.norm(obstacle_start[:2] - robot_xy))
        if distance > self.nav_trigger_distance:
            return

        if self._spawn_moving_pedestrian(
            self._current_obstacle_config,
            np.asarray(base_position, dtype=np.float32),
        ):
            print(
                f"[DynamicBenchmark] Triggered moving obstacle. "
                f"EventDist={distance:.2f}"
            )
            self._triggered_nav = True
        else:
            # Retry after the robot advances out of a possible choke point.
            self._spawn_retry_remaining = 0.5

    # --------------------------------------------------------------------------
    # Navigation obstacle
    # --------------------------------------------------------------------------

    def _prepare_navmesh(self) -> bool:
        """Rasterize ManiSkill's navigable-position mesh once per scene."""
        if self._navmesh_initialized:
            return self._nav_mask is not None
        self._navmesh_initialized = True

        scene_builder = getattr(self.env.env.unwrapped, "scene_builder", None)
        meshes = getattr(scene_builder, "navigable_positions", None)
        if not meshes or meshes[0] is None:
            print("[DynamicBenchmark] No Fetch navigation mesh; obstacle skipped.")
            return False

        vertices = np.asarray(meshes[0].vertices, dtype=np.float32)
        if vertices.ndim != 2 or len(vertices) == 0 or vertices.shape[1] < 2:
            print("[DynamicBenchmark] Invalid Fetch navigation mesh; obstacle skipped.")
            return False
        vertices = vertices[:, :2]

        spacings = []
        for axis in range(2):
            unique = np.unique(np.round(vertices[:, axis], decimals=5))
            deltas = np.diff(unique)
            spacings.extend(deltas[deltas > 1e-4].tolist())
        if not spacings:
            return False

        resolution = float(np.min(spacings))
        origin = np.min(vertices, axis=0)
        vertex_grid = np.rint((vertices - origin) / resolution).astype(np.int32)
        width, height = np.max(vertex_grid, axis=0) + 1
        nav_mask = np.zeros((height, width), dtype=bool)
        nav_mask[vertex_grid[:, 1], vertex_grid[:, 0]] = True

        self._nav_vertices = vertices
        self._nav_vertex_grid = vertex_grid
        self._nav_mask = nav_mask
        self._nav_origin = origin
        self._nav_resolution = resolution
        self._nav_tree = cKDTree(vertices)
        return True

    def _connectivity_safe(
        self,
        obstacle_xy: np.ndarray,
        robot_xy: np.ndarray,
        clearance: float,
    ) -> bool:
        """Check robot/goal connectivity with an inflated navmesh obstacle."""
        if not self._prepare_navmesh() or self._navigation_goal is None:
            return False

        obstacle_xy = np.asarray(obstacle_xy, dtype=np.float32)
        robot_xy = np.asarray(robot_xy, dtype=np.float32)
        goal_xy = self._navigation_goal
        obstacle_distance, _ = self._nav_tree.query(obstacle_xy)
        _, robot_vertex = self._nav_tree.query(robot_xy)
        _, goal_vertex = self._nav_tree.query(goal_xy)
        mesh_tolerance = 1.5 * self._nav_resolution
        if obstacle_distance > mesh_tolerance:
            return False
        if (
            np.linalg.norm(obstacle_xy - robot_xy) <= clearance
            or np.linalg.norm(obstacle_xy - goal_xy) <= clearance
        ):
            return False

        free = self._nav_mask.copy()
        lower = np.floor(
            (obstacle_xy - clearance - self._nav_origin) / self._nav_resolution
        ).astype(int)
        upper = np.ceil(
            (obstacle_xy + clearance - self._nav_origin) / self._nav_resolution
        ).astype(int)
        gx0, gy0 = np.maximum(lower, 0)
        gx1 = min(int(upper[0]), free.shape[1] - 1)
        gy1 = min(int(upper[1]), free.shape[0] - 1)
        yy, xx = np.ogrid[gy0 : gy1 + 1, gx0 : gx1 + 1]
        world_x = self._nav_origin[0] + xx * self._nav_resolution
        world_y = self._nav_origin[1] + yy * self._nav_resolution
        outside_obstacle = (world_x - obstacle_xy[0]) ** 2 + (
            world_y - obstacle_xy[1]
        ) ** 2 > clearance**2
        free[gy0 : gy1 + 1, gx0 : gx1 + 1] &= outside_obstacle

        robot_grid = self._nav_vertex_grid[int(robot_vertex)]
        goal_grid = self._nav_vertex_grid[int(goal_vertex)]
        if (
            not free[robot_grid[1], robot_grid[0]]
            or not free[goal_grid[1], goal_grid[0]]
        ):
            return False
        components, _ = label(
            free, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        )
        robot_component = components[robot_grid[1], robot_grid[0]]
        return (
            robot_component != 0
            and robot_component == components[goal_grid[1], goal_grid[0]]
        )

    def _select_motion_path(
        self, item: Dict[str, Any], robot_xy: np.ndarray
    ) -> Optional[np.ndarray]:
        """Select a connectivity-safe neighboring navmesh cell."""
        if not self._prepare_navmesh() or self._navigation_goal is None:
            return None

        current_xy = self._to_flat_numpy(item["actor"].pose.p)[:2].astype(np.float32)
        relative = current_xy - robot_xy
        radius = float(np.linalg.norm(relative))
        if radius < 1e-6:
            return None

        _, vertex_index = self._nav_tree.query(current_xy)
        current_grid = self._nav_vertex_grid[int(vertex_index)]
        gx, gy = int(current_grid[0]), int(current_grid[1])
        previous_grid = item.get("previous_grid")
        initial_direction = float(item["orbit_direction"])

        for direction in (initial_direction, -initial_direction):
            tangent = (
                direction
                * np.array([-relative[1], relative[0]], dtype=np.float32)
                / radius
            )
            candidates = []
            for candidate_grid in (
                (gx + 1, gy),
                (gx - 1, gy),
                (gx, gy + 1),
                (gx, gy - 1),
            ):
                candidate_gx, candidate_gy = candidate_grid
                if (
                    candidate_gx < 0
                    or candidate_gy < 0
                    or candidate_gx >= self._nav_mask.shape[1]
                    or candidate_gy >= self._nav_mask.shape[0]
                    or not self._nav_mask[candidate_gy, candidate_gx]
                ):
                    continue
                target_xy = self._nav_origin + self._nav_resolution * np.array(
                    [candidate_gx, candidate_gy], dtype=np.float32
                )
                step = target_xy - current_xy
                alignment = float(np.dot(step, tangent))
                radial_drift = abs(float(np.linalg.norm(target_xy - robot_xy)) - radius)
                is_backtrack = previous_grid is not None and candidate_grid == tuple(
                    previous_grid
                )
                candidates.append(
                    (is_backtrack, -alignment, radial_drift, candidate_grid, target_xy)
                )

            candidates.sort(key=lambda candidate: candidate[:3])
            for _, _, _, candidate_grid, target_xy in candidates:
                midpoint = 0.5 * (current_xy + target_xy)
                if self._connectivity_safe(
                    midpoint, robot_xy, item["clearance"]
                ) and self._connectivity_safe(target_xy, robot_xy, item["clearance"]):
                    item["previous_grid"] = (gx, gy)
                    item["orbit_direction"] = direction
                    return np.asarray([target_xy], dtype=np.float32)
        return None

    def _spawn_moving_pedestrian(
        self, obstacle_config: Dict[str, Any], base_position: np.ndarray
    ) -> bool:
        if not self._prepare_navmesh() or self._navigation_goal is None:
            return False

        configured_start = np.asarray(
            obstacle_config["start_position"], dtype=np.float32
        )
        start_rot = np.asarray(obstacle_config["start_orientation"], dtype=np.float32)
        dims = np.asarray(obstacle_config["dimension"], dtype=np.float32)
        robot_xy = np.asarray(base_position[:2], dtype=np.float32)
        clearance = float(
            self.robot_footprint_radius + 0.5 * np.hypot(dims[0], dims[1])
        )

        # Prefer positions intersecting the active path, then those closest to the
        # configured trigger. This provokes replanning while preserving a detour.
        trajectory = np.asarray(getattr(self.env, "_merged_traj", []), dtype=np.float32)
        if trajectory.ndim != 2 or trajectory.shape[1] < 2:
            return False
        start_index = int(getattr(self.env, "_last_waypoint_idx", 0))
        route_xy = trajectory[start_index:, :2]
        if len(route_xy) == 0:
            return False
        if len(route_xy) > 128:
            route_xy = route_xy[:: max(1, len(route_xy) // 128)]

        _, near_start = self._nav_tree.query(
            configured_start[:2], k=min(128, len(self._nav_vertices))
        )
        _, on_route = self._nav_tree.query(route_xy)
        candidate_indices = np.unique(
            np.concatenate((np.atleast_1d(near_start), np.atleast_1d(on_route))).astype(
                np.int32
            )
        )
        candidates = self._nav_vertices[candidate_indices]
        robot_distances = np.linalg.norm(candidates - robot_xy, axis=1)
        path_distances = np.min(
            np.linalg.norm(candidates[:, None, :] - route_xy[None, :, :], axis=2),
            axis=1,
        )
        configured_distances = np.linalg.norm(candidates - configured_start[:2], axis=1)
        valid = (
            (robot_distances > clearance)
            & (robot_distances < self.nav_trigger_distance + 0.25)
            & (path_distances < clearance)
        )
        scores = 3.0 * path_distances + configured_distances
        candidate_order = np.argsort(np.where(valid, scores, np.inf))

        start_xy = None
        for candidate_index in candidate_order[:64]:
            if not valid[candidate_index]:
                break
            candidate = candidates[candidate_index]
            if self._connectivity_safe(candidate, robot_xy, clearance):
                start_xy = candidate
                break
        if start_xy is None:
            return False

        start_pos = configured_start.copy()
        start_pos[:2] = start_xy
        actor = self._create_box(
            pose=sapien.Pose(p=start_pos, q=start_rot),
            half_size=(dims / 2.0).tolist(),
            is_static=False,
            name="dynamic_pedestrian",
        )
        if actor is None:
            return False

        speed = float(obstacle_config.get("speed", self.obstacle_speed))
        if speed <= 0.0:
            speed = self.obstacle_speed
        self._dynamic_actors.append(
            {
                "actor": actor,
                "type": "moving_pedestrian",
                "speed": speed,
                "clearance": clearance,
                "orbit_direction": 1.0,
                "previous_grid": None,
                "motion_path": None,
                "path_index": 0,
                "path_retry_remaining": 0.0,
            }
        )
        print(f"[DynamicBenchmark] Spawned moving obstacle at {start_pos}")
        return True

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
        visual_material = (0.9, 0.1, 0.1, 1.0) if name == "dynamic_pedestrian" else None
        builder.add_box_visual(half_size=half_size, material=visual_material)
        builder.initial_pose = pose

        # Use UUID to ensure global uniqueness across resets/scenes
        unique_name = f"{name}_{uuid.uuid4().hex}"
        if is_static:
            actor = builder.build_static(name=unique_name)
        else:
            actor = builder.build_kinematic(name=unique_name)

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
