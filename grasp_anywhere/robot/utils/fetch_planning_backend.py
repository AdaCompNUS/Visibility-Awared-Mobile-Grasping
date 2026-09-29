"""Motion planning and collision checking through ``fetch_planning``.

Same methods as ``VampBackend``, so ``Fetch`` can switch between them with
``planning.motion_planner``; nothing here touches the vendored VAMP fork.
Motions are planned with FLASK, the kinodynamic planner of
``fetch_planning``: it returns a time-parameterised trajectory with an
exactly nonholonomic base, resampled here into the waypoint density the
VAMP planners hand to the executor (a path follower: nearest waypoint +
look-ahead).

``fetch_planning`` has no Python API for attached objects, so the grasped
object is not part of the collision model.
"""

import dataclasses
import zlib

import numpy as np
from fetch_planning.kinematics import create_ik_solver
from fetch_planning.planning import create_planner
from fetch_planning.types import KinodynamicConfig, PlannerConfig
from scipy.spatial.transform import Rotation as R

from grasp_anywhere.utils.logger import log

# Waypoint spacing of the VAMP planners: whole-body paths are interpolated at
# 0.03 of |dxy| + 0.3 |dtheta| + 0.2 |darm| and arm paths at 16 states per
# unit of joint-space distance.
WHOLE_BODY_SPACING = 0.03
ARM_SPACING = 1.0 / 16.0
# Trajectory sampling step used for the resampling (0.6 m/s -> 3 mm).
SAMPLE_DT = 0.005
# Base sampling bounds (m). FLASK samples base positions uniformly in its
# bounds, so a box much larger than the scene sends most samples into free
# space away from the query. The first attempt samples around start and
# goal; a failed attempt is retried over the whole obstacle cloud.
LOCAL_MARGIN = 2.0
SCENE_MARGIN = 0.5
# Largest step between the states checked along a path edge by the
# replanning monitor: joint-space distance (VAMP's resolution of 32 per
# unit), base translation (m) and base rotation (rad).
CHECK_ARM_STEP = 1.0 / 32.0
CHECK_XY_STEP = 0.02
CHECK_THETA_STEP = 0.03


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _whole_body_steps(q):
    """Per-step lengths of an (N, 11) path under VAMP's whole-body metric."""
    d = np.diff(q, axis=0)
    return (
        np.linalg.norm(d[:, :2], axis=1)
        + 0.3 * np.abs(_wrap(d[:, 2]))
        + 0.2 * np.linalg.norm(d[:, 3:], axis=1)
    )


def _resample(q, steps, spacing):
    """Pick rows of ``q`` about ``spacing`` apart in cumulative ``steps``.

    Rows are picked, not interpolated, so every waypoint lies exactly on the
    planned trajectory. The first and last rows are always kept.
    """
    s = np.concatenate([[0.0], np.cumsum(steps)])
    if s[-1] <= spacing:
        return q[[0, -1]]
    targets = np.arange(spacing, s[-1], spacing)
    idx = np.unique(np.searchsorted(s, targets))
    idx = idx[(idx > 0) & (idx < len(q) - 1)]
    return q[np.concatenate([[0], idx, [len(q) - 1]])]


def _densify(q):
    """States along the straight edges of an (N, 11) path, at check resolution."""
    d = np.diff(q, axis=0)
    d[:, 2] = _wrap(d[:, 2])
    n = np.ceil(
        np.maximum.reduce(
            [
                np.linalg.norm(d[:, 3:], axis=1) / CHECK_ARM_STEP,
                np.linalg.norm(d[:, :2], axis=1) / CHECK_XY_STEP,
                np.abs(d[:, 2]) / CHECK_THETA_STEP,
                np.ones(len(d)),
            ]
        )
    ).astype(int)
    states = [q[:1]]
    for q0, dq, k in zip(q[:-1], d, n):
        t = np.arange(1, k + 1)[:, None] / k
        states.append(q0 + t * dq)
    return np.vstack(states)


class FetchPlanningBackend:
    def __init__(self, settings=None):
        """
        Args:
            settings: Optional dict of ``KinodynamicConfig`` field overrides
                (``planning.fetch_planning`` in the YAML config).
        """
        settings = dict(settings or {})
        # The base must not reverse: the head camera faces forward, so a
        # reversing base drives into space the robot has not observed.
        settings.setdefault("allow_reverse", False)
        self._kino_config = KinodynamicConfig(**settings)

        # Whole-body planner that holds the obstacles. It stays on the
        # whole-body subgroup between queries so validation takes 11-DOF
        # configurations.
        self._planner = create_planner("fetch_whole_body", config=PlannerConfig())
        self._lower = np.array(self._planner._planner.lower_bounds())
        self._upper = np.array(self._planner._planner.upper_bounds())
        # Obstacle-free planner for robot self-filtering: fetch_planning's
        # filter also drops points that touch registered obstacles, which
        # would erase mapped geometry from every new observation.
        self._filter = create_planner("fetch_whole_body", config=PlannerConfig())
        self._fk = create_ik_solver("arm_with_torso", backend="ikfast")
        self._clouds = []  # [(points, point_radius)] in the planner
        self._cloud_xy = None  # (xy min, xy max) over the clouds
        self._num_queries = 0

    # ── Collision environment ────────────────────────────────────────

    def set_base_params(self, theta, x, y):
        """No-op: every fetch_planning query carries its base pose."""

    def add_pointcloud(self, points, point_radius):
        """Add an (N, 3) list of points as obstacles."""
        if len(points) > 0:
            self._planner._add_pointcloud_impl(points, point_radius)
            self._clouds.append((points, point_radius))
            xy = np.asarray(points, dtype=np.float64)[:, :2]
            lo, hi = xy.min(axis=0), xy.max(axis=0)
            if self._cloud_xy is not None:
                lo = np.minimum(lo, self._cloud_xy[0])
                hi = np.maximum(hi, self._cloud_xy[1])
            self._cloud_xy = (lo, hi)

    def clear_pointclouds(self):
        while self._planner.remove_pointcloud():
            pass
        self._clouds = []
        self._cloud_xy = None

    def add_sphere(self, position, radius, name=None):
        self._planner._planner.add_sphere(list(position), float(radius))

    def clear_spheres(self):
        # fetch_planning only clears spheres together with the point clouds.
        self._planner.clear_environment()
        for points, point_radius in self._clouds:
            self._planner._add_pointcloud_impl(points, point_radius)

    def attach(self, spheres_params, offset_position, offset_orientation_xyzw):
        log.warning(
            "fetch_planning has no attached-object API; the grasped object is "
            "not collision-checked."
        )
        return None

    def detach(self):
        pass

    # ── Queries ──────────────────────────────────────────────────────

    def validate(self, arm_config, base_config):
        """True if the whole-body configuration is collision-free."""
        return self._planner.validate(np.r_[base_config, arm_config])

    def path_in_collision(self, arm_path, base_configs, current_waypoint_index):
        """True if the path after ``current_waypoint_index`` collides.

        Checks the straight edges between the remaining waypoints, as VAMP's
        ``check_whole_body_collisions`` does.
        """
        remaining = np.hstack(
            [
                np.asarray(base_configs, dtype=np.float64),
                np.asarray(arm_path, dtype=np.float64),
            ]
        )[current_waypoint_index + 1 :]
        if len(remaining) == 0:
            return False
        return not self._planner.validate_batch(_densify(remaining)).all()

    def filter_robot(self, points, arm_config, base_config, point_radius):
        """Drop the points within ``point_radius`` of the robot's spheres."""
        if len(points) == 0:
            return points
        config = np.r_[base_config, arm_config]
        return self._filter.filter_self_from_pointcloud(
            points, point_radius, config
        ).tolist()

    def eefk(self, arm_config):
        """End-effector (position, quaternion xyzw) in the base frame."""
        pose = self._fk.fk(np.asarray(arm_config, dtype=np.float64))
        return pose.position.tolist(), R.from_matrix(pose.rotation).as_quat().tolist()

    # ── Planning ─────────────────────────────────────────────────────

    def _clamp(self, q, indices):
        """Clamp joints into the planner's bounds and wrap the heading.

        Measured joint values and the 3-decimal rounding in
        ``Fetch.plan_whole_body_motion`` can sit just outside a limit, and
        FLASK rejects every edge that leaves the bounds.
        """
        q = np.clip(q, self._lower[indices], self._upper[indices])
        if 2 in indices:
            q[indices.index(2)] = _wrap(q[indices.index(2)])
        return q

    def _base_bounds(self, start_xy, goal_xy):
        """Base sampling boxes (x_lo, x_hi, y_lo, y_hi) for a query, in the
        order to try them: around start and goal, then the whole scene."""
        lo = np.minimum(start_xy, goal_xy) - SCENE_MARGIN
        hi = np.maximum(start_xy, goal_xy) + SCENE_MARGIN
        if self._cloud_xy is None:
            scene_lo, scene_hi = self._lower[:2], self._upper[:2]
        else:
            scene_lo = np.minimum(self._cloud_xy[0] - SCENE_MARGIN, lo)
            scene_hi = np.maximum(self._cloud_xy[1] + SCENE_MARGIN, hi)
        local_lo = np.maximum(lo - LOCAL_MARGIN, scene_lo)
        local_hi = np.minimum(hi + LOCAL_MARGIN, scene_hi)
        boxes = [(local_lo, local_hi)]
        if not (np.allclose(local_lo, scene_lo) and np.allclose(local_hi, scene_hi)):
            boxes.append((scene_lo, scene_hi))
        return [(b_lo[0], b_hi[0], b_lo[1], b_hi[1]) for b_lo, b_hi in boxes]

    def _plan(self, subgroup, indices, start, goal, body=None):
        """Run FLASK on ``subgroup`` (joints ``indices`` of the 11-DOF body).

        ``body`` is the 11-DOF configuration that pins the other joints.
        Returns (result, stats).
        """
        start = self._clamp(np.asarray(start, dtype=np.float64), indices)
        goal = self._clamp(np.asarray(goal, dtype=np.float64), indices)
        # Deterministic per query without touching the global RNG, whose
        # stream the benchmark seeds per task.
        self._num_queries += 1
        seed = zlib.crc32(np.r_[start, goal, self._num_queries].tobytes()) or 1
        config = dataclasses.replace(self._kino_config, seed=seed)

        bounds = [None]
        if 0 in indices:
            bounds = self._base_bounds(start[:2], goal[:2])
        planning_ns = simplify_ns = 0
        for box in bounds:
            if box is not None:
                self._planner.set_base_bounds(*box)
            if subgroup == "fetch_whole_body":
                result = self._planner.plan_kinodynamic(start, goal, config=config)
            else:
                self._planner.set_subgroup(subgroup, base_config=body)
                try:
                    result = self._planner.plan_kinodynamic(start, goal, config=config)
                finally:
                    self._planner.set_subgroup("fetch_whole_body")
            planning_ns += result.planning_time_ns
            simplify_ns += result.simplify_time_ns
            # Only a search failure is worth retrying with wider bounds.
            if result.status.value != "failed":
                break

        stats = {
            "status": result.status.value,
            "total_planning_time_ms": planning_ns / 1e6,
            "simplification_time_ms": simplify_ns / 1e6,
            "planning_iterations": result.iterations,
            "planning_graph_size": result.start_tree_size + result.goal_tree_size,
            "bounds_attempts": bounds.index(box) + 1,
        }
        if result.success:
            stats["trajectory_duration_s"] = result.trajectory.duration
        return result, stats

    def plan_arm(self, base, current_joints, target_joints):
        """Plan the 8-DOF torso+arm with the base parked at ``base``.

        Returns (waypoints or None, stats).
        """
        body = np.r_[base, current_joints]
        result, stats = self._plan(
            "fetch_arm_with_torso",
            list(range(3, 11)),
            current_joints,
            target_joints,
            body,
        )
        if not result.success:
            return None, stats

        _, q, _, _ = result.trajectory.sample_uniform(SAMPLE_DT)
        q = _resample(q, np.linalg.norm(np.diff(q, axis=0), axis=1), ARM_SPACING)
        return q.tolist(), stats

    def plan_whole_body(
        self,
        start_joints,
        goal_joints,
        start_base,
        goal_base,
        planner=None,
        fcit_settings_overrides=None,
    ):
        """Same contract as ``VampBackend.plan_whole_body``.

        ``planner`` and ``fcit_settings_overrides`` select VAMP planners and
        are ignored.
        """
        result, stats = self._plan(
            "fetch_whole_body",
            list(range(11)),
            np.r_[start_base, start_joints],
            np.r_[goal_base, goal_joints],
        )
        if not result.success:
            return {
                "success": False,
                "stats": stats,
                "arm_path": None,
                "base_configs": None,
            }

        _, q, _, _ = result.trajectory.sample_uniform(SAMPLE_DT)
        q = _resample(q, _whole_body_steps(q), WHOLE_BODY_SPACING)
        return {
            "success": True,
            "stats": stats,
            "arm_path": q[:, 3:].tolist(),
            "base_configs": q[:, :3].tolist(),
        }

    def plan_base(self, start_base, goal_base, arm, settings_overrides=None):
        """Same contract as ``VampBackend.plan_base``.

        ``settings_overrides`` configures VAMP's Hybrid A* and is ignored.
        """
        arm = np.asarray(arm, dtype=np.float64)
        result, stats = self._plan(
            "fetch_base",
            [0, 1, 2],
            start_base,
            goal_base,
            np.r_[start_base, arm],
        )
        if not result.success:
            return {"success": False, "stats": stats, "base_configs": None}

        _, q, _, _ = result.trajectory.sample_uniform(SAMPLE_DT)
        full = np.hstack([q, np.tile(arm, (len(q), 1))])
        q = _resample(q, _whole_body_steps(full), WHOLE_BODY_SPACING)
        stats["base_path_length"] = len(q)
        return {"success": True, "stats": stats, "base_configs": q.tolist()}
