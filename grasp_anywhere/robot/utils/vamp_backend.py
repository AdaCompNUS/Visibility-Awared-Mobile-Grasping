"""Motion planning and collision checking through the vendored VAMP fork.

``Fetch`` talks to this class and to ``FetchPlanningBackend`` through the
same methods; this one keeps the original VAMP pipeline (multilayer
RRT-Connect + Hybrid A* for whole-body motions, RRT-Connect for the arm).
"""

import numpy as np
import vamp

import grasp_anywhere.robot.utils.replanning_utils as replanning_utils
from grasp_anywhere.robot.utils.whole_body_planners import (
    plan_base_only,
    plan_fcit_wb_whole_body,
    plan_rrtc_whole_body,
)
from grasp_anywhere.utils.logger import log


class VampBackend:
    def __init__(self, planner="rrtc"):
        """
        Initialize VAMP motion planner with 8-DOF configuration and collision settings.

        Args:
            planner: VAMP arm planner ("rrtc", "fcit", "prm").
        """
        self.env = vamp.Environment()

        # Configure robot and planner with custom settings
        (
            self.vamp_module,
            self.planner_func,
            self.plan_settings,
            self.simp_settings,
        ) = vamp.configure_robot_and_planner_with_kwargs(
            "fetch",  # Robot name
            planner,  # Planner algorithm (Rapidly-exploring Random Tree Connect)
            sampler_name="halton",  # Use Halton sampler for better coverage
        )

        # Initialize the sampler
        self.sampler = self.vamp_module.halton()
        self.sampler.skip(0)  # Skip initial samples if needed

        # Bounds over XY for FCIT*, updated when adding pointclouds
        self.pc_bounds_xy = None  # tuple (x_min, x_max, y_min, y_max)
        self._bounds_padding = 0.1

        log.info("VAMP planner initialized with collision avoidance settings")

    # ── Collision environment ────────────────────────────────────────

    def set_base_params(self, theta, x, y):
        """Base pose used by VAMP's arm-only planning and validation."""
        self.vamp_module.set_base_params(theta, x, y)

    def add_pointcloud(self, points, point_radius):
        """Add an (N, 3) list of points as obstacles."""
        # Define robot-specific radius parameters
        r_min, r_max = vamp.ROBOT_RADII_RANGES[
            "fetch"
        ]  # Min/max sphere radius for Fetch robot
        self.env.add_pointcloud(points, r_min, r_max, point_radius)

        # Update FCIT* XY bounds from the point cloud for whole-body planning.
        pts_np = np.array(points, dtype=np.float64)
        if pts_np.size > 0 and pts_np.shape[1] == 3:
            min_xy = pts_np[:, :2].min(axis=0)
            max_xy = pts_np[:, :2].max(axis=0)
            x_min = float(min_xy[0] - self._bounds_padding)
            x_max = float(max_xy[0] + self._bounds_padding)
            y_min = float(min_xy[1] - self._bounds_padding)
            y_max = float(max_xy[1] + self._bounds_padding)
            self.pc_bounds_xy = (x_min, x_max, y_min, y_max)

    def clear_pointclouds(self):
        self.env.clear_pointclouds()

    def add_sphere(self, position, radius, name=None):
        sphere = vamp.Sphere(position, radius)
        if name:
            sphere.name = name
        self.env.add_sphere(sphere)

    def clear_spheres(self):
        self.env.clear_spheres()

    def attach(self, spheres_params, offset_position, offset_orientation_xyzw):
        """Attach spheres to the end effector; returns the attachment."""
        attachment = vamp.Attachment(offset_position, offset_orientation_xyzw)
        attachment.add_spheres(
            [vamp.Sphere(list(s["position"]), s["radius"]) for s in spheres_params]
        )
        self.env.attach(attachment)
        return attachment

    def detach(self):
        self.env.detach()

    # ── Queries ──────────────────────────────────────────────────────

    def validate(self, arm_config, base_config):
        """True if the whole-body configuration is collision-free."""
        return self.vamp_module.validate_whole_body_config(
            arm_config, base_config, self.env
        )

    def path_in_collision(self, arm_path, base_configs, current_waypoint_index):
        """True if the path after ``current_waypoint_index`` collides."""
        return replanning_utils.check_trajectory_for_collisions(
            self.vamp_module,
            self.env,
            arm_path,
            base_configs,
            current_waypoint_index,
        )

    def filter_robot(self, points, arm_config, base_config, point_radius):
        """Drop the points within ``point_radius`` of the robot's spheres."""
        return self.vamp_module.filter_fetch_from_pointcloud(
            points, arm_config, base_config, self.env, point_radius
        )

    def eefk(self, arm_config):
        """End-effector (position, quaternion xyzw) in the base frame."""
        return self.vamp_module.eefk(arm_config)

    # ── Planning ─────────────────────────────────────────────────────

    def plan_arm(self, base, current_joints, target_joints):
        """Plan the 8-DOF torso+arm; returns (waypoints or None, stats).

        VAMP plans at the base last passed to ``set_base_params``; ``base``
        is unused.
        """
        result = self.planner_func(
            current_joints,
            target_joints,
            self.env,
            self.plan_settings,
            self.sampler,
        )

        if result.solved:
            log.info("Path planning succeeded!")

            # Get planning statistics
            simple = self.vamp_module.simplify(
                result.path, self.env, self.simp_settings, self.sampler
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

            return trajectory_points, {
                "total_planning_time_ms": result.nanoseconds / 1e6,
                "simplification_time_ms": simple.nanoseconds / 1e6,
            }
        else:
            return None, {"total_planning_time_ms": result.nanoseconds / 1e6}

    def plan_whole_body(
        self,
        start_joints,
        goal_joints,
        start_base,
        goal_base,
        planner="rrtc",
        fcit_settings_overrides=None,
    ):
        """Plan a whole-body motion with multilayer RRTC or FCIT*."""
        if planner == "fcit_wb":
            if self.pc_bounds_xy is None:
                log.warning(
                    "FCIT* selected but XY bounds are not available. Consider calling add_pointcloud first."
                )
            res = plan_fcit_wb_whole_body(
                start_joints,
                goal_joints,
                start_base,
                goal_base,
                self.env,
                self.vamp_module,
                self.pc_bounds_xy,
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
            self.env,
            self.vamp_module,
            self.plan_settings,
            self.simp_settings,
            self.sampler,
            interpolate_density=0.03,
        )

    def plan_base(self, start_base, goal_base, arm, settings_overrides=None):
        """Plan a base-only motion with Hybrid A*, the arm held at ``arm``."""
        return plan_base_only(
            start_base,
            goal_base,
            self.env,
            self.vamp_module,
            self.simp_settings,
            self.sampler,
            arm,
            settings_overrides=settings_overrides,
        )
