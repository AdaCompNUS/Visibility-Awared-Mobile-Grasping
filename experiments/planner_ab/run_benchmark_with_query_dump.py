"""Run experiments/run_maniskill_benchmark.py, dumping every planner query.

With GA_PLAN_QUERY_DIR set, each whole-body and arm planning query (start,
goal, obstacle clouds at query time, and the logged result) is saved there as
an .npz for offline replay through several planners. Patches are applied at
module top level so the spawn-started benchmark workers get them too.
"""

import itertools
import json
import multiprocessing
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments"))

import numpy as np  # noqa: E402

DUMP_DIR = os.environ.get("GA_PLAN_QUERY_DIR")
if DUMP_DIR:
    from grasp_anywhere.robot import fetch as fetch_mod

    os.makedirs(DUMP_DIR, exist_ok=True)
    _counter = itertools.count()
    _orig_wb = fetch_mod.Fetch.plan_whole_body_motion
    _orig_arm = fetch_mod.Fetch._plan_arm
    _orig_add = fetch_mod.Fetch.add_pointcloud
    _orig_clear = fetch_mod.Fetch.clear_pointclouds

    def _record_backend(robot):
        """Wrap robot.planner_backend.add_pointcloud to keep what it was given."""
        backend = robot.planner_backend
        if getattr(backend, "_dump_wrapped", False):
            return
        backend._dump_wrapped = True
        robot._dump_clouds = []
        add = backend.add_pointcloud

        def add_pointcloud(points, point_radius):
            add(points, point_radius)
            if len(points) > 0:
                robot._dump_clouds.append((np.asarray(points, np.float32), float(point_radius)))

        backend.add_pointcloud = add_pointcloud

    def add_pointcloud_patch(self, *a, **kw):
        _record_backend(self)
        return _orig_add(self, *a, **kw)

    def clear_pointclouds_patch(self):
        _record_backend(self)
        self._dump_clouds = []
        return _orig_clear(self)

    def _save(robot, kind, start, goal, clouds, spheres, n_before, extra):
        record = robot.planning_log[-1] if len(robot.planning_log) > n_before else {}
        arrays = {f"cloud_{i}": np.asarray(c[0], np.float32) for i, c in enumerate(clouds)}
        name = f"{os.getpid()}_{next(_counter):05d}_{kind}.npz"
        np.savez_compressed(
            os.path.join(DUMP_DIR, name),
            start=np.asarray(start, np.float64),
            goal=np.asarray(goal, np.float64),
            cloud_radii=np.array([c[1] for c in clouds], np.float64),
            spheres=np.array([list(c) + [r] for c, r in spheres], np.float64).reshape(-1, 4),
            meta=json.dumps(
                {
                    "kind": kind,
                    "record": record,
                    "motion_planner": robot.motion_planner,
                    "attached": robot._current_attachment is not None,
                    **extra,
                }
            ),
            **arrays,
        )

    def plan_whole_body_motion(self, start_joints, goal_joints, start_base, goal_base, *a, **kw):
        clouds, spheres, n = list(getattr(self, '_dump_clouds', [])), [], len(self.planning_log)
        res = _orig_wb(self, start_joints, goal_joints, start_base, goal_base, *a, **kw)
        r3 = lambda v: [round(float(x), 3) for x in v]  # as plan_whole_body_motion rounds
        _save(
            self,
            "whole_body",
            r3(start_base) + r3(start_joints),
            r3(goal_base) + r3(goal_joints),
            clouds,
            spheres,
            n,
            {"n_waypoints": len(res["arm_path"]) if res and res["success"] else 0},
        )
        return res

    def _plan_arm(self, current_joints, target_joints):
        clouds, spheres, n = list(getattr(self, '_dump_clouds', [])), [], len(self.planning_log)
        base = list(self.get_base_params())
        res = _orig_arm(self, current_joints, target_joints)
        _save(
            self,
            "arm",
            list(current_joints),
            list(target_joints),
            clouds,
            spheres,
            n,
            {"base": [float(b) for b in base], "n_waypoints": len(res) if res else 0},
        )
        return res

    fetch_mod.Fetch.plan_whole_body_motion = plan_whole_body_motion
    fetch_mod.Fetch.add_pointcloud = add_pointcloud_patch
    fetch_mod.Fetch.clear_pointclouds = clear_pointclouds_patch
    fetch_mod.Fetch._plan_arm = _plan_arm

import run_maniskill_benchmark  # noqa: E402

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    run_maniskill_benchmark.run_benchmark()
