#!/usr/bin/env python3
"""Smoke test for the pixi environment.

Validates that the three compiled C/C++ modules import, that the ROS-free TRAC-IK
solver actually solves, that the ROS python bindings import (the robot code needs
them at module load), and — unless SMOKE_MANISKILL=0 — that a ManiSkill Fetch scene
builds on the GPU.  Run with:  pixi run smoke
"""
import os
import sys
import traceback

import numpy as np


def check(name, fn):
    try:
        fn()
        print(f"  [ok]   {name}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()
        return False


ok = True

print("== compiled C/C++ modules ==")
ok &= check("import pytracik", lambda: __import__("pytracik"))
ok &= check("import vamp", lambda: __import__("vamp"))
ok &= check("import ikfast_fetch", lambda: __import__("ikfast_fetch"))

print("== TRAC-IK solve (fetch arm: base_link -> gripper_link) ==")


def ik_solve():
    import pytracik
    from scipy.spatial.transform import Rotation as R

    from grasp_anywhere.robot.ik.trac_ik_solver import TracIKSolver

    urdf = "resources/fetch_ext/fetch ext.urdf"
    s = TracIKSolver(base_link="base_link", ee_link="gripper_link", urdf_path=urdf)
    n = s.num_joints
    assert n == 8, f"expected 8-DOF fetch arm chain, got {n}"
    T = np.asarray(pytracik.fk(s._solver, np.array([0.15] * n))).reshape(4, 4)
    sol = s.get_ik_near_seed(
        [0.0] * n, T[:3, 3].tolist(), R.from_matrix(T[:3, :3]).as_quat().tolist()
    )
    assert sol is not None, "IK returned no solution for a known-reachable pose"
    err = float(
        np.linalg.norm(
            np.asarray(pytracik.fk(s._solver, np.array(sol))).reshape(4, 4)[:3, 3]
            - T[:3, 3]
        )
    )
    assert err < 1e-3, f"IK round-trip position error too large: {err}"
    print(f"         DOF={n}, round-trip pos error={err:.2e} m")


ok &= check("trac_ik_solver round-trip", ik_solve)

print("== ROS python bindings (robot code imports rospy at module load) ==")
ok &= check("import rospy", lambda: __import__("rospy"))

if os.environ.get("SMOKE_MANISKILL", "1") == "1":
    print("== ManiSkill Fetch scene on the GPU ==")

    def maniskill():
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401

        env = gym.make(
            "ReplicaCAD_SceneManipulation-v1", robot_uids="fetch", render_mode="rgb_array"
        )
        env.reset(seed=0)
        env.close()

    ok &= check("gym.make ReplicaCAD + fetch", maniskill)
else:
    print("== ManiSkill scene skipped (SMOKE_MANISKILL=0) ==")

print("\nSMOKE_OK" if ok else "\nSMOKE_FAIL")
sys.exit(0 if ok else 1)
