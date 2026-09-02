"""Build MS-HAB SAC observations from a mani_skill 3.0.1 benchmark env.

Produces exactly the contract documented in sac_policy.py, from our benchmark
environment's raw observations + sim state. Joint order, link names, depth
format, and frame-stacking semantics were pinned against reference dumps from
the mshab-branch environment (adapter/reference_pairs.npz) — the active-joint
list is asserted at construction time.
"""

from __future__ import annotations

from collections import deque

import torch

# Empirical ground truth from dump_reference.py (mshab-branch env)
EXPECTED_ACTIVE_JOINTS = [
    "root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint",
    "torso_lift_joint", "head_pan_joint", "shoulder_pan_joint",
    "head_tilt_joint", "shoulder_lift_joint", "upperarm_roll_joint",
    "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint",
    "wrist_roll_joint", "r_gripper_finger_joint", "l_gripper_finger_joint",
]
BASE_LINK = "base_link"
TCP_LINK = "gripper_link"
FRAME_STACK = 3


def _vectorize_pose(pose) -> torch.Tensor:
    """[p(3), q(4, wxyz)] — matches mani_skill's vectorize_pose."""
    return torch.cat([pose.p, pose.q], dim=-1)


class MshabObsBuilder:
    def __init__(self, env, target_actor):
        self.uenv = env.unwrapped
        self.agent = self.uenv.agent
        self.target = target_actor

        names = [j.name for j in self.agent.robot.active_joints]
        assert names == EXPECTED_ACTIVE_JOINTS, (
            f"Joint order mismatch vs mshab reference:\n{names}"
        )
        self.base_link = self.agent.robot.links_map[BASE_LINK]
        self.tcp = self.agent.robot.links_map[TCP_LINK]

        self._frames: dict[str, deque] = {
            "fetch_head_depth": deque(maxlen=FRAME_STACK),
            "fetch_hand_depth": deque(maxlen=FRAME_STACK),
        }
        self._first = True

    def reset(self, target_actor=None):
        if target_actor is not None:
            self.target = target_actor
        for d in self._frames.values():
            d.clear()
        self._first = True

    def _depth(self, raw_obs, cam: str) -> torch.Tensor:
        # (B, H, W, 1) int16 mm -> (B, 1, H, W) float mm — identical to
        # FetchDepthObservationWrapper (raw values, no normalization).
        d = raw_obs["sensor_data"][cam]["depth"]
        return d.permute(0, 3, 1, 2).float()

    def build(self, raw_obs) -> dict:
        head = self._depth(raw_obs, "fetch_head")
        hand = self._depth(raw_obs, "fetch_hand")
        if self._first:
            # FrameStack.reset fills the deque with copies of the first frame
            for _ in range(FRAME_STACK):
                self._frames["fetch_head_depth"].append(head)
                self._frames["fetch_hand_depth"].append(hand)
            self._first = False
        else:
            self._frames["fetch_head_depth"].append(head)
            self._frames["fetch_hand_depth"].append(hand)

        pixels = {
            k: torch.stack(tuple(f), dim=1)  # (B, 3, 1, 128, 128), oldest..newest
            for k, f in self._frames.items()
        }

        # 3.0.1's "+state" obs mode pre-flattens agent obs; read the robot
        # handles directly instead (same values, same joint order).
        qpos = self.agent.robot.get_qpos().float()
        qvel = self.agent.robot.get_qvel().float()

        base_pose_inv = self.base_link.pose.inv()
        tcp_pose_wrt_base = _vectorize_pose(base_pose_inv * self.tcp.pose)
        obj_pose_wrt_base = _vectorize_pose(base_pose_inv * self.target.pose)
        goal_pos_wrt_base = torch.zeros(
            (qpos.shape[0], 3), device=qpos.device
        )  # Pick subtask goal is None upstream -> zeros
        is_grasped = (
            self.agent.is_grasping(self.target, max_angle=30).float().reshape(-1, 1)
        )

        state = torch.cat(
            [
                qpos[:, 3:],
                qvel[:, 3:],
                tcp_pose_wrt_base,
                obj_pose_wrt_base,
                goal_pos_wrt_base,
                is_grasped,
            ],
            dim=1,
        )
        assert state.shape[-1] == 42, state.shape
        return {"pixels": pixels, "state": state}
