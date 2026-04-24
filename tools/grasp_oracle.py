from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sapien
import trimesh
from mani_skill.agents.controllers.pd_joint_pos import PDJointPosControllerConfig

_ORACLE_ART_COUNTER = 0


@dataclass
class GraspOracleCfg:
    # Gripper geometry (very rough Fetch-like parallel jaw gripper)
    finger_length: float = 0.10
    finger_height: float = 0.02
    finger_thickness: float = 0.01
    palm_thickness: float = 0.02

    # Opening (meters, distance between inner finger faces)
    open_width: float = 0.10
    closed_width: float = 0.002

    # Controller (PD joint drives)
    # These defaults are intentionally stronger so the oracle behaves like a firm parallel jaw gripper.
    finger_drive_stiffness: float = 3000.0
    finger_drive_damping: float = 1500.0
    finger_drive_force_limit: float = 15000.0

    # Simulation schedule
    settle_steps: int = 10
    pregrasp_offset: float = 0.10
    # Shift the grasp pose forward along the grasp's +Z (approach) axis (meters).
    grasp_offset_forward: float = 0.03
    approach_steps: int = 50
    close_steps: int = 30
    retreat_steps: int = 30
    # Lift straight up in world-Z after retreating (set to 0 to disable).
    lift_distance: float = 0.10
    lift_steps: int = 30
    stable_steps: int = 30

    # Success criteria
    max_stable_object_movement: float = 0.03
    # If lift_distance > 0, require the object to lift at least this much (meters).
    min_lift_height: float = 0.03

    # --- Contact material (surface friction) ---
    # This controls *contact* friction between the oracle gripper collision shapes and other objects.
    # (Not to be confused with joint friction in set_joint_properties.)
    contact_static_friction: float = 2.0
    contact_dynamic_friction: float = 2.0
    contact_restitution: float = 0.0

    # --- Frame / axis alignment ---
    #
    # The oracle gripper model is defined in its local frame as:
    # - +Z: approach direction (fingers extend along +Z)
    # - +/-Y: opening/closing direction (fingers move along +/-Y)
    #
    # In other parts of this repo (e.g. visualization of grasp poses), grasps are often
    # represented with +Z as approach but +X as the opening direction.
    # That mismatch manifests as "approach vector is correct, but the gripper is rotated/twisted".
    #
    # We fix it by post-multiplying the incoming grasp pose by a constant transform:
    #   T_oracle = T_in @ T_grasp_to_oracle
    # Default: rotate -90deg about +Z so grasp +X (opening) maps to oracle +Y (opening).
    #
    # Stored row-major as 16 floats to keep this dataclass JSON/YAML friendly.
    grasp_to_oracle_T: tuple[float, ...] = (
        0.0,
        1.0,
        0.0,
        0.0,
        -1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    # Which axis of the (aligned) grasp pose is the approach direction (0=x,1=y,2=z).
    # Default 2 matches GraspPlanner.run() usage (approach is +Z).
    approach_axis: int = 2


def _body_actor_name(body: object) -> str:
    name = getattr(body, "name", "") or ""
    if not name and hasattr(body, "entity"):
        name = getattr(body.entity, "name", "") or ""
    return name


class _OracleGripperArticulation:
    """
    Oracle gripper loaded via ManiSkill3 URDF interface:
    - env.unwrapped.scene.create_urdf_loader().load(...)
    - URDF is defined in tools/oracle_box_gripper.urdf

    Joint naming is fixed by the URDF:
    - oracle_finger_left_joint
    - oracle_finger_right_joint
    """

    def __init__(self, scene: object, cfg: GraspOracleCfg):
        self.scene = scene
        self.cfg = cfg
        global _ORACLE_ART_COUNTER
        _ORACLE_ART_COUNTER += 1
        self.name = f"oracle_gripper_{_ORACLE_ART_COUNTER}"
        self.art = self._load_from_urdf(name=self.name)
        # Controller over floating base (6) + fingers (2). Motion is via joint targets (actions).
        self.joint_names = [
            "oracle_x_joint",
            "oracle_y_joint",
            "oracle_z_joint",
            "oracle_roll_joint",
            "oracle_pitch_joint",
            "oracle_yaw_joint",
            "oracle_finger_left_joint",
            "oracle_finger_right_joint",
        ]
        lower = [-20.0, -20.0, -20.0, -np.pi, -np.pi, -np.pi, 0.0, 0.0]
        upper = [20.0, 20.0, 20.0, np.pi, np.pi, np.pi, 0.05, 0.05]
        stiffness = [2.0e4] * 6 + [float(cfg.finger_drive_stiffness)] * 2
        damping = [3.0e3] * 6 + [float(cfg.finger_drive_damping)] * 2
        force_limit = [5.0e4] * 6 + [float(cfg.finger_drive_force_limit)] * 2

        self.controller_cfg = PDJointPosControllerConfig(
            joint_names=self.joint_names,
            lower=lower,
            upper=upper,
            stiffness=stiffness,
            damping=damping,
            force_limit=force_limit,
            friction=0.0,
            use_delta=False,
            use_target=False,
            interpolate=False,
            normalize_action=False,
            drive_mode="force",
        )
        self.controller = self.controller_cfg.controller_cls(
            self.controller_cfg,
            self.art,
            control_freq=20,
            scene=self.scene,
        )
        self.controller.set_drive_property()
        self.controller.reset()

        # Cached targets (so set_opening can re-issue a full action vector)
        # Start near the scene so it is visible in the viewer; z=1.0 avoids immediate floor penetration.
        self._base_target_xyzrpy = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)
        self._opening_q = 0.05
        self._set_action_from_targets()

        # Keep oracle gripper from "falling away" during the oracle check.
        # for link in self.art.get_links():
        #     link.set_disable_gravity(True)
        for j in self.art.active_joints:
            j.set_drive_velocity_target(0.0)

    def _load_from_urdf(self, *, name: str):
        urdf_path = Path(__file__).resolve().parent / "oracle_box_gripper.urdf"
        loader = self.scene.create_urdf_loader()
        # Configure high-friction contact materials via the URDF loader interface (no sapien_utils).
        loader.set_link_material(
            "oracle_finger_left_link",
            float(self.cfg.contact_static_friction),
            float(self.cfg.contact_dynamic_friction),
            float(self.cfg.contact_restitution),
        )
        loader.set_link_material(
            "oracle_finger_right_link",
            float(self.cfg.contact_static_friction),
            float(self.cfg.contact_dynamic_friction),
            float(self.cfg.contact_restitution),
        )
        # Friction patch tuning (same knobs exposed by ManiSkill's urdf_config docs).
        loader.set_link_patch_radius("oracle_finger_left_link", 0.1)
        loader.set_link_min_patch_radius("oracle_finger_left_link", 0.1)
        loader.set_link_patch_radius("oracle_finger_right_link", 0.1)
        loader.set_link_min_patch_radius("oracle_finger_right_link", 0.1)
        art = loader.load(str(urdf_path), name=str(name))
        return art

    def remove(self) -> None:
        # ManiSkill registers articulations by name; remove from the scene (CPU sim only).
        self.scene.remove_articulation(self.art)

    def set_root_pose(self, T_world: np.ndarray) -> None:
        T = np.asarray(T_world, dtype=np.float32).reshape(4, 4)
        p = T[:3, 3].astype(np.float32)
        # URDF base orientation is a serial roll->pitch->yaw joint chain (intrinsic XYZ).
        # Use rotating-axes Euler extraction to match that convention.
        rpy = trimesh.transformations.euler_from_matrix(T, axes="rxyz")
        self._base_target_xyzrpy = (
            float(p[0]),
            float(p[1]),
            float(p[2]),
            float(rpy[0]),
            float(rpy[1]),
            float(rpy[2]),
        )
        self._set_action_from_targets()

    def _set_action_from_targets(self) -> None:
        import torch

        x, y, z, roll, pitch, yaw = self._base_target_xyzrpy
        q = float(self._opening_q)
        action = torch.tensor(
            [[x, y, z, roll, pitch, yaw, q, q]],
            dtype=torch.float32,
            device=self.art.device,
        )
        self.controller.set_action(action)

    def set_opening(self, opening_width: float) -> None:
        # opening_width is between inner finger faces; URDF models opening ~= 2*q
        w = float(opening_width)
        self._opening_q = float(max(0.0, 0.5 * w))
        self._set_action_from_targets()

    def teleport(self, T_world: np.ndarray, opening_width: float) -> None:
        # Teleport by directly setting the articulation qpos (no swept motion / pushing objects).
        T = np.asarray(T_world, dtype=np.float32).reshape(4, 4)
        p = T[:3, 3].astype(np.float32)
        # URDF base orientation is a serial roll->pitch->yaw joint chain (intrinsic XYZ).
        # Use rotating-axes Euler extraction to match that convention.
        rpy = trimesh.transformations.euler_from_matrix(T, axes="rxyz")

        # Keep controller targets consistent with the teleported state so the next PD step doesn't
        # "yank" the gripper.
        self._base_target_xyzrpy = (
            float(p[0]),
            float(p[1]),
            float(p[2]),
            float(rpy[0]),
            float(rpy[1]),
            float(rpy[2]),
        )
        w = float(opening_width)
        self._opening_q = float(max(0.0, 0.5 * w))
        self._set_action_from_targets()

        name_to_val = {
            "oracle_x_joint": float(p[0]),
            "oracle_y_joint": float(p[1]),
            "oracle_z_joint": float(p[2]),
            "oracle_roll_joint": float(rpy[0]),
            "oracle_pitch_joint": float(rpy[1]),
            "oracle_yaw_joint": float(rpy[2]),
            "oracle_finger_left_joint": float(self._opening_q),
            "oracle_finger_right_joint": float(self._opening_q),
        }

        qpos_raw = np.asarray(self.art.get_qpos(), dtype=np.float32)
        batched = qpos_raw.ndim == 2
        qpos = (qpos_raw[0].copy() if batched else qpos_raw.copy()).reshape(-1)

        for i, j in enumerate(self.art.active_joints):
            jn = getattr(j, "name", "") or ""
            if jn in name_to_val and i < qpos.shape[0]:
                qpos[i] = float(name_to_val[jn])

        qpos_set = qpos[None, :] if batched else qpos
        self.art.set_qpos(qpos_set)
        if hasattr(self.art, "set_qvel"):
            self.art.set_qvel(np.zeros_like(qpos_set))


def _step_env(env: object, n: int, *, controller: object) -> None:
    scene = env.unwrapped.scene
    sim_freq, control_freq = getattr(env.unwrapped, "sim_freq", 500), getattr(
        env.unwrapped, "control_freq", 20
    )
    n_substeps = max(1, sim_freq // control_freq)
    for _ in range(int(n)):
        for _ in range(n_substeps):
            controller.before_simulation_step()
            scene.step()
        if hasattr(scene, "update_render"):
            scene.update_render()
        if getattr(env, "render_mode", None) is not None:
            env.render()


def _has_oracle_env_collision(scene: object, target_name: str) -> bool:
    t = (target_name or "").lower()
    for c in scene.get_contacts():
        a0, a1 = (
            _body_actor_name(c.bodies[0]).lower(),
            _body_actor_name(c.bodies[1]).lower(),
        )
        if "oracle_" in a0 and "oracle_" in a1:
            continue
        if ("oracle_" in a0 and t not in a1) or ("oracle_" in a1 and t not in a0):
            return True
    return False


def _reset_actor_state(entity: object, pose: sapien.Pose) -> None:
    entity.set_pose(pose)


def _snapshot_scene(
    scene: object, *, exclude_name_substr: str
) -> list[tuple[object, sapien.Pose]]:
    snap: list[tuple[object, sapien.Pose]] = []
    for a in scene.get_all_actors():
        name = (getattr(a, "name", "") or "").lower()
        if exclude_name_substr in name:
            continue
        snap.append((a, a.pose))
    return snap


def _restore_scene_pose(snap: list[tuple[object, sapien.Pose]]) -> None:
    for a, p in snap:
        a.set_pose(p)


def oracle_is_graspable(
    *,
    env: object,
    actor: object,
    grasp_poses_world: np.ndarray,
    cfg: GraspOracleCfg = GraspOracleCfg(),
) -> bool:
    """
    Returns True iff ANY grasp pose can close on the object and lift it.
    """
    grasps = np.asarray(grasp_poses_world, dtype=np.float32).reshape(-1, 4, 4)
    if grasps.shape[0] == 0:
        print("[oracle] FAIL no grasps")
        return False

    scene = env.unwrapped.scene

    target_name = getattr(actor, "name", "") or ""
    if not target_name and hasattr(actor, "_objs") and len(actor._objs) > 0:
        target_name = getattr(actor._objs[0], "name", "") or ""

    target_entity = (
        actor._objs[0] if hasattr(actor, "_objs") and len(actor._objs) > 0 else actor
    )
    target_pose0 = target_entity.pose
    if not target_name:
        target_name = _body_actor_name(target_entity)

    # Snapshot BEFORE the oracle exists/moves/steps, so restores always go back to a clean baseline.
    scene_pose0 = _snapshot_scene(scene, exclude_name_substr="oracle")
    gripper = _OracleGripperArticulation(scene, cfg)

    # Make the oracle gripper visible immediately: place it above the target once before trying grasps.
    # T_vis = np.eye(4, dtype=np.float32)
    # p_t = np.asarray(target_entity.pose.p, dtype=np.float32).reshape(-1, 3)[0]
    # T_vis[:3, 3] = p_t + np.array([0.0, 0.0, 0.25], dtype=np.float32)
    # gripper.set_root_pose(T_vis)
    # gripper.set_opening(float(cfg.open_width))
    # _step_env(env, 10, controller=gripper.controller)

    for T in grasps:
        _restore_scene_pose(scene_pose0)

        # Define pregrasp pose by offsetting from grasp pose along -approach.
        # We first align the incoming grasp pose into the oracle gripper's convention.
        T_grasp_in = np.asarray(T, dtype=np.float32).reshape(4, 4)
        T_align = np.asarray(cfg.grasp_to_oracle_T, dtype=np.float32).reshape(4, 4)
        T_grasp = (T_grasp_in @ T_align).astype(np.float32)

        ax = int(getattr(cfg, "approach_axis", 2))
        if ax not in (0, 1, 2):
            ax = 2
        approach_dir = T_grasp[:3, ax].astype(np.float32)
        approach_dir = approach_dir / (np.linalg.norm(approach_dir) + 1e-9)

        T_grasp = np.array(T_grasp, dtype=np.float32)
        T_grasp[:3, 3] = T_grasp[:3, 3] + approach_dir * float(cfg.grasp_offset_forward)

        T_pre = np.array(T_grasp, dtype=np.float32)
        T_pre[:3, 3] = T_grasp[:3, 3] - approach_dir * float(cfg.pregrasp_offset)

        gripper.teleport(T_pre, float(cfg.open_width))
        # Catch "fly away" impulses: collision can occur in the FIRST sim step after teleport
        # and resolve quickly; so we check right after the first step.
        _step_env(env, 1, controller=gripper.controller)
        if _has_oracle_env_collision(scene, target_name):
            print("[oracle] FAIL pregrasp-collision")
            _restore_scene_pose(scene_pose0)
            continue
        if int(cfg.settle_steps) > 1:
            _step_env(env, int(cfg.settle_steps) - 1, controller=gripper.controller)

        # Approach from pregrasp -> grasp along +approach
        for s in range(int(cfg.approach_steps)):
            a = float(s + 1) / float(max(1, int(cfg.approach_steps)))
            T_a = np.array(T_pre, dtype=np.float32)
            T_a[:3, 3] = (1.0 - a) * T_pre[:3, 3] + a * T_grasp[:3, 3]
            gripper.set_root_pose(T_a)
            gripper.set_opening(float(cfg.open_width))
            _step_env(env, 1, controller=gripper.controller)
            # if _has_oracle_env_collision(scene, target_name):
            #     break
        else:
            # Only proceed if approach finished without collisions
            pass
        # if _has_oracle_env_collision(scene, target_name):
        #     print(f"[oracle] FAIL approach-collision")
        #     if scene_pose0:
        #         _restore_scene_pose(scene_pose0)
        #     _reset_actor_state(target_entity, target_pose0)
        #     continue

        left_touch = False
        right_touch = False
        for s in range(int(cfg.close_steps)):
            a = float(s + 1) / float(max(1, int(cfg.close_steps)))
            w = float(cfg.open_width) * (1.0 - a) + float(cfg.closed_width) * a
            gripper.set_root_pose(T_grasp)
            gripper.set_opening(w)
            _step_env(env, 1, controller=gripper.controller)
            # if _has_oracle_env_collision(scene, target_name):
            #     print("Collision detected during closing")
            #     break

            for c in scene.get_contacts():
                a0 = _body_actor_name(c.bodies[0])
                a1 = _body_actor_name(c.bodies[1])
                pair = (a0.lower(), a1.lower())
                if "oracle_finger_left" in pair[0] and target_name.lower() in pair[1]:
                    left_touch = True
                if "oracle_finger_left" in pair[1] and target_name.lower() in pair[0]:
                    left_touch = True
                if "oracle_finger_right" in pair[0] and target_name.lower() in pair[1]:
                    right_touch = True
                if "oracle_finger_right" in pair[1] and target_name.lower() in pair[0]:
                    right_touch = True

        # if _has_oracle_env_collision(scene, target_name):
        #     print("[oracle] FAIL close-collision touch")
        #     if scene_pose0:
        #         _restore_scene_pose(scene_pose0)
        #     _reset_actor_state(target_entity, target_pose0)
        #     continue

        # Retreat back to pregrasp along -approach while staying closed
        for s in range(int(cfg.retreat_steps)):
            a = float(s + 1) / float(max(1, int(cfg.retreat_steps)))
            T_r = np.array(T_grasp, dtype=np.float32)
            T_r[:3, 3] = (1.0 - a) * T_grasp[:3, 3] + a * T_pre[:3, 3]
            gripper.set_root_pose(T_r)
            gripper.set_opening(float(cfg.closed_width))
            _step_env(env, 1, controller=gripper.controller)
            # if _has_oracle_env_collision(scene, target_name):
            #     print("Collision detected during retreat")
            #     break
        # if _has_oracle_env_collision(scene, target_name):
        #     print("[oracle] FAIL retreat-collision")
        #     if scene_pose0:
        #         _restore_scene_pose(scene_pose0)
        #     _reset_actor_state(target_entity, target_pose0)
        #     continue

        # Optional lift: move gripper up in world-Z while staying closed.
        # This better matches the intended "close on the object and lift it" oracle.
        p_before_lift = (
            np.asarray(target_entity.pose.p, dtype=np.float32).reshape(-1, 3)[0].copy()
        )
        if (
            float(getattr(cfg, "lift_distance", 0.0)) > 0.0
            and int(getattr(cfg, "lift_steps", 0)) > 0
        ):
            T_l0 = np.array(T_pre, dtype=np.float32)
            T_l1 = np.array(T_pre, dtype=np.float32)
            T_l1[:3, 3] = T_l1[:3, 3] + np.array(
                [0.0, 0.0, float(cfg.lift_distance)], dtype=np.float32
            )
            for s in range(int(cfg.lift_steps)):
                a = float(s + 1) / float(max(1, int(cfg.lift_steps)))
                T_l = np.array(T_l0, dtype=np.float32)
                T_l[:3, 3] = (1.0 - a) * T_l0[:3, 3] + a * T_l1[:3, 3]
                gripper.set_root_pose(T_l)
                gripper.set_opening(float(cfg.closed_width))
                _step_env(env, 1, controller=gripper.controller)
                # if _has_oracle_env_collision(scene, target_name):
                #     print("[oracle] FAIL lift-collision")
                #     break
            # if _has_oracle_env_collision(scene, target_name):
            #     if scene_pose0:
            #         _restore_scene_pose(scene_pose0)
            #     _reset_actor_state(target_entity, target_pose0)
            #     continue

        # Stability check at pregrasp location: object should remain "held" for a while.
        p0 = np.asarray(target_entity.pose.p, dtype=np.float32).reshape(-1, 3)[0].copy()
        _step_env(env, int(cfg.stable_steps), controller=gripper.controller)
        # if _has_oracle_env_collision(scene, target_name):
        #     print("[oracle] FAIL stable-collision")
        #     if scene_pose0:
        #         _restore_scene_pose(scene_pose0)
        #     _reset_actor_state(target_entity, target_pose0)
        #     continue
        p1 = np.asarray(target_entity.pose.p, dtype=np.float32).reshape(-1, 3)[0].copy()

        moved = float(np.linalg.norm(p1 - p0))
        stable = moved <= float(cfg.max_stable_object_movement)

        lifted_ok = True
        if float(getattr(cfg, "lift_distance", 0.0)) > 0.0:
            dz = float(p0[2] - p_before_lift[2])
            lifted_ok = dz >= float(getattr(cfg, "min_lift_height", 0.0))

        if stable and lifted_ok and left_touch and right_touch:
            if float(getattr(cfg, "lift_distance", 0.0)) > 0.0:
                print(
                    f"[oracle] SUCCESS moved={moved:.4f} dz={float(p0[2] - p_before_lift[2]):.4f}",
                    flush=True,
                )
            else:
                print(f"[oracle] SUCCESS moved={moved:.4f}", flush=True)
            if scene_pose0:
                _restore_scene_pose(scene_pose0)
            _reset_actor_state(target_entity, target_pose0)
            gripper.remove()
            return True
        if float(getattr(cfg, "lift_distance", 0.0)) > 0.0:
            print("[oracle] FAIL unstable/lift/contact")
        else:
            print(
                f"[oracle] FAIL unstable/contact moved={moved:.4f} touch(L/R)={left_touch}/{right_touch}"
            )

        if scene_pose0:
            _restore_scene_pose(scene_pose0)
        _reset_actor_state(target_entity, target_pose0)

    gripper.remove()
    return False
