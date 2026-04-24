"""
Functional core for ManiSkill contact monitoring (NO behavior changes).

Moved here from `grasp_anywhere/envs/maniskill/monitor_core.py` so the logic
can be reused from other envs/benchmarks without living under `envs/`.
"""

import numpy as np


def _body_actor_name(body: object) -> str:
    """Match existing name resolution logic in `maniskill_env_mpc.py`."""
    name = getattr(body, "name", "") or ""
    if not name and hasattr(body, "entity"):
        name = getattr(body.entity, "name", "") or ""
    return name


def eval_monitor_contacts(
    contacts: object,
    target_object_name: str,
    dt: float,
    force_threshold: float = 0.01,
    robot_name_substr: str = "fetch",
    base_link_exclude: str = "base_link",
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Evaluate contacts and return events for:
    - success (gripper link contacting target object)
    - collision (non-gripper robot link contacting environment)

    IMPORTANT: This preserves the original behavior, including:
    - force computed from impulse / dt
    - threshold check (<= force_threshold ignored)
    - robot/env actor assignment uses bodies[0] as "robot" and bodies[1] as "env"
      (same as original code)

    Returns:
        (success_events, collision_events) where each event is (robot_actor_name, env_actor_name)
    """
    # Use a small epsilon for dt to avoid division by zero if timestep is weird
    dt_safe = float(dt)
    if dt_safe < 1e-6:
        dt_safe = 1.0 / 30.0

    success_events: list[tuple[str, str]] = []
    collision_events: list[tuple[str, str]] = []

    for contact in contacts:
        body0 = contact.bodies[0]
        body1 = contact.bodies[1]

        actor0_name = _body_actor_name(body0)
        actor1_name = _body_actor_name(body1)

        actor0_is_robot = (
            robot_name_substr in actor0_name and base_link_exclude not in actor0_name
        )
        actor1_is_robot = (
            robot_name_substr in actor1_name and base_link_exclude not in actor1_name
        )

        if actor0_is_robot != actor1_is_robot:
            # Check force to confirm contact (avoid false positives)
            # Calculate force from impulse (F = dp/dt)
            total_impulse = np.zeros(3, dtype=np.float64)
            min_separation = float("inf")

            for point in contact.points:
                total_impulse += point.impulse
                # Separation: positive = gap, negative = penetration
                sep = getattr(point, "separation", -1.0)
                if sep < min_separation:
                    min_separation = sep

            force_mag = float(np.linalg.norm(total_impulse) / dt_safe)

            # Threshold from reference (0.001)
            if force_mag <= float(force_threshold):
                continue

            # Ignore if objects are separated (not actually touching)
            # Use 0.5mm margin (contact offset is usually 1-2mm)
            if min_separation > 5e-4:
                continue

            # Ensure robot_actor_name refers to the robot
            if actor0_is_robot:
                robot_actor_name = actor0_name
                env_actor_name = actor1_name
            else:
                robot_actor_name = actor1_name
                env_actor_name = actor0_name

            robot_actor_lower = robot_actor_name.lower()
            is_gripper = "gripper" in robot_actor_lower

            is_target_object = target_object_name in env_actor_name

            if is_gripper and is_target_object:
                success_events.append((robot_actor_name, env_actor_name))

            # Ignore collision if it contains gripper keywords
            is_ignored_link = any(
                k in robot_actor_lower for k in ["gripper", "wrist", "finger", "wheel"]
            )
            if not is_ignored_link:
                collision_events.append((robot_actor_name, env_actor_name))

    return success_events, collision_events


def update_hold_state(
    state: dict,
    now_s: float,
    has_gripper_target_contact: bool,
    hold_seconds: float = 2.0,
) -> tuple[dict, bool]:
    """
    Update a simple "hold for N seconds after lift" state machine.

    Metric:
    - This function is intentionally minimal: it only checks continuous contact.
    - The caller controls *when to start* counting (e.g., after lift) by calling it
      only after arming, or by resetting the state at the lift moment.
    - Once armed, if contact stays True continuously for hold_seconds -> success.
    - If contact breaks -> reset timer.

    Args:
        state: Mutable dict holding keys:
            - active: bool
            - t0: float
        now_s: Current wall time (seconds).
        has_gripper_target_contact: Contact flag from monitor.
        obj_p_world: (3,) object position in world frame.
        gripper_p_world: (3,) gripper reference position in world frame. (Unused; kept for minimal call-site churn.)

    Returns:
        (new_state, hold_succeeded)
    """
    if has_gripper_target_contact:
        if not bool(state["active"]):
            state["active"] = True
            state["t0"] = float(now_s)
            return state, False

        if float(now_s) - float(state["t0"]) >= float(hold_seconds):
            return state, True

        return state, False

    # No contact: reset timer
    state["active"] = False
    state["t0"] = 0.0
    return state, False
