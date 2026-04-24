import numpy as np
import vamp


def _to_list_sequence(path_like):
    """
    Normalize a path returned from VAMP into a Python list-of-lists.
    Handles bound/config types that expose to_list(), numpy arrays, and lists.
    """
    sequence = []
    for elem in path_like:
        if isinstance(elem, list):
            sequence.append(elem)
        elif isinstance(elem, np.ndarray):
            sequence.append(elem.tolist())
        elif hasattr(elem, "to_list"):
            sequence.append(elem.to_list())
        elif hasattr(elem, "config"):
            sequence.append(list(elem.config))
        else:
            sequence.append(list(elem))
    return sequence


def plan_rrtc_whole_body(
    start_joints,
    goal_joints,
    start_base,
    goal_base,
    env,
    vamp_module,
    plan_settings,
    simp_settings,
    sampler,
    interpolate_density=0.08,
):
    """
    Whole-body planning using multilayer RRTC with whole_body_simplify and interpolation.
    Returns a dict with keys: success, stats, arm_path, base_configs.
    """
    # Create the config
    if hasattr(vamp, "HybridAStarConfig"):
        config = vamp.HybridAStarConfig()
        # Set your new tuning value
        config.reverse_penalty = 50
        plan_settings.hybrid_astar_config = config

    result = vamp_module.multilayer_rrtc(
        start_joints,
        goal_joints,
        start_base,
        goal_base,
        env,
        plan_settings,
        sampler,
    )

    if not result.is_successful():
        stats = {
            "arm_planning_time_ms": result.arm_result.nanoseconds / 1e6,
            "base_planning_time_ms": result.base_result.nanoseconds / 1e6,
            "total_planning_time_ms": result.nanoseconds / 1e6,
            "planning_iterations": result.arm_result.iterations,
            "base_planning_iterations": result.base_result.iterations,
            "planning_graph_size": (
                sum(result.arm_result.size) if result.arm_result.size else 0
            ),
        }
        return {
            "success": False,
            "stats": stats,
            "arm_path": None,
            "base_configs": None,
        }

    # Convert arm path to list for simplification
    arm_path_list = _to_list_sequence(result.arm_result.path)
    base_path = result.base_path

    whole_body_result = vamp_module.whole_body_simplify(
        arm_path_list, base_path, env, simp_settings, sampler
    )

    # Interpolate both arm and base paths together for synchronized waypoints
    whole_body_result.interpolate(interpolate_density)

    arm_path = _to_list_sequence(whole_body_result.arm_result.path)
    base_configs = _to_list_sequence(whole_body_result.base_path)

    stats = {
        "arm_planning_time_ms": result.arm_result.nanoseconds / 1e6,
        "base_planning_time_ms": result.base_result.nanoseconds / 1e6,
        "total_planning_time_ms": result.nanoseconds / 1e6,
        "planning_iterations": result.arm_result.iterations,
        "base_planning_iterations": result.base_result.iterations,
        "planning_graph_size": (
            sum(result.arm_result.size) if result.arm_result.size else 0
        ),
        "simplification_time_ms": whole_body_result.arm_result.nanoseconds / 1e6,
    }

    return {
        "success": True,
        "stats": stats,
        "arm_path": arm_path,
        "base_configs": base_configs,
    }


def plan_fcit_wb_whole_body(
    start_joints,
    goal_joints,
    start_base,
    goal_base,
    env,
    vamp_module,
    bounds_xy,
    random_generator=None,
    settings_overrides=None,
    interpolate_density=0.03,
):
    """
    Whole-body planning using FCIT* across arm+base.

    Args:
            bounds_xy: tuple (x_min, x_max, y_min, y_max)
            settings_overrides: dict to override default FCITSettings fields

    Returns:
            Dict with keys: success, stats, arm_path, base_configs
    """
    if bounds_xy is None:
        raise ValueError("bounds_xy must be provided for FCIT* whole-body planner")

    # RNG
    rng = random_generator or vamp_module.halton()

    # FCIT settings
    neighbor_params = vamp.FCITNeighborParams(
        vamp_module.dimension(), vamp_module.space_measure()
    )
    settings = vamp.FCITSettings(neighbor_params)
    # Reasonable defaults similar to provided script
    settings.max_iterations = 200
    settings.max_samples = 8192
    settings.batch_size = 4096
    settings.reverse_weight = 10.0
    settings.optimize = True

    if settings_overrides:
        for k, v in settings_overrides.items():
            setattr(settings, k, v)

    x_min, x_max, y_min, y_max = bounds_xy
    res = vamp_module.fcit_wb(
        start_joints,
        goal_joints,
        start_base,
        goal_base,
        env,
        settings,
        rng,
        x_min,
        x_max,
        y_min,
        y_max,
    )

    if not (res.validate_paths() and len(res.arm_result.path) >= 2):
        stats = {
            "arm_planning_time_ms": (
                res.arm_result.nanoseconds / 1e6
                if hasattr(res.arm_result, "nanoseconds")
                else 0.0
            ),
            "planning_iterations": (
                res.arm_result.iterations
                if hasattr(res.arm_result, "iterations")
                else 0
            ),
            "planning_graph_size": (
                sum(res.arm_result.size)
                if hasattr(res.arm_result, "size") and res.arm_result.size
                else 0
            ),
        }
        return {
            "success": False,
            "stats": stats,
            "arm_path": None,
            "base_configs": None,
        }

    # Optionally interpolate to synchronize and densify
    if interpolate_density is not None:
        try:
            res.interpolate(interpolate_density)
        except Exception:
            pass

    arm_path = _to_list_sequence(res.arm_result.path)
    base_configs = _to_list_sequence(res.base_path)

    total_planning_time_ms = 0.0
    if hasattr(res, "nanoseconds"):
        total_planning_time_ms = res.nanoseconds / 1e6
    stats = {
        "arm_planning_time_ms": (
            res.arm_result.nanoseconds / 1e6
            if hasattr(res.arm_result, "nanoseconds")
            else 0.0
        ),
        "planning_iterations": (
            res.arm_result.iterations if hasattr(res.arm_result, "iterations") else 0
        ),
        "planning_graph_size": (
            sum(res.arm_result.size)
            if hasattr(res.arm_result, "size") and res.arm_result.size
            else 0
        ),
        "total_planning_time_ms": total_planning_time_ms,
    }

    return {
        "success": True,
        "stats": stats,
        "arm_path": arm_path,
        "base_configs": base_configs,
    }


def plan_base_only(
    start_base,
    goal_base,
    env,
    vamp_module,
    simp_settings,
    sampler,
    start_arm,
    settings_overrides=None,
    interpolate_density=0.03,
):
    """
    Plans a base-only path using Hybrid A*, then creates a whole-body path
    with fixed arm configuration to perform interpolation using VAMP.

    Args:
        start_base: [x, y, theta]
        goal_base: [x, y, theta]
        env: VAMP environment
        vamp_module: VAMP module
        simp_settings: settings for simplification (used to create WB path)
        sampler: sampler for simplification
        start_arm: 8-DOF arm config (Tuck joints) to pair with base path
        settings_overrides: Optional dict for HybridAStarConfig
        interpolate_density: Density for interpolation

    Returns:
        Dict with keys: success, stats, base_configs
    """
    if not hasattr(vamp_module, "HybridAStar") or not hasattr(
        vamp_module, "MobileBaseConfiguration"
    ):
        raise ImportError(
            "VAMP module does not support HybridAStar or MobileBaseConfiguration."
        )

    start_base_config = vamp_module.MobileBaseConfiguration(start_base)
    goal_base_config = vamp_module.MobileBaseConfiguration(goal_base)

    hybrid_config = vamp_module.HybridAStarConfig()

    # Apply settings overrides if any
    if settings_overrides:
        for k, v in settings_overrides.items():
            if hasattr(hybrid_config, k):
                setattr(hybrid_config, k, v)

    # config for reverse penalty and other tuning from example?
    # hybrid_config.reverse_penalty = 50 # Example value
    # For now we use default or minimal overrides

    base_path_objs = []  # Vector to store result

    success = vamp_module.HybridAStar.plan(
        start_base_config, goal_base_config, env, hybrid_config, base_path_objs
    )

    if not success:
        stats = {
            "base_path_length": 0,
        }
        return {
            "success": False,
            "stats": stats,
            "base_configs": None,
        }

    # Construct constant arm path
    arm_path_list = [start_arm for _ in range(len(base_path_objs))]

    # Perform Whole Body Interpolation (via Clean/Simplify + Interpolate)
    # We use whole_body_simplify to wrap it into a WholeBodyPath structure that supports interpolate.
    # Note: simplify might modify the path. If we want ONLY interpolation, we might check if there's direct construct.
    # But whole_body_simplify is the standard way here.

    wb_res = vamp_module.whole_body_simplify(
        arm_path_list, base_path_objs, env, simp_settings, sampler
    )

    # Interpolate
    wb_res.interpolate(interpolate_density)

    # Extract interpolated base path
    base_configs = _to_list_sequence(wb_res.base_path)
    stats = {
        "base_path_length": len(base_configs),
    }

    return {
        "success": True,
        "stats": stats,
        "base_configs": base_configs,
    }
