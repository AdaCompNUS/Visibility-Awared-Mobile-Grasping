import numpy as np
import torch
from pytorch_kinematics.transforms.rotation_conversions import matrix_to_euler_angles
from scipy.spatial.transform import Rotation as R

from grasp_anywhere.dataclass.reachability_map import (
    CapabilityMap,
    InverseReachabilityMap,
    ReachabilityMap,
)


def query_capability_score(
    capability_map: CapabilityMap,
    base_pose: list[float],
    target_pose: list[float],
) -> float:
    """
    Query capability score for a given base pose and target pose.

    Args:
        capability_map: CapabilityMap instance containing poses, scores, and kdtree
        base_pose: [x, y, theta] of base (2D pose on ground plane)
        target_pose: [x, y, z, qx, qy, qz, qw] of target ee pose in WORLD frame

    Returns:
        score (0 if unreachable)
    """
    # 1. Construct T_world_base
    T_world_base = np.eye(4)
    T_world_base[:3, :3] = R.from_euler("z", base_pose[2]).as_matrix()
    T_world_base[0, 3] = base_pose[0]
    T_world_base[1, 3] = base_pose[1]

    # 2. Construct T_world_target
    T_world_target = np.eye(4)
    tgt_pos = target_pose[:3]
    tgt_quat = target_pose[3:]
    T_world_target[:3, :3] = R.from_quat(tgt_quat).as_matrix()
    T_world_target[:3, 3] = tgt_pos

    # 3. Compute T_base_target = inv(T_world_base) @ T_world_target
    T_base_target = np.linalg.inv(T_world_base) @ T_world_target

    # 4. Extract relative position and orientation
    rel_pos = T_base_target[:3, 3]
    rel_rot = T_base_target[:3, :3]

    # Note: The map stores Euler angles in 'xyz' order (roll, pitch, yaw)
    rel_euler = R.from_matrix(rel_rot).as_euler("xyz")

    # Combine into 6D pose for KDTree query and comparison
    rel = np.concatenate([rel_pos, rel_euler])

    # 5. Query KDTree with relative position
    # Find all neighbors within 5cm (position only)
    indices: list[int] = capability_map.kdtree.query_ball_point(rel_pos, r=0.05)

    if not indices:
        return 0.0

    # Get candidates and check orientation
    candidates_idx: np.ndarray = np.array(indices)
    candidate_poses: np.ndarray = capability_map.poses[candidates_idx]

    # Calculate angular differences for all candidates
    ang_diff: np.ndarray = np.abs(rel[3:] - candidate_poses[:, 3:])
    ang_diff = np.minimum(ang_diff, 2 * np.pi - ang_diff)  # Wrap around

    # Check if all Euler angles are within tolerance (0.3 rad)
    matches: np.ndarray = np.all(ang_diff < 0.3, axis=1)

    if np.any(matches):
        return float(np.max(capability_map.scores[candidates_idx[matches]]))

    return 0.0


def query_reachability_score(
    reach_map: ReachabilityMap,
    base_config: list[float],
    target_position: list[float],
) -> float:
    """Query a reachability score for reaching a *world-frame* target location (x,y,z).

    This is the "base score" query (no theta enumeration): you provide a specific base pose
    and get back a single reachability score from the position-only voxel map.

    Assumption: any end-effector orientation at the (x,y,z) location is acceptable.

    Args:
        pos_reach_vox: PositionOnlyReachabilityVoxelMap storing reachable EE positions in the
            base/reachability-map frame (orientation marginalized).
        base_config: [x, y, theta] base pose in WORLD frame (theta is yaw).
        target_position: [x, y, z] target position in WORLD frame.

    Returns:
        score (0.0 if unreachable / voxel not present).
    """
    x_t, y_t, z_t = [float(v) for v in target_position]
    x_b, y_b, theta = [float(v) for v in base_config]

    # Relative position of target in base frame: p_b = Rz(-theta) * (p_w - p_base_w)
    dx = x_t - x_b
    dy = y_t - y_b
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    rel_x = c * dx + s * dy
    rel_y = -s * dx + c * dy
    rel_z = z_t  # base is assumed on ground (z=0)

    vs = float(reach_map.voxel_size)
    origin = reach_map.origin.astype(np.float32)
    rel = np.array([rel_x, rel_y, rel_z], dtype=np.float32)
    ijk = np.floor((rel - origin) / vs).astype(np.int32)

    matches = np.where(np.all(reach_map.voxels_ijk == ijk[None, :], axis=1))[0]
    if matches.size == 0:
        return 0.0

    return float(reach_map.scores[int(matches[0])])


def query_inverse_reachability_map(
    irm: InverseReachabilityMap,
    target_pose: list[float],
    cartesian_res: float = 0.05,
    ang_thresh: float = np.pi / 8,
) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """Given a *world-frame* EE pose, return candidate base configs [x,y,theta] with scores.

    Args:
        irm: InverseReachabilityMap containing `inv_transf_batch` and `M_scores`.
        target_pose: [x, y, z, qx, qy, qz, qw] target pose in WORLD frame.
        cartesian_res: Ground filter tolerance around z=0 (meters).
        ang_thresh: Allowed base roll/pitch magnitude threshold (radians), applied as in
            `query_inverse_reachability_map`.

    Returns:
        tuple[np.ndarray[float], np.ndarray[float]]: base configs [x,y,theta] with scores.
    """
    if len(target_pose) != 7:
        raise ValueError(
            f"target_pose must be [x,y,z,qx,qy,qz,qw] (len=7). Got len={len(target_pose)}"
        )

    pos = np.asarray(target_pose[:3], dtype=np.float32)
    quat = np.asarray(target_pose[3:], dtype=np.float32)

    goal_transf = torch.zeros(
        (4, 4), dtype=irm.inv_transf_batch.dtype, device=irm.inv_transf_batch.device
    )
    goal_transf[:3, :3] = torch.tensor(
        R.from_quat(quat).as_matrix(),
        dtype=goal_transf.dtype,
        device=goal_transf.device,
    )
    goal_transf[:3, 3] = torch.tensor(
        pos, dtype=goal_transf.dtype, device=goal_transf.device
    )
    goal_transf[3, 3] = 1.0

    # Broadcast goal transform to batch size (N,4,4)
    goal_transf_batch = goal_transf.unsqueeze(0).repeat(
        irm.inv_transf_batch.shape[0], 1, 1
    )

    # Base transforms in world
    base_transf_batch = torch.bmm(goal_transf_batch, irm.inv_transf_batch)

    # Filter ground poses (base z near 0)
    ground_ind = (base_transf_batch[:, 2, 3] > (-cartesian_res / 2)) & (
        base_transf_batch[:, 2, 3] <= (cartesian_res / 2)
    )
    base_transf_batch = base_transf_batch[ground_ind]
    scores_filtered = irm.M_scores[ground_ind]

    if base_transf_batch.shape[0] == 0:
        return np.empty((0, 3)), np.empty((0,))

    # Filter by roll/pitch (keep base upright-ish)
    base_euler = matrix_to_euler_angles(base_transf_batch[:, :3, :3], "ZYX")
    base_poses_6d = torch.hstack((base_transf_batch[:, :3, 3], base_euler))
    filtered_ind = (
        (base_poses_6d[:, 4:6] > -ang_thresh) & (base_poses_6d[:, 4:6] <= ang_thresh)
    ).all(dim=1)

    valid_poses = base_poses_6d[filtered_ind].cpu().numpy()
    valid_scores = scores_filtered[filtered_ind].cpu().numpy()

    if valid_poses.shape[0] == 0:
        return np.empty((0, 3)), np.empty((0,))

    # base_poses_6d is [x, y, z, yaw, pitch, roll] in this codepath
    base_configs = valid_poses[:, [0, 1, 3]]
    return base_configs, valid_scores


if __name__ == "__main__":
    import time

    # Example map path - adjust as needed or use a dummy for syntax check
    capability_map_path = "resources/capability_map.pkl"
    reachability_map_path = "resources/reachability_map.pkl"
    irm_path = "resources/inverse_reachability_map.pkl"

    print("Loading capability map...")
    capability_map = CapabilityMap.from_file(capability_map_path)

    print("Loading reachability voxel map...")
    reachability_map = ReachabilityMap.from_file(reachability_map_path)

    print("Loading inverse reachability map...")
    irm = InverseReachabilityMap.from_file(irm_path)

    # Generate test data
    np.random.seed(42)
    n_tests: int = 1000

    print(f"Generating {n_tests} test queries...")
    test_data: list[tuple[list[float], list[float]]] = []

    # Base poses: robot at different positions/orientations - passed as [x, y, theta]
    base_poses: list[list[float]] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.5, 0.785],
        [0.5, -0.3, -0.5],
    ]

    # Target poses: sample workspace
    for _ in range(n_tests):
        base: list[float] = base_poses[np.random.randint(len(base_poses))]

        # Sample target in typical workspace
        x: float = np.random.uniform(0.2, 1.0)  # Forward
        y: float = np.random.uniform(-0.6, 0.6)  # Left/right
        z: float = np.random.uniform(0.3, 1.5)  # Height

        roll: float = np.random.uniform(-np.pi, np.pi)
        pitch: float = np.random.uniform(-np.pi / 2, np.pi / 2)
        yaw: float = np.random.uniform(-np.pi, np.pi)

        quat = R.from_euler("xyz", [roll, pitch, yaw]).as_quat()

        # Target is now [x, y, z, qx, qy, qz, qw]
        target: list[float] = [x, y, z, quat[0], quat[1], quat[2], quat[3]]
        test_data.append((base, target))

    # Run benchmark
    print(f"Running {n_tests} queries...\n")

    start_time: float = time.time()
    scores: list[float] = []
    reachable_count: int = 0

    for base, target in test_data:
        score: float = query_capability_score(capability_map, base, target)
        scores.append(score)
        if score > 0:
            reachable_count += 1

    end_time: float = time.time()

    # Statistics
    total_time: float = end_time - start_time
    avg_time: float = total_time / n_tests

    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total queries:        {n_tests}")
    print(
        f"Reachable:            {reachable_count} ({100*reachable_count/n_tests:.1f}%)"
    )
    print(
        f"Unreachable:          {n_tests - reachable_count} ({100*(n_tests-reachable_count)/n_tests:.1f}%)"
    )
    print(f"\nTotal time:           {total_time:.4f} seconds")
    print(f"Average query time:   {avg_time*1000:.4f} ms")
    print(f"Queries per second:   {n_tests/total_time:.1f}")
    print("\nScore statistics (reachable only):")
    reachable_scores: list[float] = [s for s in scores if s > 0]
    if reachable_scores:
        print(f"  Min:  {min(reachable_scores):.6f}")
        print(f"  Max:  {max(reachable_scores):.6f}")
        print(f"  Mean: {np.mean(reachable_scores):.6f}")
    print("=" * 60)

    # ---------------------------------------------------------
    # Benchmark / Test for Position-only Reachability Score
    # ---------------------------------------------------------
    print("\n[Testing Reachability Score]")
    n_reach_tests: int = 1000

    # Generate randomized (base, target_xyz) queries similar to the capability-map benchmark.
    np.random.seed(7)
    reach_test_data: list[tuple[list[float], list[float]]] = []

    base_poses_reach: list[list[float]] = [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.0, np.pi / 2],
        [0.5, 0.0, -np.pi / 2],
        [0.0, 0.5, np.pi],
    ]

    for _ in range(n_reach_tests):
        base_cfg = base_poses_reach[np.random.randint(len(base_poses_reach))]
        x = float(np.random.uniform(0.2, 1.0))
        y = float(np.random.uniform(-0.6, 0.6))
        z = float(np.random.uniform(0.3, 1.5))
        reach_test_data.append((base_cfg, [x, y, z]))

    print(f"Running {n_reach_tests} reachability-score queries...\n")
    t0 = time.time()
    reach_scores: list[float] = []
    reach_count = 0

    for base_cfg, tgt_xyz in reach_test_data:
        s = query_reachability_score(reachability_map, base_cfg, tgt_xyz)
        reach_scores.append(s)
        if s > 0:
            reach_count += 1

    t1 = time.time()
    total_time = t1 - t0
    avg_time = total_time / n_reach_tests

    print("=" * 60)
    print("REACHABILITY SCORE RESULTS (position-only)")
    print("=" * 60)
    print(f"Total queries:        {n_reach_tests}")
    print(f"Reachable:            {reach_count} ({100*reach_count/n_reach_tests:.1f}%)")
    print(
        f"Unreachable:          {n_reach_tests - reach_count} ({100*(n_reach_tests-reach_count)/n_reach_tests:.1f}%)"
    )
    print(f"\nTotal time:           {total_time:.4f} seconds")
    print(f"Average query time:   {avg_time*1000:.4f} ms")
    print(f"Queries per second:   {n_reach_tests/total_time:.1f}")
    print("\nScore statistics (reachable only):")
    reachable_scores = [s for s in reach_scores if s > 0]
    if reachable_scores:
        print(f"  Min:  {min(reachable_scores):.6f}")
        print(f"  Max:  {max(reachable_scores):.6f}")
        print(f"  Mean: {np.mean(reachable_scores):.6f}")
    print("=" * 60)

    # ---------------------------------------------------------
    # Benchmark / Test for Inverse Reachability Map (IRM)
    # ---------------------------------------------------------
    print("\n[Testing Inverse Reachability Map]")

    # Use a target coordinate that is likely reachable
    target_3d = [0.8, 0.0, 0.8]
    print(f"Querying IRM for Target: {target_3d}")

    t_start = time.time()
    # Provide an explicit world-frame EE pose (identity quaternion) for the IRM query.
    ee_pose_world = [target_3d[0], target_3d[1], target_3d[2], 0.0, 0.0, 0.0, 1.0]
    result = query_inverse_reachability_map(irm, ee_pose_world)
    t_end = time.time()

    base_configs, scores = result

    print(f"IRM Query finished in {(t_end - t_start)*1000:.4f} ms")
    print(f"Found {len(base_configs)} valid base options.")
    print(f"Min score: {min(scores):.6f}")
    print(f"Max score: {max(scores):.6f}")
    print(f"Mean score: {np.mean(scores):.6f}")
    print("=" * 60)
