from scipy.spatial.transform import Rotation as R


def select_diverse_grasps(grasp_poses, num_select):
    """
    Select diverse grasp poses using greedy farthest-point sampling.
    Diversity is measured by rotation angle difference between poses.

    Args:
        grasp_poses: List of 4x4 grasp pose matrices (already sorted by score)
        num_select: Number of diverse poses to select

    Returns:
        List of indices into grasp_poses for the selected diverse grasps
    """
    if len(grasp_poses) <= num_select:
        return list(range(len(grasp_poses)))

    # Extract rotation matrices
    rotations = [R.from_matrix(pose[:3, :3]) for pose in grasp_poses]

    # Start with the top-scored grasp (index 0)
    selected = [0]

    for _ in range(num_select - 1):
        max_min_dist = -1
        best_idx = -1

        for i in range(len(grasp_poses)):
            if i in selected:
                continue

            # Compute minimum angular distance to all selected poses
            min_dist = float("inf")
            for j in selected:
                # Angular distance in radians
                angle = (rotations[i].inv() * rotations[j]).magnitude()
                min_dist = min(min_dist, angle)

            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_idx = i

        if best_idx >= 0:
            selected.append(best_idx)

    return selected
