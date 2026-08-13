import numpy as np


def get_zoom_transform(object_center, target_distance=0.3):
    """
    Computes the transformation matrix that translates the point cloud such that
    the object center is at 'target_distance' from the origin, along the original
    line from origin to object center.

    Effectively 'zooms in' by moving the object closer to the camera.

    Args:
        object_center (np.ndarray): (3,) Object center in Original (Camera) Frame
        target_distance (float): Desired distance from origin to object center

    Returns:
        T (4,4): Transformation matrix (Translation only)
    """
    object_center = np.asarray(object_center)
    current_dist = np.linalg.norm(object_center)

    if current_dist < 1e-6:
        # Already at origin or invalid, return identity
        return np.eye(4)

    # Direction from origin to object
    direction = object_center / current_dist

    # We want new_center = target_distance * direction
    # new_center = old_center + translation
    # translation = new_center - old_center
    #             = (target_distance * direction) - (current_dist * direction)
    #             = (target_distance - current_dist) * direction

    translation_vector = (target_distance - current_dist) * direction

    T = np.eye(4)
    T[:3, 3] = translation_vector

    return T


def transfer_point_cloud(pcd, T):
    """
    Transfers point cloud using transformation matrix T.

    Args:
        pcd (np.ndarray): (N, 3) points
        T (np.ndarray): (4, 4) transformation matrix

    Returns:
        pcd_out (np.ndarray): (N, 3) transformed points
    """
    pcd = np.asarray(pcd)
    if len(pcd) == 0:
        return pcd

    N = pcd.shape[0]
    pad = np.ones((N, 1), dtype=pcd.dtype)
    pcd_h = np.hstack((pcd, pad))

    pcd_out_h = (T @ pcd_h.T).T
    return pcd_out_h[:, :3]


def transfer_grasps_back(grasps_transformed, T):
    """
    Transfers grasps from Transformed frame back to Original frame.

    Args:
        grasps_transformed (np.ndarray): (N, 4, 4) or (4, 4) grasp poses
        T (np.ndarray): (4, 4) transformation matrix used to transform points

    Returns:
        grasps_original (np.ndarray): Transformed grasps
    """
    # We want inverted transform: P_old = T_inv @ P_new
    T_inv = np.linalg.inv(T)

    if grasps_transformed.ndim == 2:
        return T_inv @ grasps_transformed
    else:
        return np.matmul(T_inv, grasps_transformed)


def crop_point_cloud(pcd, center, radius=0.3):
    """
    Crops the point cloud to keep points within 'radius' of 'center'.

    Args:
        pcd (np.ndarray): (N, 3) points
        center (np.ndarray): (3,) Center point
        radius (float): Radius to keep

    Returns:
        pcd_cropped (np.ndarray): Points within radius
    """
    if len(pcd) == 0:
        return pcd

    dists = np.linalg.norm(pcd - center, axis=1)
    mask = dists <= radius
    return pcd[mask]


def densify_point_cloud(pcd, k=5):
    """
    Densifies the point cloud by adding midpoints between each point and its k nearest neighbors.

    Args:
        pcd (np.ndarray): (N, 3) points
        k (int): Number of nearest neighbors to consider

    Returns:
        pcd_dense (np.ndarray): Densified point cloud
    """
    if len(pcd) < k + 1:
        return pcd

    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(pcd)
    distances, indices = nbrs.kneighbors(pcd)

    new_points = []
    # indices[:, 0] is the point itself, so we start from 1
    for i in range(len(pcd)):
        p_curr = pcd[i]
        for j in range(1, k + 1):
            p_neighbor = pcd[indices[i, j]]
            # Add midpoint
            new_points.append((p_curr + p_neighbor) / 2.0)

    if not new_points:
        return pcd

    pcd_dense = np.vstack((pcd, np.array(new_points)))
    return pcd_dense


if __name__ == "__main__":
    # Test
    # Assume object is at (0, 0, 2.0) in camera frame
    obj_center = np.array([0.0, 0.0, 2.0])
    print(f"Original Object Center: {obj_center}")
    print(f"Original Distance: {np.linalg.norm(obj_center)}")

    target_dist = 0.3
    T = get_zoom_transform(obj_center, target_distance=target_dist)

    print("Transformation Matrix:")
    print(T)

    # Apply transform
    obj_center_h = np.append(obj_center, 1.0)
    new_center = (T @ obj_center_h)[:3]

    print(f"New Center: {new_center}")
    new_dist = np.linalg.norm(new_center)
    print(f"New Distance: {new_dist}")

    err = abs(new_dist - target_dist)
    print(f"Error: {err}")

    assert err < 1e-5

    # Test 2: Arbitrary direction
    obj_center_2 = np.array([1.0, 1.0, 1.0])
    target_dist_2 = 0.5
    T2 = get_zoom_transform(obj_center_2, target_distance=target_dist_2)
    new_center_2 = (T2 @ np.append(obj_center_2, 1.0))[:3]
    new_dist_2 = np.linalg.norm(new_center_2)
    print(f"Test 2 New Dist (Expected {target_dist_2}): {new_dist_2}")
    assert abs(new_dist_2 - target_dist_2) < 1e-5

    # Check direction preservation
    orig_dir = obj_center_2 / np.linalg.norm(obj_center_2)
    new_dir = new_center_2 / new_dist_2
    assert np.allclose(orig_dir, new_dir)
    print("Test 2 Direction Preserved")

    print("All Tests Passed!")

    dist = np.linalg.norm(obj_center)
    expected_p_obj_cam = np.array([0, 0, dist])

    print(f"Expected Obj in Best View: {expected_p_obj_cam}")

    err_obj = np.linalg.norm(obj_center - expected_p_obj_cam)
    print(f"Object Pos Error: {err_obj}")

    assert err_obj < 1e-5

    print("Test Passed!")
