import numpy as np
import open3d as o3d
import small_gicp

from grasp_anywhere.utils.logger import log


class PointCloudGenerator:
    """
    Generates a point cloud from the robot's camera.

    It gets camera data from the robot interface and provides a method
    to get the latest point cloud.
    """

    def __init__(self, robot, camera_frame_id="head_camera_rgb_optical_frame"):
        """
        Initializes the PointCloudGenerator.

        Args:
            robot: The Fetch robot instance.
            camera_frame_id: The TF frame of the camera.
        """
        self.robot = robot
        self.camera_frame = camera_frame_id
        log.info("PointCloudGenerator initialized.")

    @property
    def camera_frame_id(self):
        return self.camera_frame

    def get_latest_point_cloud(self):
        """
        Generates a point cloud from the latest depth image.

        Returns:
            numpy.ndarray: An (N, 3) numpy array representing the point cloud in the camera frame.
                           Returns None if a point cloud could not be generated.
        """
        depth_image = self.robot.get_depth()
        camera_intrinsics = self.robot.get_camera_intrinsic()

        if depth_image is None:
            log.warning("No depth image received yet.")
            return None
        if camera_intrinsics is None:
            log.warning("No camera intrinsics received yet.")
            return None

        # Convert ROS Image message to numpy array
        depth_image = np.nan_to_num(depth_image)

        height, width = depth_image.shape

        # Get camera intrinsics from CameraInfo
        fx = camera_intrinsics[0, 0]
        fy = camera_intrinsics[1, 1]
        cx = camera_intrinsics[0, 2]
        cy = camera_intrinsics[1, 2]
        intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

        # Create Open3D depth image
        o3d_depth = o3d.geometry.Image(depth_image)

        # Create point cloud from depth in camera frame (no extrinsic transformation)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            o3d_depth, intrinsics, depth_scale=1.0
        )

        # Downsample the point cloud to reduce density
        pcd = pcd.voxel_down_sample(voxel_size=0.04)

        points = np.asarray(pcd.points)
        # Manually filter points to remove those beyond 3.0 meters
        max_depth = 3.0
        points = points[points[:, 2] <= max_depth]

        log.info(
            f"Generated point cloud with {len(points)} points in camera frame after depth filtering."
        )
        return points


def register_point_clouds_icp(
    source_pcd_np,
    target_pcd_np,
    icp_method="VGICP",
    downsampling_resolution=0.25,
    voxel_resolution=0.25,
    max_correspondence_distance=0.25,
    num_threads=4,
):
    """
    Registers a source point cloud to a target point cloud using VGICP or GICP.
    The source is transformed to align with the target.

    Args:
        source_pcd_np (np.ndarray): The source point cloud (Nx3) to be transformed.
        target_pcd_np (np.ndarray): The target point cloud (Mx3), the reference.
        icp_method (str): The ICP method to use. Either "VGICP" or "GICP".
        downsampling_resolution (float): Voxel size for downsampling.
        voxel_resolution (float): Voxel size for voxel grid.
        max_correspondence_distance (float): Maximum distance for matching points between point clouds.
        num_threads (int): Number of threads to use for parallel processing.
    Returns:
        np.ndarray: The transformed source point cloud, aligned to the target.
    """
    if source_pcd_np.shape[0] == 0:
        log.warning(f"{icp_method} registration skipped: source point cloud is empty.")
        return source_pcd_np
    if target_pcd_np.shape[0] == 0:
        log.warning(f"{icp_method} registration skipped: target point cloud is empty.")
        return source_pcd_np

    log.info(
        f"Starting {icp_method} registration. Source points: {len(source_pcd_np)}, Target points: {len(target_pcd_np)}"
    )

    result = small_gicp.align(
        target_pcd_np,
        source_pcd_np,
        registration_type=icp_method,
        downsampling_resolution=downsampling_resolution,
        voxel_resolution=voxel_resolution,
        max_correspondence_distance=max_correspondence_distance,
        num_threads=num_threads,
    )

    log.info(
        f"{icp_method} registration finished. Converged: {result.converged}, Iterations: {result.iterations}, "
        f"Num inliers: {result.num_inliers}"
    )

    if not result.converged:
        log.warning(
            f"{icp_method} registration did not converge. Error: {result.error}. Discarding alignment."
        )
        return np.empty((0, 3), dtype=source_pcd_np.dtype)

    transformation = result.T_target_source
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_pcd_np)
    source_pcd.transform(transformation)

    return np.asarray(source_pcd.points)


def register_point_clouds_for_pose_refinement(
    source_pcd_np,
    target_pcd_np,
    icp_method="VGICP",
    downsampling_resolution=0.25,
    voxel_resolution=0.25,
    max_correspondence_distance=0.25,
    num_threads=4,
):
    """
    Registers a source point cloud to a target point cloud using VGICP or GICP and returns the transform.

    Args:
        source_pcd_np (np.ndarray): The source point cloud (Nx3) to be transformed.
        target_pcd_np (np.ndarray): The target point cloud (Mx3), the reference.
        icp_method (str): The ICP method to use. Either "VGICP" or "GICP".
        downsampling_resolution (float): Voxel size for downsampling.
        voxel_resolution (float): Voxel size for voxel grid.
        max_correspondence_distance (float): Maximum distance for matching points between point clouds.
        num_threads (int): Number of threads to use for parallel processing.
    Returns:
        np.ndarray: The 4x4 transformation matrix that aligns the source to the target,
                    or None if registration fails.
    """
    if source_pcd_np.shape[0] == 0:
        log.warning(f"{icp_method} registration skipped: source point cloud is empty.")
        return None
    if target_pcd_np.shape[0] == 0:
        log.warning(f"{icp_method} registration skipped: target point cloud is empty.")
        return None

    log.info(f"Starting {icp_method} registration for pose refinement.")

    result = small_gicp.align(
        target_pcd_np,
        source_pcd_np,
        registration_type=icp_method,
        downsampling_resolution=downsampling_resolution,
        voxel_resolution=voxel_resolution,
        max_correspondence_distance=max_correspondence_distance,
        num_threads=num_threads,
    )

    log.info(
        f"{icp_method} registration finished. Converged: {result.converged}, Iterations: {result.iterations}, "
        f"Num inliers: {result.num_inliers}"
    )

    if not result.converged:
        log.warning(
            f"{icp_method} registration did not converge. Error: {result.error}. Discarding alignment."
        )
        return None

    return result.T_target_source


def get_pcd_from_mask(depth_image, mask, camera_intrinsics):
    """
    Extracts a 3D point cloud from a depth image using a 2D segmentation mask.

    Args:
        depth_image (np.ndarray): The depth image (height, width).
        mask (np.ndarray): The boolean segmentation mask (height, width).
        camera_intrinsics (np.ndarray): The 3x3 camera intrinsic matrix.

    Returns:
        np.ndarray: The resulting point cloud (Nx3) in the camera frame, or None if extraction fails.
    """
    if depth_image.shape != mask.shape:
        log.error(
            f"Shape mismatch: depth image is {depth_image.shape}, mask is {mask.shape}"
        )
        return None

    # Get the (y, x) pixel coordinates of the segmented object
    y_coords, x_coords = np.where(mask)

    # Get the depth values for these pixels (assumes depth is in meters)
    depth_values_m = depth_image[y_coords, x_coords]

    # Filter out zero-depth pixels, which are invalid
    valid_indices = np.where(depth_values_m > 0)
    x_coords = x_coords[valid_indices]
    y_coords = y_coords[valid_indices]
    depth_values_m = depth_values_m[valid_indices]

    if len(depth_values_m) == 0:
        log.error("Masked object has no valid depth values.")
        return None

    # Perform back-projection for only the valid points
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

    points_x = (x_coords - cx) * depth_values_m / fx
    points_y = (y_coords - cy) * depth_values_m / fy
    points_z = depth_values_m

    # Combine into a single point cloud array
    pcd_camera_frame = np.vstack((points_x, points_y, points_z)).T
    return pcd_camera_frame


def depth2pc(depth, K, rgb=None, voxel_size=0.05, stride=1):
    """
    Convert depth and intrinsics to point cloud and optionally point cloud color
    :param depth: hxw depth map in m
    :param K: 3x3 Camera Matrix with intrinsics
    :param rgb: Optional RGB image for coloring the point cloud
    :param voxel_size: Voxel size for downsampling (in meters). Default 0.05m for manipulation
    :param stride: Stride for depth map slicing (decimation). Default 1 (no slicing).
    :returns: (Nx3 point cloud, point cloud color)
    """
    if stride > 1:
        depth = depth[::stride, ::stride]
        K = K.copy()
        K[0, :] /= stride
        K[1, :] /= stride
        if rgb is not None:
            rgb = rgb[::stride, ::stride, :]

    mask = np.where(depth > 0)
    y, x = mask[0], mask[1]  # Note the order is y, x

    normalized_x = x.astype(np.float32) - K[0, 2]
    normalized_y = y.astype(np.float32) - K[1, 2]

    world_x = normalized_x * depth[y, x] / K[0, 0]
    world_y = normalized_y * depth[y, x] / K[1, 1]
    world_z = depth[y, x]

    if rgb is not None:
        rgb = rgb[y, x, :]

    pc = np.vstack((world_x, world_y, world_z)).T

    # Downsample the point cloud for manipulation
    # (Set voxel_size <= 0 to disable downsampling, useful for visualization/debug.)
    if len(pc) > 0 and voxel_size is not None and float(voxel_size) > 0.0:
        # Create Open3D point cloud for downsampling
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)

        if rgb is not None:
            pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float32) / 255.0)

        # Apply voxel downsampling
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)

        # Convert back to numpy arrays
        pc = np.asarray(pcd_downsampled.points)
        if rgb is not None and len(pcd_downsampled.colors) > 0:
            rgb = (np.asarray(pcd_downsampled.colors) * 255.0).astype(np.uint8)
        else:
            rgb = None

    return (pc, rgb)


def extract_point_clouds(
    depth, K, segmap=None, rgb=None, z_range=[0.2, 2.5], segmap_id=0
):
    """
    Converts depth map + intrinsics to point cloud.
    """
    if K.shape[0] != 3:
        K = np.array(K).reshape(3, 3)

    # Convert to pc
    pc_full, pc_colors = depth2pc(depth, K, rgb)

    # Threshold distance
    if pc_colors is not None:
        pc_colors = pc_colors[
            (pc_full[:, 2] < z_range[1]) & (pc_full[:, 2] > z_range[0])
        ]
    pc_full = pc_full[(pc_full[:, 2] < z_range[1]) & (pc_full[:, 2] > z_range[0])]

    # Extract instance point clouds from segmap and depth map
    pc_segments = {}
    if segmap is not None:
        pass

    return pc_full, pc_segments, pc_colors


def refine_camera_pose(depth, camera_intrinsic, camera_pose, map_points):
    """
    Refines camera pose by registering observed point cloud with a map.
    """
    if map_points is None or len(map_points) == 0:
        log.warning("Refinement failed: map point cloud is empty.")
        return camera_pose

    observed_pcd_cam, _, _ = extract_point_clouds(depth, camera_intrinsic)
    if observed_pcd_cam is None or len(observed_pcd_cam) == 0:
        log.warning("Refinement failed: observed point cloud is empty.")
        return camera_pose

    observed_pcd_world_h = np.hstack(
        (observed_pcd_cam, np.ones((observed_pcd_cam.shape[0], 1)))
    )
    observed_pcd_world = (camera_pose @ observed_pcd_world_h.T).T[:, :3]

    transformation = register_point_clouds_for_pose_refinement(
        observed_pcd_world, map_points
    )

    if transformation is not None:
        log.info("Camera pose refined successfully.")
        refined_pose = transformation @ camera_pose
        return refined_pose
    else:
        log.warning("Refinement failed. Returning original camera pose.")
        return camera_pose
