# FILE: visualization_utils.py
import cv2
import numpy as np
import open3d as o3d

from grasp_anywhere.utils.logger import log

# Optional Pyrender imports for ROS-independent real-time 3D visualization
try:
    import pyrender
    import trimesh
except Exception:
    pyrender = None
    trimesh = None


def visualize_grasp(
    pred_grasps_cam, depth, K, scores, top_grasp_idx=None, rgb=None, z_range=[0.2, 2.5]
):
    """
    This is the EXACT visualization function from your provided source code.
    """
    from grasp_anywhere.utils.perception_utils import extract_point_clouds

    cam_K = np.array(K).reshape(3, 3)
    pc_full, pc_segments, pc_colors = extract_point_clouds(
        depth, cam_K, rgb=rgb, z_range=z_range
    )

    # Create point cloud visualization
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_full)
    if rgb is not None:
        pc_colors = pc_colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(pc_colors)

    # Create gripper visualization for each grasp
    gripper_geometries = []

    for i, grasp in enumerate(pred_grasps_cam):
        grasp_mat = np.array(grasp).reshape(4, 4)

        # Define gripper dimensions (in meters) - matching reference code
        width = 0.12  # Gripper width
        depth = 0.1  # Gripper depth

        # Define 6 control points for gripper in local coordinates
        # Two points for base, four points for fingers
        points = [
            [0, 0, 0],  # center of the gripper
            [width / 2, 0, 0],  # right base
            [-width / 2, 0, 0],  # left base
            [0, 0, -0.06],  # handle
            [width / 2, 0, depth],  # Right finger front
            [-width / 2, 0, depth],  # Left finger front
        ]

        # Convert points to homogeneous coordinates and transform
        points = np.array(points)
        points_homog = np.concatenate([points, np.ones((6, 1))], axis=1)
        # Using np.dot instead of @ for compatibility
        transformed_points = np.dot(grasp_mat, points_homog.T).T[:, :3]

        # Create lines connecting the points
        lines = [
            [0, 1],  # Right base
            [0, 2],  # Left base
            [0, 3],  # handle
            [1, 4],  # Right finger
            [2, 5],  # Left finger
        ]

        # Create LineSet for the gripper
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(transformed_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)

        # Color the lines based on whether it's the top grasp
        if top_grasp_idx is not None and i == top_grasp_idx:
            colors = [[0, 0, 1] for _ in range(len(lines))]  # Blue color for top grasp
        else:
            colors = [[1, 0, 0] for _ in range(len(lines))]  # Red color for others

        line_set.colors = o3d.utility.Vector3dVector(colors)
        gripper_geometries.append(line_set)

    # Create visualization using the same approach as reference code
    # Use draw_geometries for simpler visualization
    o3d.visualization.draw_geometries([pcd] + gripper_geometries)


def visualize_prepose_pipeline(
    base_config,
    prepose_world,
    combined_points=None,
    object_center_world=None,
    sampling_radius=None,
    object_pcd_world=None,
    is_candidate=False,
    sample_info="",
):
    """
    Visualize base config, pre-pose, object center, sampling sphere and map in Open3D.
    Args:
        base_config: [x, y, theta] of the base config
        prepose_world: 4x4 pre-pose in world frame
        combined_points: Nx3 array of map points (optional)
        object_center_world: [x,y,z] of the object center (optional)
        sampling_radius: radius of the sampling sphere (optional)
        object_pcd_world: o3d.geometry.PointCloud of the segmented object (optional)
        is_candidate: bool, True if the pose is just a candidate
        sample_info: str, information about the sample number
    """
    import open3d as o3d

    title = "Pre-Pose Visualization"
    if is_candidate:
        title = f"Candidate Pre-Pose | {sample_info}"
        log.info(f"=== Visualizing Candidate Pre-Pose ({sample_info}) ===")
    else:
        log.info("=== Creating Final Pose Visualization ===")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1200, height=800)

    # Add coordinate frame for world origin
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.3, origin=[0, 0, 0]
    )
    vis.add_geometry(world_frame)

    # Add base config coordinate frame
    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.15, origin=[0, 0, 0]
    )

    from scipy.spatial.transform import Rotation as R

    base_transform = np.eye(4)
    base_transform[:3, :3] = R.from_euler("z", base_config[2]).as_matrix()
    base_transform[0, 3] = base_config[0]
    base_transform[1, 3] = base_config[1]

    base_frame.transform(base_transform)
    vis.add_geometry(base_frame)

    # Add pre-pose coordinate frame
    prepose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.15, origin=[0, 0, 0]
    )
    prepose_frame.transform(prepose_world)
    vis.add_geometry(prepose_frame)

    # Add pre-pose position marker (sphere)
    prepose_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
    prepose_color = (
        [1, 1, 0] if is_candidate else [0, 1, 0]
    )  # Yellow for candidate, Green for final
    prepose_sphere.paint_uniform_color(prepose_color)
    prepose_sphere.translate(prepose_world[:3, 3])
    vis.add_geometry(prepose_sphere)

    # Add object center marker (sphere)
    if object_center_world is not None:
        center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
        center_sphere.paint_uniform_color([0, 0, 1])  # Blue for object center
        center_sphere.translate(object_center_world)
        vis.add_geometry(center_sphere)
        log.info(f"Added object center at {object_center_world}")

    # Add the segmented object point cloud
    if object_pcd_world is not None:
        object_pcd_world.paint_uniform_color([1, 0, 1])  # Magenta
        vis.add_geometry(object_pcd_world)
        log.info("Added segmented object point cloud to visualization.")

    # Add sampling sphere (wireframe)
    if object_center_world is not None and sampling_radius is not None:
        sampling_sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(
            radius=sampling_radius
        )
        sampling_sphere_mesh.translate(object_center_world)
        sampling_sphere_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(
            sampling_sphere_mesh
        )
        sampling_sphere_wireframe.paint_uniform_color(
            [0.8, 0.8, 0.8]
        )  # Light gray for sampling sphere
        vis.add_geometry(sampling_sphere_wireframe)
        log.info(f"Added sampling sphere with radius {sampling_radius}")

    # Add map point cloud if available
    if combined_points is not None:
        map_pcd = o3d.geometry.PointCloud()
        map_pcd.points = o3d.utility.Vector3dVector(combined_points)
        map_pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Gray for map
        vis.add_geometry(map_pcd)
        log.info(f"Added {len(combined_points)} map points to visualization")

    # Log pose information
    legend_pose_name = "Candidate Pre-pose" if is_candidate else "Final Pre-pose"
    log.info("=== Visualization Legend ===")
    log.info("- Large RGB frame: World origin")
    log.info(
        f"- Small RGB frame + {'Yellow' if is_candidate else 'Green'} sphere: {legend_pose_name}"
    )
    log.info("- Blue sphere: Calculated Object Center")
    log.info("- Magenta points: Segmented Object Point Cloud")
    log.info("- Gray wireframe sphere: Sampling sphere for pre-poses")
    log.info("- Gray points: Map/Environment")
    log.info("- RGB axes: R=X(red), G=Y(green), B=Z(blue)")

    log.info(
        f"Pre-pose position: [{prepose_world[0,3]:.3f}, {prepose_world[1,3]:.3f}, {prepose_world[2,3]:.3f}]"
    )

    # Run visualization
    log.info("Close the visualization window to continue...")
    vis.run()
    vis.destroy_window()


def visualize_segmentation(rgb_image, mask):
    """
    Overlays a segmentation mask on an RGB image and displays it.
    Args:
        rgb_image (np.ndarray): The RGB image.
        mask (np.ndarray): The binary segmentation mask.
    """
    overlay = rgb_image.copy()
    overlay[mask > 0] = [0, 255, 0]  # Green for segmented area

    # Blend the original image and the overlay
    blended = cv2.addWeighted(rgb_image, 0.7, overlay, 0.3, 0)

    cv2.imshow("Segmentation", blended)
    cv2.waitKey(1000)  # Display for 1 sec
    cv2.destroyWindow("Segmentation")


def visualize_place_pipeline(
    place_pose_world,
    combined_points=None,
    placement_pcd_world=None,
    is_candidate=False,
    sample_info="",
):
    """
    Visualize place-pose, placement surface, and map in Open3D.
    Args:
        place_pose_world: 4x4 place-pose in world frame
        combined_points: Nx3 array of map points (optional)
        placement_pcd_world: o3d.geometry.PointCloud of the placement surface (optional)
        is_candidate: bool, True if the pose is just a candidate
        sample_info: str, information about the sample number
    """
    import open3d as o3d

    title = "Place-Pose Visualization"
    if is_candidate:
        title = f"Candidate Place-Pose | {sample_info}"
        log.info(f"=== Visualizing Candidate Place-Pose ({sample_info}) ===")
    else:
        log.info("=== Creating Final Place-Pose Visualization ===")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1200, height=800)

    # Add coordinate frame for world origin
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.3, origin=[0, 0, 0]
    )
    vis.add_geometry(world_frame)

    # Add place-pose coordinate frame
    place_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.15, origin=[0, 0, 0]
    )
    place_frame.transform(place_pose_world)
    vis.add_geometry(place_frame)

    # Add place-pose position marker (sphere)
    place_pose_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
    place_pose_color = (
        [1, 1, 0] if is_candidate else [0, 1, 0]
    )  # Yellow for candidate, Green for final
    place_pose_sphere.paint_uniform_color(place_pose_color)
    place_pose_sphere.translate(place_pose_world[:3, 3])
    vis.add_geometry(place_pose_sphere)

    # Add the placement surface point cloud
    if placement_pcd_world is not None:
        placement_pcd_world.paint_uniform_color([0, 1, 1])  # Cyan
        vis.add_geometry(placement_pcd_world)
        log.info("Added placement point cloud to visualization.")

    # Add map point cloud if available
    if combined_points is not None:
        map_pcd = o3d.geometry.PointCloud()
        map_pcd.points = o3d.utility.Vector3dVector(combined_points)
        map_pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Gray for map
        vis.add_geometry(map_pcd)
        log.info(f"Added {len(combined_points)} map points to visualization")

    # Log pose information
    legend_pose_name = "Candidate Place-pose" if is_candidate else "Final Place-pose"
    log.info("=== Visualization Legend ===")
    log.info("- Large RGB frame: World origin")
    log.info(
        f"- Small RGB frame + {'Yellow' if is_candidate else 'Green'} sphere: {legend_pose_name}"
    )
    log.info("- Cyan points: Placement Surface Point Cloud")
    log.info("- Gray points: Map/Environment")
    log.info("- RGB axes: R=X(red), G=Y(green), B=Z(blue)")

    log.info(
        f"Place-pose position: [{place_pose_world[0,3]:.3f}, {place_pose_world[1,3]:.3f}, {place_pose_world[2,3]:.3f}]"
    )

    # Run visualization
    log.info("Close the visualization window to continue...")
    vis.run()
    vis.destroy_window()


def visualize_grasps(pred_grasps_cam, scores, rgb, depth, K):
    """
    Visualize grasp poses in 3D with the full point cloud, in the camera frame.
    This implementation now follows the reference style from visualize_grasp.py.
    """
    from grasp_anywhere.utils.perception_utils import depth2pc

    # 1. Reconstruct the full point cloud from depth data in the camera frame
    cam_K = np.array(K).reshape(3, 3)
    pc_full, pc_colors = depth2pc(depth, cam_K, rgb=rgb, voxel_size=0.0)
    pcd_full = o3d.geometry.PointCloud()
    pcd_full.points = o3d.utility.Vector3dVector(pc_full)
    if pc_colors is not None:
        pcd_full.colors = o3d.utility.Vector3dVector(pc_colors / 255.0)

    # 2. Create line-based gripper visualizations for each grasp in the camera frame
    gripper_geometries = []
    top_grasp_idx = np.argmax(scores)

    for i, grasp_cam in enumerate(pred_grasps_cam):
        grasp_mat_cam = np.array(grasp_cam).reshape(4, 4)

        # Define gripper geometry from the reference code
        width = 0.12
        depth_g = 0.1
        points = [
            [0, 0, 0],
            [width / 2, 0, 0],
            [-width / 2, 0, 0],
            [0, 0, -0.06],
            [width / 2, 0, depth_g],
            [-width / 2, 0, depth_g],
        ]
        lines = [[0, 1], [0, 2], [0, 3], [1, 4], [2, 5]]

        # Transform points from local gripper frame to camera frame
        points_homog = np.concatenate([np.array(points), np.ones((6, 1))], axis=1)
        transformed_points_cam = np.dot(grasp_mat_cam, points_homog.T).T[:, :3]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(transformed_points_cam)
        line_set.lines = o3d.utility.Vector2iVector(lines)

        # Color the gripper lines
        color = [0, 0, 1] if i == top_grasp_idx else [1, 0, 0]
        line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])

        gripper_geometries.append(line_set)

    # 3. Visualize the full scene in the camera frame
    o3d.visualization.draw_geometries(
        [pcd_full] + gripper_geometries,
        window_name="Grasp Visualization (Camera Frame)",
    )


def visualize_grasp_pose_world(grasp_pose_world, rgb, depth, K, camera_pose):
    """
    Visualizes the target end-effector grasp pose in the world frame, along with the
    object point cloud and camera.
    """
    import open3d as o3d

    from grasp_anywhere.utils.perception_utils import extract_point_clouds

    # 1. Get object point cloud in camera frame
    cam_K = np.array(K).reshape(3, 3)
    pcd_full, _, pc_colors = extract_point_clouds(
        depth,
        cam_K,
        rgb=rgb,
        z_range=[0.2, 1.5],  # Reasonable range for objects
    )

    if pcd_full is None or len(pcd_full) == 0:
        print("Warning: No points in point cloud from mask. Cannot visualize.")
        return

    pcd_cam = o3d.geometry.PointCloud()
    pcd_cam.points = o3d.utility.Vector3dVector(pcd_full)
    if pc_colors is not None:
        pcd_cam.colors = o3d.utility.Vector3dVector(pc_colors / 255.0)

    # 2. Transform point cloud to world frame
    pcd_world = pcd_cam.transform(camera_pose)

    # 3. Create geometries
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.2, origin=[0, 0, 0]
    )
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    camera_frame.transform(camera_pose)
    grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
    grasp_frame.transform(grasp_pose_world)

    # 4. Visualize
    o3d.visualization.draw_geometries(
        [pcd_world, world_frame, camera_frame, grasp_frame],
        window_name="End-Effector Target Pose (World Frame)",
        width=1024,
        height=768,
    )


def visualize_grasps_pcd(
    pred_grasps, scores, points, colors=None, window_name="Grasp Visualization"
):
    """
    Visualize grasp poses in 3D with a given point cloud.
    Args:
        pred_grasps: (N, 4, 4) numpy array of grasp poses
        scores: (N,) numpy array of scores
        points: (M, 3) numpy array of point cloud points
        colors: (M, 3) numpy array of point cloud colors (optional)
        window_name: title of the window
    """
    # 1. Create the point cloud geometry
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        colors = np.asarray(colors)
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # 2. Create line-based gripper visualizations for each grasp
    gripper_geometries = []
    top_grasp_idx = np.argmax(scores) if len(scores) > 0 else -1

    for i, grasp in enumerate(pred_grasps):
        grasp_mat = np.array(grasp).reshape(4, 4)

        # Define gripper geometry (standard dimensions)
        width = 0.12
        depth_g = 0.1
        pts = [
            [0, 0, 0],
            [width / 2, 0, 0],
            [-width / 2, 0, 0],
            [0, 0, -0.06],
            [width / 2, 0, depth_g],
            [-width / 2, 0, depth_g],
        ]
        lines = [[0, 1], [0, 2], [0, 3], [1, 4], [2, 5]]

        # Transform points to grasp frame
        points_homog = np.concatenate([np.array(pts), np.ones((6, 1))], axis=1)
        transformed_points = np.dot(grasp_mat, points_homog.T).T[:, :3]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(transformed_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)

        # Color: Blue for top grasp, Red for others
        color = [0, 0, 1] if i == top_grasp_idx else [1, 0, 0]
        line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])

        gripper_geometries.append(line_set)

    # Visualize the world frame
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.2, origin=[0, 0, 0]
    )

    # 3. Visualize
    o3d.visualization.draw_geometries(
        [pcd] + gripper_geometries + [world_frame],
        window_name=window_name,
    )


# ========================
# Pyrender live viewer API
# ========================

_pyrender_viewer_instance = None


class PyrenderLiveViewer:
    """
    Lightweight Pyrender-based live viewer for point cloud visualization.
    Runs in a background thread via pyrender.Viewer(run_in_thread=True).
    """

    def __init__(self):
        if pyrender is None:
            raise RuntimeError("Pyrender is not installed.")
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0])
        # Add a world axis if trimesh is available
        if trimesh is not None:
            try:
                axis_tm = trimesh.creation.axis(origin_size=0.02, axis_length=0.2)
                axis_mesh = pyrender.Mesh.from_trimesh(axis_tm)
                self.scene.add(axis_mesh)
            except Exception:
                pass
        self.viewer = pyrender.Viewer(
            self.scene,
            run_in_thread=True,
            use_raymond_lighting=True,
            viewport_size=(1024, 768),
        )
        self.live_node = None
        self.static_node = None

    def _get_lock(self):
        try:
            return getattr(self.viewer, "render_lock", None)
        except Exception:
            return None

    def set_static_pointcloud(self, points):
        pts = np.asarray(points, dtype=np.float32)
        if pts.size == 0 or pts.shape[1] != 3:
            return
        colors = np.tile(
            np.array([[0.7, 0.7, 0.7]], dtype=np.float32), (pts.shape[0], 1)
        )
        mesh = pyrender.Mesh.from_points(pts, colors=colors)
        lock = self._get_lock()
        if lock is not None:
            with lock:
                if self.static_node is not None:
                    try:
                        self.scene.remove_node(self.static_node)
                    except Exception:
                        pass
                self.static_node = self.scene.add(mesh)
        else:
            if self.static_node is not None:
                try:
                    self.scene.remove_node(self.static_node)
                except Exception:
                    pass
            self.static_node = self.scene.add(mesh)

    def update_live_pointcloud(self, points):
        pts = np.asarray(points, dtype=np.float32)
        if pts.size == 0 or pts.shape[1] != 3:
            return
        colors = np.tile(
            np.array([[1.0, 0.0, 1.0]], dtype=np.float32), (pts.shape[0], 1)
        )
        mesh = pyrender.Mesh.from_points(pts, colors=colors)
        lock = self._get_lock()
        if lock is not None:
            with lock:
                if self.live_node is not None:
                    try:
                        self.scene.remove_node(self.live_node)
                    except Exception:
                        pass
                self.live_node = self.scene.add(mesh)
        else:
            if self.live_node is not None:
                try:
                    self.scene.remove_node(self.live_node)
                except Exception:
                    pass
            self.live_node = self.scene.add(mesh)

    def close(self):
        try:
            # pyrender.Viewer does not expose an explicit close API; rely on GC/window close
            pass
        except Exception:
            pass


def init_pyrender_viewer():
    """
    Initialize a global Pyrender live viewer instance, if Pyrender is available.
    Returns True on success, False otherwise.
    """
    global _pyrender_viewer_instance
    if _pyrender_viewer_instance is not None:
        return True
    if pyrender is None:
        log.warning("Pyrender not available. Skipping live viewer init.")
        return False
    try:
        _pyrender_viewer_instance = PyrenderLiveViewer()
        log.info("Pyrender live viewer initialized.")
        return True
    except Exception as e:
        log.warning(f"Failed to initialize Pyrender live viewer: {e}")
        _pyrender_viewer_instance = None
        return False


def pyrender_set_static_pointcloud(points):
    """Set or update the static map point cloud in the live viewer."""
    if _pyrender_viewer_instance is None:
        return False
    try:
        _pyrender_viewer_instance.set_static_pointcloud(points)
        return True
    except Exception as e:
        log.debug(f"pyrender_set_static_pointcloud failed: {e}")
        return False


def pyrender_update_pointcloud(points):
    """Update the live point cloud geometry in the viewer."""
    if _pyrender_viewer_instance is None:
        return False
    try:
        _pyrender_viewer_instance.update_live_pointcloud(points)
        return True
    except Exception as e:
        log.debug(f"pyrender_update_pointcloud failed: {e}")
        return False


def pyrender_is_available():
    """Return True if a live viewer instance is active."""
    return _pyrender_viewer_instance is not None


def pyrender_close():
    """Close the viewer if needed."""
    global _pyrender_viewer_instance
    if _pyrender_viewer_instance is None:
        return
    try:
        _pyrender_viewer_instance.close()
    finally:
        _pyrender_viewer_instance = None


def show_costmap(costmap):
    """
    Save a 2D costmap visualization to /tmp as a PNG using OpenCV colormap.
    Assumes cost values in [0,1]. Overwrites the same file each call.
    """
    import cv2
    import numpy as np

    cm = np.nan_to_num(costmap, nan=1.0, posinf=1.0, neginf=0.0)
    img = (np.clip(cm, 0.0, 1.0) * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
    out_path = "debug/ga_costmap.png"
    try:
        cv2.imwrite(out_path, colored)
    except Exception:
        pass


def visualize_pcd(points, rgb=None, depth=None, K=None):
    """
    Visualizes a point cloud.
    Args:
        points: (N, 3) numpy array of points
        rgb: (H, W, 3) RGB image
        depth: (H, W) Depth image (optional)
        K: (3, 3) Camera intrinsic matrix (optional)
    """
    import open3d as o3d

    if points is None or len(points) == 0:
        log.warning("visualize_pcd called with empty points.")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    o3d.visualization.draw_geometries(
        [pcd, coord_frame], window_name="Point Cloud Visualization"
    )
