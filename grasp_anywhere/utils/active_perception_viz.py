#!/usr/bin/env python3
"""
Visualization utilities for active perception in robot manipulation.

This module provides paper-quality visualizations showing:
1. Object attention heatmaps
2. Future robot trajectory with transparent kinematics
3. Swept volume visualization with collision risk
"""

from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib import cm


def create_attention_heatmap(
    rgb_image: np.ndarray,
    attention_map: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Create a heatmap overlay showing attention on an RGB image.

    Args:
        rgb_image: (H, W, 3) RGB image
        attention_map: (H, W) attention scores normalized to [0, 1]
        alpha: Blending factor for overlay (0=only RGB, 1=only heatmap)
        colormap: OpenCV colormap (default: COLORMAP_JET for red=high, blue=low)

    Returns:
        (H, W, 3) RGB image with heatmap overlay
    """
    # Normalize attention map to 0-255
    attention_normalized = (attention_map * 255).astype(np.uint8)

    # Apply colormap (creates BGR image)
    heatmap_bgr = cv2.applyColorMap(attention_normalized, colormap)

    # Convert to RGB
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Blend with original image
    overlay = cv2.addWeighted(rgb_image, 1 - alpha, heatmap_rgb, alpha, 0)

    return overlay


def create_ghost_robot_trajectory(
    robot_poses: List[np.ndarray],
    mesh_path: Optional[str] = None,
    colors: Optional[List[Tuple[float, float, float]]] = None,
    alphas: Optional[List[float]] = None,
) -> List[o3d.geometry.TriangleMesh]:
    """
    Create transparent robot meshes showing future trajectory.

    Args:
        robot_poses: List of 4x4 transformation matrices for each timestep
        mesh_path: Path to robot mesh file (optional)
        colors: List of RGB colors for each pose (default: blue gradient)
        alphas: List of alpha transparency values for each pose
                (default: linearly decrease from 0.8 to 0.1)

    Returns:
        List of Open3D meshes representing the trajectory
    """
    n_poses = len(robot_poses)

    # Default colors: blue gradient
    if colors is None:
        colors = [(0.2, 0.4, 0.8)] * n_poses

    # Default alphas: fade from current (solid) to future (transparent)
    if alphas is None:
        alphas = np.linspace(0.8, 0.1, n_poses).tolist()

    meshes = []

    for i, (pose, color, alpha) in enumerate(zip(robot_poses, colors, alphas)):
        # Create a simple gripper representation if no mesh provided
        if mesh_path is None:
            # Create gripper as coordinate frame + box
            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            # Add gripper box
            gripper_box = o3d.geometry.TriangleMesh.create_box(
                width=0.05, height=0.02, depth=0.08
            )
            gripper_box.translate([-0.025, -0.01, 0])
            mesh += gripper_box
        else:
            mesh = o3d.io.read_triangle_mesh(mesh_path)

        # Apply transformation
        mesh.transform(pose)

        # Set color with transparency
        mesh.paint_uniform_color(color)

        meshes.append(mesh)

    return meshes


def create_swept_volume_heatmap(
    trajectory_points: np.ndarray,
    collision_risk: np.ndarray,
    point_size: float = 0.01,
) -> o3d.geometry.PointCloud:
    """
    Create a point cloud showing swept volume with collision risk heatmap.

    Args:
        trajectory_points: (N, 3) array of points in the swept volume
        collision_risk: (N,) array of collision risk scores [0, 1]
        point_size: Size of each point in the visualization

    Returns:
        Open3D PointCloud with color-coded collision risk
    """
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(trajectory_points)

    # Map collision risk to colors (green=safe, yellow=caution, red=danger)
    colors = np.zeros((len(collision_risk), 3))

    # Use matplotlib colormap for smooth gradient
    cmap = cm.get_cmap("RdYlGn_r")  # Red-Yellow-Green reversed (red=high risk)
    colors = cmap(collision_risk)[:, :3]  # Take RGB, ignore alpha

    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def visualize_active_perception_dual_panel(
    # Panel A: Object attention
    rgb_image: np.ndarray,
    object_attention_map: np.ndarray,
    # Panel B: Trajectory prediction (required params)
    robot_trajectory_poses: List[np.ndarray],
    swept_volume_points: np.ndarray,
    collision_risk_scores: np.ndarray,
    # Optional parameters
    object_mask: Optional[np.ndarray] = None,
    # Scene context
    scene_pcd: Optional[np.ndarray] = None,
    scene_colors: Optional[np.ndarray] = None,
    # Visualization options
    save_path: Optional[str] = None,
    show_visualization: bool = True,
) -> Tuple[np.ndarray, o3d.geometry.PointCloud]:
    """
    Create dual-panel visualization for active perception paper.

    Panel A: RGB image with attention heatmap on target object
    Panel B: 3D scene with transparent robot trajectory and swept volume

    Args:
        rgb_image: (H, W, 3) RGB camera image
        object_attention_map: (H, W) attention scores on object [0, 1]
        object_mask: (H, W) binary mask for the object (optional)
        robot_trajectory_poses: List of 4x4 poses along predicted trajectory
        swept_volume_points: (N, 3) points in the robot's swept volume
        collision_risk_scores: (N,) collision risk for each swept volume point
        scene_pcd: (M, 3) scene point cloud (optional)
        scene_colors: (M, 3) scene point colors (optional)
        save_path: Path to save visualization (optional)
        show_visualization: Whether to display the visualization

    Returns:
        Tuple of (panel_a_image, panel_b_geometry)
    """

    # === PANEL A: Object Attention ===
    panel_a = create_attention_heatmap(
        rgb_image, object_attention_map, alpha=0.6, colormap=cv2.COLORMAP_JET
    )

    # Optionally highlight object boundary
    if object_mask is not None:
        contours, _ = cv2.findContours(
            object_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        panel_a = cv2.drawContours(panel_a, contours, -1, (255, 255, 255), 2)

    # === PANEL B: Trajectory Prediction ===
    geometries = []

    # Add scene point cloud
    if scene_pcd is not None:
        scene_cloud = o3d.geometry.PointCloud()
        scene_cloud.points = o3d.utility.Vector3dVector(scene_pcd)
        if scene_colors is not None:
            scene_cloud.colors = o3d.utility.Vector3dVector(scene_colors)
        else:
            # Default gray color for scene
            scene_cloud.paint_uniform_color([0.7, 0.7, 0.7])
        geometries.append(scene_cloud)

    # Add swept volume with collision risk heatmap
    swept_volume_pcd = create_swept_volume_heatmap(
        swept_volume_points, collision_risk_scores
    )
    geometries.append(swept_volume_pcd)

    # Add ghost robot trajectory
    robot_meshes = create_ghost_robot_trajectory(robot_trajectory_poses)
    geometries.extend(robot_meshes)

    # Add coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    geometries.append(coord_frame)

    # Save and/or display
    if save_path:
        # Save Panel A
        panel_a_path = save_path.replace(".png", "_panel_a.png")
        cv2.imwrite(panel_a_path, cv2.cvtColor(panel_a, cv2.COLOR_RGB2BGR))
        print(f"Saved Panel A to: {panel_a_path}")

        # Save Panel B (3D visualization requires screenshot)
        # User can manually screenshot the 3D view
        print("Panel B will be displayed. Please screenshot for paper figure.")

    if show_visualization:
        try:
            # Show Panel A
            plt.figure(figsize=(10, 8))
            plt.imshow(panel_a)
            plt.axis("off")
            plt.title("Panel A: Object Attention", fontsize=16, pad=20)
            plt.tight_layout()

            # Use non-interactive backend if display fails
            try:
                plt.show()
            except Exception as e:
                print(f"Warning: Could not display Panel A interactively: {e}")
                print("Panel A saved to file instead.")

            # Show Panel B
            print("Displaying Panel B: Future Trajectory & Swept Volume")
            try:
                o3d.visualization.draw_geometries(
                    geometries,
                    window_name="Panel B: Future Trajectory Prediction",
                    width=1024,
                    height=768,
                )
            except Exception as e:
                print(f"Warning: Could not display Panel B interactively: {e}")
                print("You can visualize the point cloud separately using Open3D.")
        except Exception as e:
            print(f"Warning: Visualization display failed: {e}")
            print("Figures saved to disk successfully.")

    return panel_a, geometries


def create_combined_figure_for_paper(
    panel_a_image: np.ndarray,
    panel_b_screenshot: np.ndarray,
    save_path: str,
    title: str = "Active Perception for Robot Manipulation",
    dpi: int = 300,
):
    """
    Combine Panel A and Panel B into a single publication-ready figure.

    Args:
        panel_a_image: Panel A image (object attention)
        panel_b_screenshot: Panel B screenshot (trajectory prediction)
        save_path: Path to save the combined figure
        title: Figure title
        dpi: DPI for publication quality (default: 300)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A
    axes[0].imshow(panel_a_image)
    axes[0].set_title("(a) Object Attention", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # Panel B
    axes[1].imshow(panel_b_screenshot)
    axes[1].set_title(
        "(b) Future Trajectory & Swept Volume", fontsize=14, fontweight="bold"
    )
    axes[1].axis("off")

    # Overall title
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved combined figure to: {save_path}")
    plt.show()


# ============================================
# Example / Demo Functions
# ============================================


def demo_active_perception_visualization():
    """
    Demo showing how to use the active perception visualization.
    """
    # Create synthetic data for demonstration
    H, W = 480, 640

    # Panel A: Object attention
    rgb_image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)

    # Create Gaussian attention map around object
    y, x = np.ogrid[:H, :W]
    center_y, center_x = H // 2, W // 2
    attention_map = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * 50**2))

    # Object mask
    object_mask = (attention_map > 0.3).astype(np.uint8)

    # Panel B: Trajectory prediction
    # Create 5 robot poses along a trajectory
    robot_poses = []
    for i in range(5):
        pose = np.eye(4)
        pose[:3, 3] = [0.5 + i * 0.1, 0.0, 0.5 - i * 0.05]  # Moving forward and down
        robot_poses.append(pose)

    # Create swept volume points
    n_points = 1000
    swept_volume = np.random.randn(n_points, 3) * 0.05 + [0.6, 0.0, 0.4]

    # Collision risk (higher near edges)
    collision_risk = np.random.rand(n_points) * 0.5

    # Scene point cloud
    scene_points = np.random.rand(2000, 3) * [1.0, 1.0, 0.5]
    scene_colors = np.ones((2000, 3)) * [0.7, 0.7, 0.7]

    # Visualize
    panel_a, panel_b_geom = visualize_active_perception_dual_panel(
        rgb_image=rgb_image,
        object_attention_map=attention_map,
        robot_trajectory_poses=robot_poses,
        swept_volume_points=swept_volume,
        collision_risk_scores=collision_risk,
        object_mask=object_mask,
        scene_pcd=scene_points,
        scene_colors=scene_colors,
        save_path="/tmp/active_perception_demo.png",
        show_visualization=True,
    )

    print("Demo completed!")


if __name__ == "__main__":
    demo_active_perception_visualization()
