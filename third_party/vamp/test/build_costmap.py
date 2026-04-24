import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os
import glob
from scipy.ndimage import distance_transform_edt
from shapely.geometry import MultiPoint
from shapely import Point
import alphashape


def load_pointcloud(pcd_path):
    """Load a point cloud from file using Open3D."""
    print(f"Loading point cloud from {pcd_path}...")
    try:
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)
        print(f"Loaded {len(points)} points from {os.path.basename(pcd_path)}")
        return points
    except Exception as e:
        print(f"Error loading point cloud {pcd_path}: {e}")
        return None


def load_all_pointclouds(directory):
    """
    Load all point cloud files from a directory.
    Args:
        directory: Directory containing point cloud files
    Returns:
        Combined point cloud as a numpy array
    """
    print(f"Searching for point cloud files in {directory}...")
    # Find all point cloud files (PLY, PCD, etc.)
    ply_files = glob.glob(os.path.join(directory, "*.ply"))
    pcd_files = glob.glob(os.path.join(directory, "*.pcd"))
    all_files = ply_files + pcd_files
    print(f"Found {len(all_files)} point cloud files")

    if not all_files:
        print(f"No point cloud files found in {directory}")
        return None

    # Load and combine all point clouds
    combined_points = []
    total_points = 0
    for file_path in all_files:
        pc = load_pointcloud(file_path)
        if pc is not None and len(pc) > 0:
            combined_points.append(pc)
            total_points += len(pc)

    if not combined_points:
        print("No valid point clouds loaded")
        return None

    # Combine all point clouds into a single array
    combined_pc = np.vstack(combined_points)
    print(
        f"Combined {len(combined_points)} point clouds with total {len(combined_pc)} points"
    )
    return combined_pc


def create_concave_hull(points_2d, alpha=0.3):
    """
    Create a concave hull (alpha shape) around 2D points.

    Args:
        points_2d: Nx2 array of 2D points
        alpha: Alpha value for the alpha shape (lower = more detailed boundary)

    Returns:
        Polygon object representing the concave hull
    """
    try:
        # Use alphashape to create concave hull
        if len(points_2d) < 4:  # Not enough points for concave hull
            return MultiPoint(points_2d).convex_hull

        alpha_shape = alphashape.alphashape(points_2d, alpha)

        # If alpha shape creation failed, fall back to convex hull
        if alpha_shape.is_empty:
            return MultiPoint(points_2d).convex_hull

        return alpha_shape
    except Exception as e:
        print(f"Error creating concave hull: {e}")
        # Fall back to convex hull if there's any error
        return MultiPoint(points_2d).convex_hull


def points_within_boundary(grid_coords, boundary_polygon, grid_to_world_transform):
    """
    Check which grid coordinates are within the boundary polygon.

    Args:
        grid_coords: Nx2 array of grid coordinates
        boundary_polygon: Shapely polygon in world coordinates
        grid_to_world_transform: Function to transform grid to world coordinates

    Returns:
        Boolean mask of which coordinates are inside the boundary
    """
    # Transform grid coordinates to world coordinates
    world_coords = np.array([grid_to_world_transform(x, y) for x, y in grid_coords])

    # Check if points are within the polygon
    mask = np.array([boundary_polygon.contains(Point(x, y)) for x, y in world_coords])
    return mask


def create_2d_costmap(
    points_3d,
    resolution=0.05,
    max_height=2.0,
    min_height=0.0,
    max_distance=5.0,
    alpha=0.3,
    buffer_distance=1.0,
):
    """
    Convert 3D pointcloud to 2D ESDF costmap with automatic boundary detection.

    Args:
        points_3d: Nx3 numpy array of 3D points
        resolution: Grid cell size in meters
        max_height: Maximum height to consider for obstacles
        min_height: Minimum height to consider for obstacles
        max_distance: Maximum distance to compute in the ESDF (meters)
        alpha: Alpha value for concave hull (lower = more detailed boundary)
        buffer_distance: Distance to buffer the boundary (meters)

    Returns:
        2D costmap as numpy array with values 0-1 (NaN outside boundary)
        grid_info: Dictionary with grid metadata
    """
    # Import shapely Point here to avoid circular import issues
    from shapely.geometry import Point

    # Filter points by height
    mask = (points_3d[:, 2] >= min_height) & (points_3d[:, 2] <= max_height)
    filtered_points = points_3d[mask]

    if len(filtered_points) == 0:
        print("No points within the specified height range")
        return None, None

    # Extract 2D points (x, y coordinates)
    points_2d = filtered_points[:, 0:2]

    # Create boundary using concave hull (alpha shape)
    boundary = create_concave_hull(points_2d, alpha)

    # Buffer the boundary to include some margin
    buffered_boundary = boundary.buffer(buffer_distance)

    # Determine grid dimensions from the buffered boundary
    minx, miny, maxx, maxy = buffered_boundary.bounds

    # Calculate grid dimensions
    grid_width = int(np.ceil((maxx - minx) / resolution))
    grid_height = int(np.ceil((maxy - miny) / resolution))

    print(
        f"Creating costmap with dimensions {grid_width}x{grid_height} at {resolution}m resolution"
    )
    print(f"World dimensions: {maxx-minx:.2f}m x {maxy-miny:.2f}m")

    # Initialize mask for the valid area (points inside the boundary)
    # and occupancy grid for obstacles
    valid_area_mask = np.zeros((grid_height, grid_width), dtype=bool)
    occupancy_grid = np.zeros((grid_height, grid_width), dtype=bool)

    # Create grid coordinates meshgrid
    y_grid, x_grid = np.mgrid[0:grid_height, 0:grid_width]
    np.column_stack((x_grid.ravel(), y_grid.ravel()))

    # Define grid to world coordinate transformation
    def grid_to_world(grid_x, grid_y):
        world_x = minx + grid_x * resolution
        world_y = miny + grid_y * resolution
        return world_x, world_y

    # Define world to grid coordinate transformation
    def world_to_grid(world_x, world_y):
        grid_x = int((world_x - minx) / resolution)
        grid_y = int((world_y - miny) / resolution)
        return grid_x, grid_y

    # Mark valid area based on boundary
    for i in range(grid_height):
        for j in range(grid_width):
            world_x, world_y = grid_to_world(j, i)
            point = Point(world_x, world_y)
            if buffered_boundary.contains(point):
                valid_area_mask[i, j] = True

    # Mark occupied cells
    for point in filtered_points:
        x, y = point[0:2]
        grid_x, grid_y = world_to_grid(x, y)

        # Ensure within bounds
        if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
            occupancy_grid[grid_y, grid_x] = True

    # Compute distance transform only within the valid area
    # Use masked arrays to ignore areas outside the boundary
    masked_occupancy = np.ma.array(~occupancy_grid, mask=~valid_area_mask)

    # Compute distance transform
    distance_grid = distance_transform_edt(masked_occupancy.filled(True)) * resolution

    # Normalize distances to [0, 1] range with exponential decay
    # Places close to obstacles will be close to 1, far away will be close to 0
    normalized_distance = np.exp(-distance_grid / max_distance)

    # Set occupied cells to exactly 1
    normalized_distance[occupancy_grid] = 1.0

    # Set areas outside the valid region to NaN (no value)
    final_costmap = np.full_like(normalized_distance, np.nan, dtype=float)
    final_costmap[valid_area_mask] = normalized_distance[valid_area_mask]

    # Grid info for reference
    grid_info = {
        "resolution": resolution,
        "origin_x": minx,
        "origin_y": miny,
        "width": grid_width,
        "height": grid_height,
        "max_distance": max_distance,
        "valid_area_mask": valid_area_mask,
        "boundary": buffered_boundary,
    }

    return final_costmap, grid_info


def save_costmap(costmap, grid_info, output_path):
    """
    Save the costmap as a colorful image and as a NumPy file.

    Args:
        costmap: 2D numpy array with values from 0 to 1 (and NaN outside boundary)
        grid_info: Grid metadata
        output_path: Base path for the output files (without extension)
    """
    # Save as colorful image
    plt.figure(figsize=(10, 10))

    # Use a reversed colormap to make higher values deeper/darker
    # and lower values lighter
    cmap = plt.cm.viridis_r  # Add _r to reverse the colormap
    cmap.set_bad(alpha=0)  # Make NaN values transparent

    plt.imshow(costmap, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    plt.colorbar(label="Cost (0-1)")
    plt.title(f'ESDF Costmap (Resolution: {grid_info["resolution"]}m)')

    # Add grid info as text
    info_text = (
        f"Resolution: {grid_info['resolution']}m\n"
        f"Size: {grid_info['width']}x{grid_info['height']} cells\n"
        f"Origin: ({grid_info['origin_x']:.2f}, {grid_info['origin_y']:.2f})\n"
        f"Max Distance: {grid_info['max_distance']}m"
    )
    plt.figtext(0.02, 0.02, info_text, bbox=dict(facecolor="white", alpha=0.8))

    # Save as PNG
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Save as NumPy file with metadata
    np.savez(
        f"{output_path}.npz",
        costmap=costmap,
        resolution=grid_info["resolution"],
        origin_x=grid_info["origin_x"],
        origin_y=grid_info["origin_y"],
        width=grid_info["width"],
        height=grid_info["height"],
        max_distance=grid_info["max_distance"],
        valid_area_mask=grid_info["valid_area_mask"],
    )

    print(f"Costmap saved to {output_path}.png and {output_path}.npz")


def main():
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert 3D pointcloud to 2D ESDF costmap"
    )
    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="Directory containing point cloud files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="costmap",
        help="Output path for the costmap (without extension)",
    )
    parser.add_argument(
        "--resolution", type=float, default=0.01, help="Grid resolution in meters"
    )
    parser.add_argument(
        "--max_height",
        type=float,
        default=2.0,
        help="Maximum height to consider for obstacles",
    )
    parser.add_argument(
        "--min_height",
        type=float,
        default=0.0,
        help="Minimum height to consider for obstacles",
    )
    parser.add_argument(
        "--max_distance",
        type=float,
        default=0.5,
        help="Maximum distance to compute in the ESDF (meters)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Alpha value for concave hull (lower = more detailed boundary)",
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=0.1,
        help="Distance to buffer the boundary (meters)",
    )

    args = parser.parse_args()

    # Load point cloud using the reference code structure
    points_3d = load_all_pointclouds(args.directory)

    if points_3d is None or len(points_3d) == 0:
        print("No valid points loaded. Exiting.")
        return

    # Create costmap
    costmap, grid_info = create_2d_costmap(
        points_3d,
        resolution=args.resolution,
        max_height=args.max_height,
        min_height=args.min_height,
        max_distance=args.max_distance,
        alpha=args.alpha,
        buffer_distance=args.buffer,
    )

    if costmap is not None:
        # Save costmap
        save_costmap(costmap, grid_info, args.output)

        # Display costmap
        plt.figure(figsize=(10, 10))

        # Use a reversed colormap with transparency for NaN values
        cmap = plt.cm.viridis_r  # Add _r to reverse the colormap
        cmap.set_bad(alpha=0)  # Make NaN values transparent

        plt.imshow(costmap, cmap=cmap, vmin=0, vmax=1)
        plt.colorbar(label="Cost (0-1)")
        plt.title("ESDF Costmap with Natural Boundary")
        plt.show()


if __name__ == "__main__":
    main()
