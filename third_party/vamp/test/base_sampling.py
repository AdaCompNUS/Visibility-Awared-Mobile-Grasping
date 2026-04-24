import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow
import argparse
import math


class Pose:
    """Simple Pose class to hold position and orientation data."""

    def __init__(self):
        self.position = Position()
        self.orientation = Orientation()


class Position:
    """Simple Position class to hold x, y, z coordinates."""

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


class Orientation:
    """Simple Orientation class to hold quaternion values."""

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.w = 1.0


def quaternion_to_yaw(orientation):
    """
    Convert quaternion to yaw angle (rotation around z-axis).

    Args:
        orientation: Orientation object with quaternion values

    Returns:
        yaw: Yaw angle in radians
    """
    # Extract quaternion components
    x, y, z, w = orientation.x, orientation.y, orientation.z, orientation.w

    # Calculate yaw (rotation around z-axis)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return yaw


def load_costmap(costmap_path):
    """
    Load a previously generated costmap from file.

    Args:
        costmap_path: Path to the costmap .npz file

    Returns:
        costmap: 2D numpy array representing the costmap
        metadata: Dictionary with costmap metadata
    """
    print(f"Loading costmap from {costmap_path}")

    try:
        data = np.load(costmap_path)

        # Extract costmap and metadata
        costmap = data["costmap"]

        # Create metadata dictionary
        metadata = {
            "resolution": float(data["resolution"]),
            "origin_x": float(data["origin_x"]),
            "origin_y": float(data["origin_y"]),
            "width": int(data["width"]),
            "height": int(data["height"]),
            "max_distance": float(data["max_distance"]),
        }

        # Check if valid_area_mask exists in the data
        if "valid_area_mask" in data:
            metadata["valid_area_mask"] = data["valid_area_mask"]

        print(
            f"Loaded costmap with dimensions {metadata['width']}x{metadata['height']}"
        )
        return costmap, metadata

    except Exception as e:
        print(f"Error loading costmap: {e}")
        return None, None


def world_to_grid(world_x, world_y, metadata):
    """
    Convert world coordinates to grid coordinates.

    Args:
        world_x, world_y: World coordinates
        metadata: Costmap metadata

    Returns:
        grid_x, grid_y: Grid coordinates
    """
    grid_x = int((world_x - metadata["origin_x"]) / metadata["resolution"])
    grid_y = int((world_y - metadata["origin_y"]) / metadata["resolution"])
    return grid_x, grid_y


def grid_to_world(grid_x, grid_y, metadata):
    """
    Convert grid coordinates to world coordinates.

    Args:
        grid_x, grid_y: Grid coordinates
        metadata: Costmap metadata

    Returns:
        world_x, world_y: World coordinates
    """
    world_x = metadata["origin_x"] + grid_x * metadata["resolution"]
    world_y = metadata["origin_y"] + grid_y * metadata["resolution"]
    return world_x, world_y


def find_valid_base_positions(
    costmap, metadata, ee_pose, manipulation_radius, cost_threshold=0.3
):
    """
    Find valid base positions for the robot by intersecting the costmap with a circle
    representing the manipulation range around the end effector.

    Args:
        costmap: 2D costmap array (values 0-1, NaN outside boundary)
        metadata: Costmap metadata dictionary
        ee_pose: End effector pose
        manipulation_radius: Radius of the manipulation range in meters
        cost_threshold: Minimum cost value to consider a cell valid (0-1)

    Returns:
        valid_positions: List of valid base positions in world coordinates with costs and orientation scores
        circle_mask: 2D boolean array showing the circular region
        valid_mask: 2D boolean array showing the valid base positions
    """
    # Convert end effector position to grid coordinates
    ee_grid_x, ee_grid_y = world_to_grid(
        ee_pose.position.x, ee_pose.position.y, metadata
    )

    # Create 2D grid coordinates
    y_grid, x_grid = np.mgrid[0 : metadata["height"], 0 : metadata["width"]]

    # Calculate distances from each grid cell to end effector position
    distances = (
        np.sqrt((x_grid - ee_grid_x) ** 2 + (y_grid - ee_grid_y) ** 2)
        * metadata["resolution"]
    )

    # Create circle mask (cells within manipulation range)
    circle_mask = distances <= manipulation_radius

    # Create valid cells mask (cells with cost >= threshold and not NaN)
    valid_cells = (costmap >= cost_threshold) & ~np.isnan(costmap)

    # Combine circle mask and valid cells mask to get final valid positions
    valid_mask = circle_mask & valid_cells

    # Convert valid grid positions to world coordinates
    valid_positions = []
    for y in range(metadata["height"]):
        for x in range(metadata["width"]):
            if valid_mask[y, x]:
                world_x, world_y = grid_to_world(x, y, metadata)

                # Calculate orientation towards the end effector
                dx = ee_pose.position.x - world_x
                dy = ee_pose.position.y - world_y
                theta = math.atan2(dy, dx)

                # Get the cost value (lower is better for sampling)
                cost_value = costmap[y, x]

                valid_positions.append((world_x, world_y, cost_value, theta))

    return valid_positions, circle_mask, valid_mask


def probabilistic_sample_base_positions(valid_positions, num_samples=5, temp=0.1):
    """
    Sample base positions from the valid positions using a probabilistic approach.
    Favors positions with lower cost values.

    Args:
        valid_positions: List of valid base positions (world_x, world_y, cost, theta)
        num_samples: Number of positions to sample
        temp: Temperature parameter for softmax (lower means more deterministic selection)

    Returns:
        sampled_positions: Sampled base positions with orientation (world_x, world_y, theta)
    """
    if not valid_positions:
        print("No valid base positions to sample from")
        return []

    # If fewer valid positions than requested samples, return all positions
    if len(valid_positions) <= num_samples:
        return [(x, y, theta) for x, y, _, theta in valid_positions]

    # Extract costs and convert to probabilities (lower cost = higher probability)
    costs = np.array([cost for _, _, cost, _ in valid_positions])

    # Invert costs so lower costs have higher probability
    inverted_costs = 1.0 / (costs + 1e-6)  # Add small epsilon to avoid division by zero

    # Apply softmax with temperature to get sampling probabilities
    # Lower temperature makes sampling more deterministic (favoring lowest costs)
    exp_costs = np.exp(inverted_costs / temp)
    probabilities = exp_costs / np.sum(exp_costs)

    # Sample positions with the calculated probabilities
    selected_indices = np.random.choice(
        len(valid_positions), size=num_samples, replace=False, p=probabilities
    )

    # Return sampled positions with their orientation
    sampled_positions = [
        (valid_positions[i][0], valid_positions[i][1], valid_positions[i][3])
        for i in selected_indices
    ]

    return sampled_positions


def visualize_base_sampling(
    costmap,
    metadata,
    ee_pose,
    manipulation_radius,
    valid_positions,
    sampled_positions,
    output_path=None,
):
    """
    Visualize the base position sampling results with robot orientations.

    Args:
        costmap: 2D costmap array
        metadata: Costmap metadata
        ee_pose: End effector pose
        manipulation_radius: Radius of the manipulation range
        valid_positions: List of valid base positions
        sampled_positions: List of sampled base positions with orientation
        output_path: Path to save the visualization (optional)
    """
    plt.figure(figsize=(12, 10))

    # Plot the costmap
    cmap = plt.cm.viridis_r
    cmap.set_bad(alpha=0)  # Make NaN values transparent
    plt.imshow(
        costmap,
        extent=[
            metadata["origin_x"],
            metadata["origin_x"] + metadata["width"] * metadata["resolution"],
            metadata["origin_y"],
            metadata["origin_y"] + metadata["height"] * metadata["resolution"],
        ],
        origin="lower",
        cmap=cmap,
        alpha=0.7,
    )

    # Plot the end effector position
    plt.plot(
        ee_pose.position.x,
        ee_pose.position.y,
        "r*",
        markersize=15,
        label="End Effector",
    )

    # Plot the manipulation range circle
    circle = Circle(
        (ee_pose.position.x, ee_pose.position.y),
        manipulation_radius,
        fill=False,
        color="r",
        linestyle="--",
        linewidth=2,
        label="Manipulation Range",
    )
    plt.gca().add_patch(circle)

    # Plot all valid positions
    if valid_positions:
        valid_positions_arr = np.array([(x, y) for x, y, _, _ in valid_positions])
        plt.scatter(
            valid_positions_arr[:, 0],
            valid_positions_arr[:, 1],
            c="blue",
            s=5,
            alpha=0.3,
            label="Valid Base Positions",
        )

    # Plot sampled positions with orientation arrows
    if sampled_positions:
        arrow_length = 0.3  # Length of the orientation arrow

        for i, (x, y, theta) in enumerate(sampled_positions):
            # Calculate arrow endpoint
            dx = arrow_length * math.cos(theta)
            dy = arrow_length * math.sin(theta)

            # Create and add arrow to plot
            arrow = Arrow(x, y, dx, dy, width=0.2, color="green")
            plt.gca().add_patch(arrow)

            # Add position number
            plt.text(
                x,
                y,
                f"{i+1}",
                color="white",
                fontsize=12,
                ha="center",
                va="center",
                weight="bold",
                bbox=dict(facecolor="green", alpha=0.7, boxstyle="circle"),
            )

    plt.colorbar(label="Costmap Value")
    plt.title("Robot Base Position Sampling for Whole-Body IK")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.legend(loc="upper right")
    plt.axis("equal")
    plt.grid(True)

    # Add small legend for the arrows
    plt.plot([], [], "g->", markersize=15, label="Robot Base Orientation")
    plt.legend(loc="upper right")

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Sample base positions for whole-body IK"
    )
    parser.add_argument(
        "--costmap", type=str, required=True, help="Path to the costmap NPZ file"
    )
    parser.add_argument(
        "--radius", type=float, default=1.0, help="Manipulation radius in meters"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Cost threshold for valid positions (0-1)",
    )
    parser.add_argument(
        "--samples", type=int, default=5, help="Number of base positions to sample"
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.1,
        help="Temperature parameter for softmax sampling (lower = more deterministic)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="base_sampling.png",
        help="Output path for visualization",
    )

    args = parser.parse_args()

    # Load the costmap
    costmap, metadata = load_costmap(args.costmap)
    if costmap is None:
        print("Failed to load costmap. Exiting.")
        return

    # Set desired pose (this would come from input in a real application)
    pose = Pose()
    pose.position.x = -3.66
    pose.position.y = -3.23
    pose.position.z = 1.27
    pose.orientation.x = 0.427
    pose.orientation.y = -0.726
    pose.orientation.z = -0.476
    pose.orientation.w = 0.254

    print(
        f"End effector pose: ({pose.position.x}, {pose.position.y}, {pose.position.z})"
    )

    # Find valid base positions
    valid_positions, circle_mask, valid_mask = find_valid_base_positions(
        costmap, metadata, pose, args.radius, args.threshold
    )

    print(f"Found {len(valid_positions)} valid base positions")

    # Sample base positions
    sampled_positions = probabilistic_sample_base_positions(
        valid_positions, args.samples, args.temp
    )

    print(f"Sampled {len(sampled_positions)} base positions")

    # Visualize results
    visualize_base_sampling(
        costmap,
        metadata,
        pose,
        args.radius,
        valid_positions,
        sampled_positions,
        args.output,
    )

    # Save sampled positions to a file
    np.save("sampled_base_positions.npy", np.array(sampled_positions))
    print("Sampled base positions saved to sampled_base_positions.npy")


if __name__ == "__main__":
    main()
