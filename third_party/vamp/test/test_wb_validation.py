import vamp
import json
import numpy as np
import open3d as o3d
from pathlib import Path
from vamp import pybullet_interface as vpb
import pybullet as pb

def main():
    # Paths
    root = Path(__file__).parent.parent
    debug_dir = root / "debug_pcd"
    
    # Load Data
    with open(debug_dir / "target_collision.json") as f:
        data = json.load(f)
    
    pcd = o3d.io.read_point_cloud(str(debug_dir / "target_collision.ply"))
    points = np.asarray(pcd.points)

    # VAMP Validation
    robot = vamp.fetch
    env = vamp.Environment()
    # Add points with default radius settings
    env.add_pointcloud(points.tolist(), *vamp.ROBOT_RADII_RANGES["fetch"], 0.03)
    
    # Check validity
    is_valid = robot.validate_whole_body_config(data["goal_joints"], data["goal_base"], env)
    print(f"Configuration Valid: {is_valid}")

    # Visualization
    urdf_path = root / "resources/fetch/fetch_spherized.urdf"
    sim = vpb.PyBulletSimulator(str(urdf_path), vamp.ROBOT_JOINTS["fetch"], True)
    
    # Draw subsampled points and robot
    sim.draw_pointcloud(points.tolist())
    
    sim.client.resetBasePositionAndOrientation(
        sim.skel_id, 
        [data["goal_base"][0], data["goal_base"][1], 0], 
        pb.getQuaternionFromEuler([0, 0, data["goal_base"][2]])
    )
    sim.set_joint_positions(data["goal_joints"])

    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
