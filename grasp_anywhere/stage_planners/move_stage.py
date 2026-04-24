import time

import numpy as np

from grasp_anywhere.utils.logger import log
from grasp_anywhere.utils.motion_utils import move_to_config_with_replanning
from grasp_anywhere.utils.perception_utils import depth2pc


class MovePlanner:
    """
    Handles planning and execution of robot base movements to predefined locations.
    """

    def __init__(self, robot, logger=None):
        self.robot = robot
        # Predefined locations for the robot to move to.
        # These are represented as [x, y, theta] poses for the robot's base.
        # TODO: Make this extendable by loading from a config file.
        self.locations = {
            "table": {
                "pose": [-1.019, 1.649, 3.072],
                "head": [0.0, 0.4],  # [pan, tilt],
                "joints": [0.3, 1.32, 1.4, -0.2, 1.72, 0.0, 1.66, 0.0],
            },
            "kitchen counter": {
                "pose": [-1.149, -1.182, -1.807],
                "head": [0.0, 0.4],  # [pan, tilt],
                "joints": [0.3, 1.32, 1.4, -0.2, 1.72, 0.0, 1.66, 0.0],
            },
            "table manipulation": {
                "pose": [-1.52, 1.7, -3.14],
                "head": [-0.05744338035583496, 0.8858116885131836],  # [pan, tilt]
                "joints": [
                    0.35,
                    -1.4870163272827148,
                    -0.6402787641784669,
                    1.0788297245040894,
                    1.567980115637207,
                    2.3214089101593016,
                    -0.9990689204467773,
                    3.0158881247613527,
                ],
            },
        }
        log.info(
            f"MovePlanner initialized with locations: {list(self.locations.keys())}"
        )

    def run_reset_arm_inplace(self, static_collision_points, **kwargs):
        """
        Resets the robot's arm, torso, and head to a predefined configuration.
        If the robot is at the table manipulation pose, it uses that configuration.
        Otherwise, it uses a default configuration.
        Supports dynamic collision avoidance.
        """
        depth = kwargs.get("depth")
        camera_intrinsic = kwargs.get("camera_intrinsic")
        camera_extrinsic = kwargs.get("camera_extrinsic")

        use_dynamic_collision = all(
            param is not None for param in [depth, camera_intrinsic, camera_extrinsic]
        )

        # Check if current base pose matches table manipulation pose
        current_base = self.robot.get_base_params()  # [x, y, theta]
        table_manip_pose = self.locations["table manipulation"]["pose"]

        # Position and angle thresholds
        POS_THRESHOLD = 0.5  # 15 cm tolerance
        ANGLE_THRESHOLD = 1.0  # ~11 degrees tolerance

        pos_dist = np.linalg.norm(
            np.array(current_base[:2]) - np.array(table_manip_pose[:2])
        )
        angle_diff = abs(current_base[2] - table_manip_pose[2])
        angle_dist = min(angle_diff, 2 * np.pi - angle_diff)

        at_table_manip_pose = pos_dist < POS_THRESHOLD and angle_dist < ANGLE_THRESHOLD

        if use_dynamic_collision:
            if at_table_manip_pose:
                log.info(
                    "Robot is at table manipulation pose. Resetting arm, torso, "
                    "and head in place with dynamic collision avoidance..."
                )
            else:
                log.info(
                    "Resetting arm, torso, and head in place to "
                    "default configuration with dynamic collision avoidance..."
                )

            depth_map = np.array(depth)
            camera_intrinsic_arr = np.array(camera_intrinsic)
            camera_extrinsic_arr = np.array(camera_extrinsic)

            try:
                observed_pcd_cam, _ = depth2pc(depth_map, camera_intrinsic_arr)
                log.info(f"Generated {len(observed_pcd_cam)} points from depth map")

                if len(observed_pcd_cam) > 0:
                    observed_pcd_cam_h = np.hstack(
                        (observed_pcd_cam, np.ones((observed_pcd_cam.shape[0], 1)))
                    )
                    observed_pcd_world = (
                        camera_extrinsic_arr @ observed_pcd_cam_h.T
                    ).T[:, :3]
                    log.info("Transformed observed pointcloud to world frame")

                    filtered_observed_pcd, _ = self.robot.filter_points_on_robot(
                        observed_pcd_world
                    )
                    # filter out the points on the ground
                    filtered_observed_pcd = filtered_observed_pcd[
                        filtered_observed_pcd[:, 2] > 0.3
                    ]
                    log.info(
                        f"Filtered observed pointcloud, kept {len(filtered_observed_pcd)} points"
                    )

                    combined_pcd = np.vstack(
                        (static_collision_points, filtered_observed_pcd)
                    )
                    log.info(
                        f"Combined static ({len(static_collision_points)}) and "
                        "observed ({len(filtered_observed_pcd)}) pointclouds"
                    )
                else:
                    combined_pcd = static_collision_points
                    log.warning(
                        "No valid points in observed pointcloud, using only static collision data"
                    )

                self.robot.clear_pointclouds()
                self.robot.add_pointcloud(combined_pcd, point_radius=0.03)
                log.info("Updated collision environment for motion planning")
            except Exception as e:
                log.error(f"Error processing dynamic collision data: {e}")
                log.warning("Falling back to static collision points only")
                self.robot.clear_pointclouds()
                self.robot.add_pointcloud(static_collision_points, point_radius=0.03)
        else:
            if at_table_manip_pose:
                log.info(
                    "Robot is at table manipulation pose."
                    "Resetting arm, torso, and head in place with static collision avoidance..."
                )
            else:
                log.info("Resetting arm, torso, and head in place")

        try:
            if at_table_manip_pose:
                target_joints = self.locations["table manipulation"]["joints"]
                target_head = self.locations["table manipulation"]["head"]
                success_msg = "In-place arm reset successful. Robot configured for table manipulation."
            else:
                target_joints = self.locations["table"]["joints"]
                target_head = self.locations["table"]["head"]
                success_msg = (
                    "In-place arm reset successful. Robot at default configuration."
                )

            log.info("Moving arm to target configuration with collision avoidance...")
            if self.robot.send_joint_values(target_joints, duration=5.0) is None:
                self.robot.clear_pointclouds()
                self.robot.add_pointcloud(static_collision_points)
                return (
                    False,
                    "Failed to move arm joints to target configuration with collision avoidance.",
                )

            log.info(f"Moving head to target pose: {target_head}")
            self.robot.move_head(target_head[0], target_head[1])

            self.robot.clear_pointclouds()
            self.robot.add_pointcloud(static_collision_points)
            log.info(
                "Restored original collision environment (static pointclouds only)"
            )

            return True, success_msg
        except Exception as e:
            log.error(f"Error during in-place reset: {e}")
            try:
                self.robot.clear_pointclouds()
                self.robot.add_pointcloud(static_collision_points)
                log.info("Restored original collision environment after error")
            except Exception as e:
                log.error(f"Failed to restore collision environment after error: {e}")
            return False, f"In-place reset failed: {str(e)}"

    def run(
        self,
        location_name,
        collision_points,
        enable_replanning=True,
        enable_pcd_alignment=False,
    ):
        """
        Plans and executes a move to a named location.

        Args:
            location_name (str): The name of the location to move to.
            collision_points (np.ndarray): The point cloud of the environment for collision checking.
            enable_replanning (bool): Whether to enable replanning during motion execution.
            enable_pcd_alignment (bool): Whether to enable point cloud alignment during replanning.

        Returns:
            tuple: (success, message) where success is a boolean and message is a string.
        """
        if location_name not in self.locations:
            available_locations = list(self.locations.keys())
            msg = f"Unknown location '{location_name}'. Available locations: {available_locations}"
            log.info(f"ERROR: {msg}")
            return False, msg

        target_base_pose = self.locations[location_name]["pose"]
        head_pose = self.locations[location_name]["head"]
        log.info(f"Planning move to location '{location_name}' at {target_base_pose}")

        # 1. Check if we're already close enough to the target
        current_base = self.robot.get_base_params()
        pos_dist = np.linalg.norm(
            np.array(current_base[:2]) - np.array(target_base_pose[:2])
        )
        angle_diff = abs(current_base[2] - target_base_pose[2])
        angle_dist = min(angle_diff, 2 * np.pi - angle_diff)

        POS_THRESHOLD = 0.1  # 10 cm
        ANGLE_THRESHOLD = 0.1

        if pos_dist < POS_THRESHOLD and angle_dist < ANGLE_THRESHOLD:
            log.info("Already at target location. Moving joints to target config.")

            self.robot.clear_pointclouds()
            self.robot.add_pointcloud(collision_points, filter_robot=True)

            goal_joints = self.locations[location_name]["joints"]
            if self.robot.send_joint_values(goal_joints, duration=5.0) is None:
                msg = "Failed to move joints to config."
                log.info(f"WARNING: {msg}")
                return False, msg

            log.info(f"Moving head to pan={head_pose[0]}, tilt={head_pose[1]}")
            self.robot.move_head(pan=head_pose[0], tilt=head_pose[1])
            time.sleep(0.5)  # Give time for head to move

            return True, "Already at location, moved joints to target config."

        # 3. Execute the move
        log.info(f"Executing move to base pose: {target_base_pose} with replanning")

        goal_joints = self.locations[location_name]["joints"]

        success, msg = move_to_config_with_replanning(
            robot=self.robot,
            goal_joints=goal_joints,
            goal_base=target_base_pose,
            enable_replanning=enable_replanning,
            enable_pcd_alignment=enable_pcd_alignment,
        )
        if not success:
            return success, msg

        log.info(f"Moving head to pan={head_pose[0]}, tilt={head_pose[1]}")
        self.robot.move_head(pan=head_pose[0], tilt=head_pose[1])
        time.sleep(0.5)  # Give time for head to move

        return success, msg

    def run_observe_object(
        self,
        object_coord,
        collision_points,
        manipulation_radius=1.5,
        enable_replanning=True,
        enable_pcd_alignment=False,
    ):
        """
        Plans and executes a move to observe an object from an optimal position.

        The method finds a position on a circle around the object that:
        - Looks forward to the object
        - Is relatively close to the current robot pose
        - Has a good cost value from the 2D SDF costmap

        Args:
            object_coord (np.ndarray or list): 3D world coordinate of the object (x, y, z).
            collision_points (np.ndarray): The point cloud of the environment for collision checking.
            manipulation_radius (float): Radius around the object to search for observation positions.
            enable_replanning (bool): Whether to enable replanning during motion execution.
            enable_pcd_alignment (bool): Whether to enable point cloud alignment during replanning.

        Returns:
            tuple: (success, message) where success is a boolean and message is a string.
        """
        object_coord = np.asarray(object_coord, dtype=np.float32).flatten()
        if len(object_coord) != 3:
            msg = f"Invalid object coordinate: {object_coord}. Expected 3D coordinate."
            log.error(msg)
            return False, msg

        log.info(f"Planning observation position for object at {object_coord}")

        # Update the robot's costmap if dynamic costmap is enabled
        if self.robot.base_sampler.dynamic_costmap and collision_points is not None:
            log.info("Updating costmap from current scene point cloud...")
            self.robot.base_sampler.update_from_pointcloud(collision_points)
            if (
                self.robot.enable_visualization
                and self.robot.base_sampler.costmap is not None
            ):
                from grasp_anywhere.utils.visualization_utils import show_costmap

                show_costmap(self.robot.base_sampler.costmap)

        # Get current robot base position
        current_base = self.robot.get_base_params()  # [x, y, theta]
        current_pos_2d = np.array(current_base[:2])

        # Find the best observation position
        best_pose = self._find_best_observation_pose(
            object_coord, current_pos_2d, manipulation_radius
        )

        if best_pose is None:
            msg = "Failed to find a valid observation position on the costmap."
            log.error(msg)
            return False, msg

        target_base_pose = list(best_pose)  # [x, y, theta]
        log.info(f"Selected observation pose: {target_base_pose}")

        # Check if we're already close enough to the target
        pos_dist = np.linalg.norm(current_pos_2d - np.array(target_base_pose[:2]))
        angle_diff = abs(current_base[2] - target_base_pose[2])
        angle_dist = min(angle_diff, 2 * np.pi - angle_diff)

        POS_THRESHOLD = 0.15  # 15 cm
        ANGLE_THRESHOLD = 0.2  # ~11 degrees

        if pos_dist < POS_THRESHOLD and angle_dist < ANGLE_THRESHOLD:
            log.info("Already at a good observation position. Just adjusting head.")
            self.robot.point_head_at(object_coord.tolist())
            time.sleep(0.5)
            return True, "Already at good observation position."

        # Set up collision environment
        self.robot.clear_pointclouds()
        self.robot.add_pointcloud(collision_points, filter_robot=True)

        # Use a default neutral arm configuration for observation
        goal_joints = [0.3, 1.32, 1.4, -0.2, 1.72, 0.0, 1.66, 0.0]

        # Execute the move with replanning
        log.info(f"Executing move to observation pose: {target_base_pose}")
        success, msg = move_to_config_with_replanning(
            robot=self.robot,
            goal_joints=goal_joints,
            goal_base=target_base_pose,
            enable_replanning=enable_replanning,
            enable_pcd_alignment=enable_pcd_alignment,
        )

        if not success:
            return success, msg

        # Point head at the object
        log.info(f"Pointing head at object: {object_coord}")
        self.robot.point_head_at(object_coord.tolist())
        time.sleep(0.5)

        return True, f"Successfully moved to observation position at {target_base_pose}"

    def _find_best_observation_pose(
        self, object_coord, current_pos_2d, manipulation_radius
    ):
        """
        Finds the best observation pose on a circle around the object.

        Balances between:
        - Low costmap cost (good navigation position)
        - Close to current robot position
        - Within manipulation radius of the object

        Args:
            object_coord (np.ndarray): 3D world coordinate of the object.
            current_pos_2d (np.ndarray): Current 2D position of the robot base [x, y].
            manipulation_radius (float): Radius around the object to search.

        Returns:
            tuple or None: (x, y, theta) for the best observation pose, or None if not found.
        """
        base_sampler = self.robot.base_sampler

        if base_sampler.costmap is None:
            log.error("No costmap available for observation pose planning.")
            return None

        object_pos_2d = object_coord[:2]
        height = base_sampler.metadata["height"]
        width = base_sampler.metadata["width"]

        # Create a grid of world coordinates
        grid_y, grid_x = np.mgrid[0:height, 0:width]
        world_x = (
            base_sampler.metadata["origin_x"]
            + grid_x * base_sampler.metadata["resolution"]
        )
        world_y = (
            base_sampler.metadata["origin_y"]
            + grid_y * base_sampler.metadata["resolution"]
        )

        # Calculate distances to object and current position
        dist_to_object = np.sqrt(
            (world_x - object_pos_2d[0]) ** 2 + (world_y - object_pos_2d[1]) ** 2
        )
        dist_to_current = np.sqrt(
            (world_x - current_pos_2d[0]) ** 2 + (world_y - current_pos_2d[1]) ** 2
        )

        # Create masks for valid positions
        # Must be within manipulation radius of object
        radius_mask = dist_to_object <= manipulation_radius

        # Must be at least some minimum distance from object (e.g., 0.5m)
        min_dist_mask = dist_to_object >= 0.5

        # Combine masks
        valid_mask = radius_mask & min_dist_mask

        # Get costs from the costmap (lower is better)
        costs = np.asarray(base_sampler.costmap, dtype=float).copy()
        costs[~np.isfinite(costs)] = 1.0
        costs = np.clip(costs, 0.0, 1.0)

        # Filter out high-cost (occupied) cells
        costs[costs >= 0.95] = np.inf
        costs[~valid_mask] = np.inf

        if np.all(np.isinf(costs)):
            log.warning("No valid cells found in costmap for observation.")
            return None

        # Normalize distance to current position for scoring
        max_dist_to_current = (
            np.max(dist_to_current[valid_mask]) if np.any(valid_mask) else 1.0
        )
        max_dist_to_current = max(max_dist_to_current, 0.1)  # Avoid division by zero
        normalized_dist_to_current = dist_to_current / max_dist_to_current

        # Combined score: balance costmap cost and distance to current position
        # Lower score is better
        # Weight: 60% costmap cost, 40% distance to current position
        combined_score = 0.6 * costs + 0.4 * normalized_dist_to_current
        combined_score[~valid_mask] = np.inf

        # Find the best position
        if np.all(np.isinf(combined_score)):
            log.warning("No valid observation position found with combined score.")
            return None

        best_idx = np.unravel_index(np.argmin(combined_score), combined_score.shape)
        best_grid_y, best_grid_x = best_idx

        # Convert to world coordinates
        best_x = world_x[best_grid_y, best_grid_x]
        best_y = world_y[best_grid_y, best_grid_x]

        # Calculate yaw to face the object
        direction_vector = object_pos_2d - np.array([best_x, best_y])
        yaw = np.arctan2(direction_vector[1], direction_vector[0])

        log.info(
            f"Best observation pose: ({best_x:.3f}, {best_y:.3f}, {yaw:.3f}) "
            f"with cost={costs[best_grid_y, best_grid_x]:.3f}, "
            f"dist_to_current={dist_to_current[best_grid_y, best_grid_x]:.3f}"
        )

        return (best_x, best_y, yaw)
