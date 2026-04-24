from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class RobotEnv(ABC):
    """
    Abstract base class for a robot environment.
    This interface abstracts away the specifics of how the robot is controlled
    (e.g., ROS for a real robot, or a simulator's API).

    Canonical ordering (used throughout this project):
    - Base configs (waypoints) are [x, y, theta]
    - Body action channel is [head_pan, head_tilt, torso_lift]
    - Action channel concatenation order is [arm, gripper, body, base]
    """

    @abstractmethod
    def get_joint_states(self) -> Optional[Tuple[List[str], List[float]]]:
        """Returns the names and positions of all joints."""
        raise NotImplementedError

    @abstractmethod
    def get_rgb(self) -> Optional[np.ndarray]:
        """Returns the latest RGB image from the camera."""
        raise NotImplementedError

    @abstractmethod
    def get_depth(self) -> Optional[np.ndarray]:
        """Returns the latest depth image from the camera."""
        raise NotImplementedError

    @abstractmethod
    def get_camera_intrinsics(self) -> Optional[np.ndarray]:
        """Returns the camera intrinsic matrix."""
        raise NotImplementedError

    @abstractmethod
    def get_base_pose(
        self, world_frame: str = "map", robot_base_frame: str = "base_link"
    ) -> Optional[Tuple[float, float, float]]:
        """Returns the current base pose as (x, y, theta)."""
        raise NotImplementedError

    @abstractmethod
    def get_camera_pose(
        self, camera_frame: str = "head_camera_rgb_optical_frame"
    ) -> Optional[np.ndarray]:
        """Get the current pose of a specified camera in the world frame.

        The default frame name mirrors the real robot environment so callers
        may omit the argument across both real and simulation backends.
        """
        raise NotImplementedError

    @abstractmethod
    def execute_whole_body_motion(self, arm_path: List, base_configs: List) -> bool:
        """Executes a whole-body motion plan."""
        raise NotImplementedError

    @abstractmethod
    def start_whole_body_motion(self, arm_path: List, base_configs: List) -> bool:
        """Starts a whole-body motion plan execution without blocking."""
        raise NotImplementedError

    @abstractmethod
    def stop_whole_body_motion(self):
        """Stops the current whole-body motion."""
        raise NotImplementedError

    @abstractmethod
    def is_motion_done(self) -> bool:
        """Checks if a whole-body motion is currently executing."""
        raise NotImplementedError

    @abstractmethod
    def get_motion_result(self) -> bool:
        """Returns the success status of the last motion."""
        raise NotImplementedError

    @abstractmethod
    def get_arm_action_state(self) -> Any:
        """Gets the state of the arm action client."""
        raise NotImplementedError

    @abstractmethod
    def cancel_arm_goals(self):
        """Cancels all goals for the arm action client."""
        raise NotImplementedError

    @abstractmethod
    def cancel_torso_goals(self):
        """Cancels all goals for the torso action client."""
        raise NotImplementedError

    @abstractmethod
    def cancel_head_goals(self):
        """Cancels all pending head movement goals."""
        raise NotImplementedError

    @abstractmethod
    def get_arm_action_result(self) -> Any:
        """Gets the result of the arm action client."""
        raise NotImplementedError

    @abstractmethod
    def send_joint_values(self, target_joints: List[float], duration: float) -> Any:
        """Move the arm and torso to specified joint positions."""
        raise NotImplementedError

    @abstractmethod
    def execute_joint_trajectory(self, trajectory_points: List, duration: float) -> Any:
        """Execute a joint trajectory using the arm and torso controllers."""
        raise NotImplementedError

    @abstractmethod
    def start_joint_trajectory_async(
        self, trajectory_points: List, duration: float
    ) -> bool:
        """Starts an arm trajectory execution without blocking."""
        raise NotImplementedError

    @abstractmethod
    def move_base(self, linear_x: float, angular_z: float):
        """Move the robot base with specified linear and angular velocities."""
        raise NotImplementedError

    @abstractmethod
    def stop_base(self):
        """Stop all base movement."""
        raise NotImplementedError

    @abstractmethod
    def navigate_to(self, position: List[float], orientation: List[float]) -> Any:
        """Send the robot base to a target position in the map frame."""
        raise NotImplementedError

    @abstractmethod
    def set_torso_height(self, height: float, duration: float) -> Any:
        """Set the torso height to a specific value."""
        raise NotImplementedError

    @abstractmethod
    def control_gripper(self, position: float, max_effort: float) -> Any:
        """Control the gripper position."""
        raise NotImplementedError

    @abstractmethod
    def get_gripper_status(self) -> Dict[str, Any]:
        """Get the current gripper status and position."""
        raise NotImplementedError

    @abstractmethod
    def move_head(self, pan: float, tilt: float, duration: float):
        """Moves the robot's head to a given pan and tilt position."""
        raise NotImplementedError

    def get_sensor_snapshot(self) -> Optional[Dict[str, Any]]:
        """
        Get depth, intrinsics, and joint states from the same timestep.

        For simulators, this should read all data atomically from a single observation.
        Camera pose can be computed from joint states using FK.

        Returns:
            Dict with keys: 'depth', 'intrinsics', 'joint_states'
            where joint_states is (joint_names, joint_positions)
            or None if data is unavailable
        """
        # Default implementation: separate calls (acceptable for real robot)
        rgb = self.get_rgb()
        depth = self.get_depth()
        intrinsics = self.get_camera_intrinsics()
        joint_states = self.get_joint_states()

        if depth is None or intrinsics is None or joint_states is None:
            return None

        return {
            "rgb": rgb,
            "depth": depth,
            "intrinsics": intrinsics,
            "joint_states": joint_states,  # (names, positions)
        }
