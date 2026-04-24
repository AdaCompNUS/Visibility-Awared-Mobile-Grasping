import struct
from time import time

import actionlib
import cv2
import numpy as np
import rospy
import tf2_ros
from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
    GripperCommandAction,
    GripperCommandGoal,
)
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import (
    CameraInfo,
    CompressedImage,
    JointState,
)
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from grasp_anywhere.envs.base import RobotEnv
from grasp_anywhere.utils.logger import log


class FetchRealEnv(RobotEnv):
    def __init__(self, planning_joint_names):
        """Initialize the Fetch robot environment."""
        self.planning_joint_names = planning_joint_names
        try:
            rospy.init_node("fetch_controller", anonymous=True)
        except rospy.exceptions.ROSException:
            print("Node has already been initialized, do nothing")

        # Add joint states subscriber
        self.joint_states = None
        self.joint_state_subscriber = rospy.Subscriber(
            "/joint_states", JointState, self._joint_states_callback
        )

        # For camera data
        self.bridge = CvBridge()
        self.latest_rgb = None
        self.latest_depth = None
        self.camera_intrinsics = None
        # Log what topic we're subscribing to
        # Note: Using compressed topic directly since throttled version isn't publishing
        rgb_topic = "/throttled_camera/rgb/compressed"
        log.info(f"Subscribing to RGB topic: {rgb_topic}")
        self.rgb_sub = rospy.Subscriber(
            rgb_topic, CompressedImage, self._rgb_callback, queue_size=1
        )
        self.depth_sub = rospy.Subscriber(
            "/throttled_camera/depth/compressedDepth",
            CompressedImage,
            self._depth_callback,
            queue_size=1,
        )
        self.info_sub = rospy.Subscriber(
            "/head_camera/rgb/camera_info",
            CameraInfo,
            self._info_callback,
            queue_size=1,
        )

        # Wait for initial messages
        log.info("Waiting for joint states and camera info...")
        while (
            self.joint_states is None
            or self.camera_intrinsics is None
            or self.latest_depth is None
        ) and not rospy.is_shutdown():
            rospy.sleep(0.1)
        log.info("Initial joint states and camera info received.")

        # Publisher for base movement commands
        self._base_publisher = rospy.Publisher(
            "/base_controller/command", Twist, queue_size=2
        )

        # Initialize action clients
        self.arm_traj_client = actionlib.SimpleActionClient(
            "arm_controller/follow_joint_trajectory", FollowJointTrajectoryAction
        )
        self.arm_traj_client.wait_for_server()

        self.gripper_client = actionlib.SimpleActionClient(
            "gripper_controller/gripper_action", GripperCommandAction
        )
        self.gripper_client.wait_for_server()

        self.move_base_client = actionlib.SimpleActionClient(
            "move_base", MoveBaseAction
        )
        self.move_base_client.wait_for_server()

        # End effector pose publisher
        self.ee_pose_publisher = rospy.Publisher(
            "/arm_controller/cartesian_pose_vel_controller/command",
            PoseStamped,
            queue_size=10,
        )

        # Initialize torso action client
        self.torso_client = actionlib.SimpleActionClient(
            "torso_controller/follow_joint_trajectory", FollowJointTrajectoryAction
        )
        self.torso_client.wait_for_server()

        # Add TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Head Action Client
        self.head_traj_client = actionlib.SimpleActionClient(
            "head_controller/follow_joint_trajectory", FollowJointTrajectoryAction
        )
        log.info("Waiting for head controller action server...")
        self.head_traj_client.wait_for_server()
        log.info("Head controller action server found.")
        self.head_joint_names = ["head_pan_joint", "head_tilt_joint"]

        # Publisher for whole body trajectory execution
        self.arm_path_pub = rospy.Publisher(
            "/fetch_whole_body_controller/arm_path", JointTrajectory, queue_size=1
        )
        self.base_path_pub = rospy.Publisher(
            "/fetch_whole_body_controller/base_path", JointTrajectory, queue_size=1
        )

        # Publisher to stop the whole body controller
        self.stop_pub = rospy.Publisher(
            "/fetch_whole_body_controller/stop", Bool, queue_size=1
        )
        rospy.sleep(0.1)  # Allow time for publisher to connect

        # Subscriber for whole body execution completion
        self.execution_finished = False
        self.execution_finished_sub = rospy.Subscriber(
            "/fetch_whole_body_controller/execution_finished",
            Bool,
            self._execution_finished_callback,
        )
        self.motion_state = "IDLE"  # Can be IDLE, RUNNING, SUCCEEDED, FAILED

    def get_joint_states(self):
        if self.joint_states:
            return self.joint_states.name, self.joint_states.position
        return None

    def get_rgb(self):
        rospy.sleep(0.1)
        return self.latest_rgb

    def get_depth(self):
        rospy.sleep(0.1)
        return self.latest_depth

    def get_camera_intrinsics(self):
        return self.camera_intrinsics

    def get_base_pose(self, world_frame="map", robot_base_frame="base_link"):
        transform = self.tf_buffer.lookup_transform(
            world_frame, robot_base_frame, rospy.Time(0), rospy.Duration(1.0)
        )
        trans = transform.transform.translation
        rot = transform.transform.rotation
        quaternion = [rot.x, rot.y, rot.z, rot.w]
        yaw = R.from_quat(quaternion).as_euler("xyz", degrees=False)[2]
        return (trans.x, trans.y, yaw)

    def get_camera_pose(
        self,
        camera_frame="head_camera_rgb_optical_frame",
        world_frame="map",
    ):
        log.info(f"Attempting to get pose of '{camera_frame}' in frame '{world_frame}'")
        transform_stamped = self.tf_buffer.lookup_transform(
            world_frame,  # Target frame
            camera_frame,  # Source frame
            rospy.Time(),  # Get the latest transform
            rospy.Duration(1.0),  # Wait for up to 1 second
        )
        translation = transform_stamped.transform.translation
        rotation = transform_stamped.transform.rotation
        quaternion = [rotation.x, rotation.y, rotation.z, rotation.w]
        rotation_matrix = R.from_quat(quaternion).as_matrix()
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[0, 3] = translation.x
        transform_matrix[1, 3] = translation.y
        transform_matrix[2, 3] = translation.z
        log.info("Successfully retrieved camera pose as transformation matrix.")
        return transform_matrix

    def execute_whole_body_motion(self, arm_path, base_configs):
        if not self.start_whole_body_motion(arm_path, base_configs):
            return False

        # Wait for execution to finish with a timeout
        time_per_waypoint = 0.15  # seconds
        estimated_duration = len(arm_path) * time_per_waypoint
        timeout = 2.0 * estimated_duration
        start_time = time()

        while not self.is_motion_done() and not rospy.is_shutdown():
            if time() - start_time > timeout:
                log.warning(
                    f"Motion execution timed out after {timeout:.2f}s! Stopping motion."
                )
                self.stop_whole_body_motion()
                return False
            rospy.sleep(0.1)

        if not self.get_motion_result():
            log.warning("Motion execution did not finish successfully.")
            return False

        log.info("Whole body motion execution completed successfully.")
        return True

    def start_whole_body_motion(self, arm_path, base_configs):
        # Validate input
        assert (
            arm_path is not None and base_configs is not None
        ), "Cannot execute motion: arm_path or base_configs is None"

        # Reset completion flag before starting
        self.motion_state = "RUNNING"

        # Process arm path
        processed_arm_path = []
        for point in arm_path:
            if isinstance(point, list):
                processed_arm_path.append(point)
            elif isinstance(point, np.ndarray):
                processed_arm_path.append(point.tolist())
            else:
                # Assume it has a to_list method
                processed_arm_path.append(point.to_list())

        # Create the arm trajectory message
        arm_traj_msg = JointTrajectory()
        arm_traj_msg.header.stamp = rospy.Time.now()
        arm_traj_msg.joint_names = self.planning_joint_names

        # Create the base trajectory message (using joint_names as a placeholder)
        base_traj_msg = JointTrajectory()
        base_traj_msg.header.stamp = rospy.Time.now()
        base_traj_msg.joint_names = ["x", "y", "theta"]  # Base has 3 DOF

        # Calculate time spacing for points
        estimated_duration = len(processed_arm_path) * 0.1  # Heuristic
        time_step = (
            estimated_duration / len(processed_arm_path)
            if len(processed_arm_path) > 0
            else 0
        )

        # Add points to both trajectories
        for i in range(len(processed_arm_path)):
            arm_point = JointTrajectoryPoint()
            arm_point.positions = processed_arm_path[i]
            arm_point.time_from_start = rospy.Duration(i * time_step)
            arm_traj_msg.points.append(arm_point)

            base_point = JointTrajectoryPoint()
            base_point.positions = base_configs[i]
            base_point.time_from_start = rospy.Duration(i * time_step)
            base_traj_msg.points.append(base_point)

        # Publish trajectories
        self.arm_path_pub.publish(arm_traj_msg)
        self.base_path_pub.publish(base_traj_msg)

        # Give time for controller to receive and process the message
        rospy.sleep(0.5)
        return True

    def stop_whole_body_motion(self):
        if self.motion_state == "RUNNING":
            log.info("Sending stop signal to whole-body controller.")
            self.stop_pub.publish(Bool(data=True))
            # The controller will publish a failure status, which will update the state.
            # We can also pre-emptively set it to failed.
            self.motion_state = "FAILED"

    def is_motion_done(self):
        return self.motion_state != "RUNNING"

    def get_motion_result(self):
        return self.motion_state == "SUCCEEDED"

    def cancel_head_goals(self):
        """Cancels all pending head movement goals."""
        self.head_traj_client.cancel_all_goals()

    def get_arm_action_state(self):
        """Gets the state of the arm action client."""
        return self.arm_traj_client.get_state()

    def cancel_arm_goals(self):
        """Cancels all goals for the arm action client."""
        self.arm_traj_client.cancel_all_goals()

    def cancel_torso_goals(self):
        """Cancels all goals for the torso action client."""
        self.torso_client.cancel_all_goals()

    def get_arm_action_result(self):
        """Gets the result of the arm action client."""
        return self.arm_traj_client.get_result()

    def start_joint_trajectory_async(self, trajectory_points, duration):
        """Send arm trajectory to action clients (non-blocking)."""
        # Split into torso and arm
        torso_points = [[point[0]] for point in trajectory_points]
        arm_points = [point[1:] for point in trajectory_points]

        # Create goals
        torso_goal = FollowJointTrajectoryGoal()
        torso_goal.trajectory = JointTrajectory()
        torso_goal.trajectory.joint_names = ["torso_lift_joint"]

        arm_goal = FollowJointTrajectoryGoal()
        arm_goal.trajectory = JointTrajectory()
        arm_goal.trajectory.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "upperarm_roll_joint",
            "elbow_flex_joint",
            "forearm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
        ]

        # Add trajectory points
        point_duration = duration / len(trajectory_points)
        for i, (torso_pos, arm_pos) in enumerate(zip(torso_points, arm_points)):
            torso_point = JointTrajectoryPoint()
            torso_point.positions = torso_pos
            torso_point.time_from_start = rospy.Duration(point_duration * (i + 1))
            torso_goal.trajectory.points.append(torso_point)

            arm_point = JointTrajectoryPoint()
            arm_point.positions = arm_pos
            arm_point.time_from_start = rospy.Duration(point_duration * (i + 1))
            arm_goal.trajectory.points.append(arm_point)

        # Send goals (non-blocking)
        self.torso_client.send_goal(torso_goal)
        self.arm_traj_client.send_goal(arm_goal)
        return True

    def send_joint_values(self, target_joints, duration):
        raise NotImplementedError

    def execute_joint_trajectory(self, trajectory_points, duration):
        # Split trajectory for torso and arm
        torso_points = [[point[0]] for point in trajectory_points]
        arm_points = [point[1:] for point in trajectory_points]

        # Create torso trajectory
        torso_goal = FollowJointTrajectoryGoal()
        torso_goal.trajectory = JointTrajectory()
        torso_goal.trajectory.joint_names = ["torso_lift_joint"]

        # Create arm trajectory
        arm_goal = FollowJointTrajectoryGoal()
        arm_goal.trajectory = JointTrajectory()
        arm_goal.trajectory.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "upperarm_roll_joint",
            "elbow_flex_joint",
            "forearm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
        ]

        # Add trajectory points with timing
        point_duration = duration / len(trajectory_points)
        for i in range(len(trajectory_points)):
            # Torso trajectory point
            torso_point = JointTrajectoryPoint()
            torso_point.positions = torso_points[i]
            torso_point.time_from_start = rospy.Duration(point_duration * (i + 1))
            torso_goal.trajectory.points.append(torso_point)

            # Arm trajectory point
            arm_point = JointTrajectoryPoint()
            arm_point.positions = arm_points[i]
            arm_point.time_from_start = rospy.Duration(point_duration * (i + 1))
            arm_goal.trajectory.points.append(arm_point)

        # Execute trajectories
        log.info(f"Executing trajectory with {len(trajectory_points)} waypoints...")

        # Send goals to both controllers
        self.torso_client.send_goal(torso_goal)
        self.arm_traj_client.send_goal(arm_goal)

        # Wait for completion
        timeout = rospy.Duration(duration + 5.0)
        torso_success = self.torso_client.wait_for_result(timeout)
        arm_success = self.arm_traj_client.wait_for_result(timeout)

        if torso_success and arm_success:
            log.info(
                "Cartesian interpolated trajectory execution completed successfully"
            )
            return self.arm_traj_client.get_result()
        else:
            log.warning("Trajectory execution failed or timed out")
            if not torso_success:
                log.warning(
                    f"Torso controller failed with state: {self.torso_client.get_state()}"
                )
            if not arm_success:
                log.warning(
                    f"Arm controller failed with state: {self.arm_traj_client.get_state()}"
                )
            return None

    def move_base(self, linear_x: float, angular_z: float):
        """Move the robot base with specified linear and angular velocities."""
        # Create and publish movement command
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self._base_publisher.publish(twist)

    def stop_base(self):
        """Stop all base movement."""
        twist = Twist()
        self._base_publisher.publish(twist)

    def navigate_to(self, position, orientation):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()

        # Set position
        goal.target_pose.pose.position.x = position[0]
        goal.target_pose.pose.position.y = position[1]
        goal.target_pose.pose.position.z = position[2]

        # Set orientation
        goal.target_pose.pose.orientation.x = orientation[0]
        goal.target_pose.pose.orientation.y = orientation[1]
        goal.target_pose.pose.orientation.z = orientation[2]
        goal.target_pose.pose.orientation.w = orientation[3]

        # Send goal and wait for result
        self.move_base_client.send_goal(goal)
        self.move_base_client.wait_for_result()

        return self.move_base_client.get_result()

    def set_torso_height(self, height, duration):
        log.info(f"Setting torso height to {height}")

        # Create a goal for the torso controller
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = ["torso_lift_joint"]

        # Create a single trajectory point
        point = JointTrajectoryPoint()
        point.positions = [height]
        point.time_from_start = rospy.Duration(duration)

        goal.trajectory.points.append(point)

        # Send the goal and wait for completion
        self.torso_client.send_goal(goal)
        success = self.torso_client.wait_for_result(rospy.Duration(duration + 5.0))

        if success:
            log.info("Successfully set torso height.")
            return self.torso_client.get_result()
        else:
            log.warning("Failed to set torso height.")
            return None

    def control_gripper(self, position, max_effort):
        goal = GripperCommandGoal()
        goal.command.position = position
        goal.command.max_effort = max_effort

        self.gripper_client.send_goal(goal)
        self.gripper_client.wait_for_result()

        return self.gripper_client.get_result()

    def get_gripper_status(self):
        try:
            state = self.gripper_client.get_state()
            result = None
            if state == actionlib.GoalStatus.SUCCEEDED:
                result = self.gripper_client.get_result()
            current_position = None
            if self.joint_states is not None:
                joint_dict = dict(
                    zip(self.joint_states.name, self.joint_states.position)
                )
                for joint_name in joint_dict.keys():
                    if (
                        "gripper" in joint_name.lower()
                        or "finger" in joint_name.lower()
                    ):
                        current_position = joint_dict[joint_name]
                        break
            if current_position is None:
                current_position = 0.5
            if current_position < 0.05:
                normalized_position = 0.0
            elif current_position > 0.08:
                normalized_position = 1.0
            else:
                normalized_position = (current_position - 0.05) / 0.03
                normalized_position = max(0.0, min(1.0, normalized_position))

            return {
                "position": normalized_position,
                "effort": result.effort if result else 0.0,
                "stalled": result.stalled if result else False,
                "reached_goal": (
                    result.reached_goal
                    if result
                    else (state == actionlib.GoalStatus.SUCCEEDED)
                ),
                "raw_position": current_position,
                "state": state,
            }

        except Exception as e:
            log.warning(f"Failed to get gripper status: {e}")
            return {
                "position": 0.5,
                "effort": 0.0,
                "stalled": False,
                "reached_goal": False,
                "raw_position": None,
                "state": "UNKNOWN",
                "error": str(e),
            }

    def move_head(self, pan, tilt, duration=1.0):
        # Compute current head positions
        name_to_pos = dict(zip(self.joint_states.name, self.joint_states.position))
        start_pan = name_to_pos[self.head_joint_names[0]]
        start_tilt = name_to_pos[self.head_joint_names[1]]

        # Auto time-scale based on delta and a simple max velocity
        # Keep it simple: linear interpolation with fixed rate
        default_max_vel = 1.0  # rad/s
        delta_pan = pan - start_pan
        delta_tilt = tilt - start_tilt
        max_delta = max(abs(delta_pan), abs(delta_tilt))
        min_time = max_delta / default_max_vel if max_delta > 1e-6 else 0.2
        total_time = max(duration, min_time)

        rate_hz = 30
        num_points = max(int(total_time * rate_hz), 10)
        dt = total_time / num_points

        goal = FollowJointTrajectoryGoal()
        trajectory = JointTrajectory()
        trajectory.joint_names = self.head_joint_names

        for i in range(1, num_points + 1):
            alpha = i / num_points
            pos_pan = start_pan + alpha * delta_pan
            pos_tilt = start_tilt + alpha * delta_tilt

            point = JointTrajectoryPoint()
            point.positions = [pos_pan, pos_tilt]
            point.time_from_start = rospy.Duration(i * dt)
            trajectory.points.append(point)

        goal.trajectory = trajectory

        self.head_traj_client.send_goal(goal)

    def _joint_states_callback(self, msg):
        if msg is not None and len(msg.name) > 3:
            self.joint_states = msg

    def _execution_finished_callback(self, msg):
        if self.motion_state == "RUNNING":
            if msg.data:
                self.motion_state = "SUCCEEDED"
            else:
                self.motion_state = "FAILED"
            log.debug(f"Motion execution finished callback: success={msg.data}")

    def _rgb_callback(self, msg):
        str_msg = msg.data
        buf = np.ndarray(shape=(1, len(str_msg)), dtype=np.uint8, buffer=msg.data)
        cv_image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        self.latest_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    def _depth_callback(self, msg):
        depth_fmt, compr_type = msg.format.split(";")
        depth_fmt = depth_fmt.strip()
        compr_type = compr_type.strip()
        depth_header_size = 12
        raw_data = msg.data[depth_header_size:]
        depth_img_raw = cv2.imdecode(
            np.frombuffer(raw_data, np.uint8), cv2.IMREAD_UNCHANGED
        )
        if depth_fmt == "16UC1":
            self.latest_depth = depth_img_raw.astype(np.float32) / 1000.0
        elif depth_fmt == "32FC1":
            raw_header = msg.data[:depth_header_size]
            [compfmt, depthQuantA, depthQuantB] = struct.unpack("iff", raw_header)
            depth_img_scaled = depthQuantA / (
                depth_img_raw.astype(np.float32) - depthQuantB
            )
            depth_img_scaled[depth_img_raw == 0] = 0
            self.latest_depth = depth_img_scaled
        self.latest_depth = np.nan_to_num(self.latest_depth)

    def _info_callback(self, msg):
        if self.camera_intrinsics is None:
            self.camera_intrinsics = np.array(msg.K).reshape(3, 3)
