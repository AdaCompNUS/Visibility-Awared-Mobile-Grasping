try:
    import rospy
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Path
    from tf.transformations import quaternion_from_euler
except Exception:
    rospy = None
    PoseStamped = None
    Path = None
    quaternion_from_euler = None

from grasp_anywhere.utils.logger import log


def publish_base_path(path_pub, base_configs, frame_id):
    """
    Publishes a base path as a nav_msgs/Path for RViz visualization.

    Args:
        path_pub: The rospy.Publisher for the Path message.
        base_configs: A list of base configurations, where each is [x, y, theta].
        frame_id: The frame_id for the path's header.
    """
    if (
        rospy is None
        or Path is None
        or PoseStamped is None
        or quaternion_from_euler is None
    ):
        return
    path_msg = Path()
    path_msg.header.frame_id = frame_id
    path_msg.header.stamp = rospy.Time.now()

    for config in base_configs:
        pose_stamped = PoseStamped()
        # The header stamp should be the same for all poses in the path
        pose_stamped.header.frame_id = frame_id
        pose_stamped.header.stamp = path_msg.header.stamp

        pose_stamped.pose.position.x = config[0]
        pose_stamped.pose.position.y = config[1]
        pose_stamped.pose.position.z = 0  # Assuming base path is on a 2D plane

        q = quaternion_from_euler(0, 0, config[2])
        pose_stamped.pose.orientation.x = q[0]
        pose_stamped.pose.orientation.y = q[1]
        pose_stamped.pose.orientation.z = q[2]
        pose_stamped.pose.orientation.w = q[3]

        path_msg.poses.append(pose_stamped)

    path_pub.publish(path_msg)
    log.info(f"Published base path with {len(base_configs)} poses to {path_pub.name}.")
