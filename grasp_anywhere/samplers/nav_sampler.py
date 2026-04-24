from grasp_anywhere.utils import reachability_utils
from grasp_anywhere.utils.logger import log
from grasp_anywhere.utils.visualization_utils import (
    show_costmap,
)


class NavSampler:
    """
    Pure navigation base placement sampler for baseline navigation and manipulation.
    A simplified version of PreposeSampler that:
    1. Samples base pose using reachability map and 2D SDF (costmap).
    2. Does NOT plan for the arm (assumes fixed TUCK_JOINTS).
    3. Does NOT use capability map.
    4. Validate collision with TUCK_JOINTS.
    """

    def __init__(
        self,
        robot,
        object_center_world,
        manipulation_radius,
        arm_config,
        num_samples=20,
        enable_visualization=False,
    ):
        self.robot = robot
        self.object_center_world = object_center_world
        self.manipulation_radius = manipulation_radius
        self.arm_config = arm_config
        self.num_samples = num_samples
        self.enable_visualization = enable_visualization
        self.failed_due_to_reachability = False

    def generator(self, combined_points):
        """
        Yields valid (base_config, arm_config) tuples.
        arm_config is always self.arm_config.
        """
        self.failed_due_to_reachability = False

        # 1. Update base sampler from point cloud (2D SDF / Costmap)
        if self.robot.base_sampler.dynamic_costmap:
            self.robot.base_sampler.update_from_pointcloud(combined_points)
            if (
                self.enable_visualization
                and self.robot.base_sampler.costmap is not None
            ):
                show_costmap(self.robot.base_sampler.costmap)

        reject_base_sample = 0
        reject_reachability = 0
        reject_collision = 0

        # We perform sampling and yielding
        valid_count = 0
        for _ in range(self.num_samples):
            # 2. Sample Base Config
            base_config = self.robot.base_sampler.sample_base_pose(
                self.object_center_world, manipulation_radius=self.manipulation_radius
            )
            if base_config is None:
                reject_base_sample += 1
                continue

            # 2b. Filter base configs with low reachability (position-only map)
            base_reach = reachability_utils.query_reachability_score(
                self.robot.reachability_map,
                base_config,
                self.object_center_world.tolist(),
            )

            if base_reach <= 0.04:
                reject_reachability += 1
                continue

            # 3. Check Collision with Fixed Arm Config
            if self.robot.validate_whole_body_config(self.arm_config, base_config):
                reject_collision += 1
                continue

            # Yield valid config
            valid_count += 1
            yield base_config, self.arm_config

        # Logging summary
        log.info(
            f"NavSampler summary: Total={self.num_samples}, Valid={valid_count}. "
            f"Rejections: Base={reject_base_sample}, Reach={reject_reachability}, Collision={reject_collision}"
        )

        usable_base = self.num_samples - reject_base_sample
        if (usable_base == 0) or (
            usable_base > 0 and reject_reachability == usable_base and valid_count == 0
        ):
            self.failed_due_to_reachability = True
