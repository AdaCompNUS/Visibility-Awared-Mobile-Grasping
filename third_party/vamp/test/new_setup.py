from pathlib import Path
import vamp
from fire import Fire
import time
from typing import List

# Starting configurations for two robots (half as many joints)
start_config: List[float] = [0.1, 1.32, 1.4, -0.2, 1.72, 0, 1.66, 0]

# Goal configuration for single robot
goal_config: List[float] = [
    9.534866408572472e-05,
    0.7963384324722192,
    -0.1402975254110037,
    1.632407460562802,
    -1.194633123110403,
    -0.1493910070461753,
    0.1045060268051927,
    -1.571510562525571,
]

# Define spheres for the environment
problem = [
    # Workspace obstacles
    [0.0, 3, 3],
]


def check_configuration_validity(
    config: List[float], env: vamp.Environment, vamp_module
) -> bool:
    """Check if a configuration is valid and print detailed information."""
    try:
        # Create configuration object
        config_obj = vamp_module.Configuration(config)

        # Check validity
        is_valid = vamp_module.validate(config_obj, env)
        if not is_valid:
            print("Configuration is invalid!")
            validity = vamp_module.sphere_validity(config, env)
            print("Sphere validity check:", validity)
        return is_valid
    except Exception as e:
        print(f"Error checking configuration validity: {str(e)}")
        return False


def main(
    visualize: bool = True,
    planner: str = "rrtc",
    sampler_name: str = "halton",
    radius: float = 0.0001,  # Radius for obstacle spheres
    skip_rng_iterations: int = 0,
    **kwargs,
):
    # Configure planner for fetch robot
    (vamp_module, planner_func, plan_settings, simp_settings) = (
        vamp.configure_robot_and_planner_with_kwargs("fetch", planner, **kwargs)
    )

    # Create sampler
    sampler = getattr(vamp_module, sampler_name)()
    sampler.skip(skip_rng_iterations)

    # Create environment
    e = vamp.Environment()
    for sphere in problem:
        e.add_sphere(vamp.Sphere(sphere, radius))

    # Print information about the planning attempt
    print("Planning motion for Fetch robot...")
    print(f"Start configuration: {start_config}")
    print(f"Goal configuration: {goal_config}")

    # Validate start configuration
    print("\nValidating start configuration...")
    if vamp_module.validate(start_config, e):
        print("Starting config is valid")
    else:
        print("Starting config is not valid!")

    # Validate goal configuration
    print("\nValidating goal configuration...")
    if vamp_module.validate(goal_config, e):
        print("Goal config is valid")
    else:
        print("The goal config is not valid!")

    # Attempt motion planning
    print("\nStarting motion planning...")
    start_time = time.time()

    try:
        result = planner_func(start_config, goal_config, e, plan_settings, sampler)

        end_time = time.time()
        total_time = (end_time - start_time) * 1000

        if result.solved:
            print(f"Successfully found a path! Total time: {total_time:.2f}ms")
            simple = vamp_module.simplify(result.path, e, simp_settings, sampler)

            stats = vamp.results_to_dict(result, simple)
            print(
                f"""
Planning Time: {stats['planning_time'].microseconds:8d}μs
Simplify Time: {stats['simplification_time'].microseconds:8d}μs
   Total Time: {stats['total_time'].microseconds:8d}μs

Planning Iters: {stats['planning_iterations']}
n Graph States: {stats['planning_graph_size']}

Path Length:
   Initial: {stats['initial_path_cost']:5.3f}
Simplified: {stats['simplified_path_cost']:5.3f}"""
            )

            if visualize:
                from vamp import pybullet_interface as vpb

                robot_dir = Path(__file__).parent.parent / "resources" / "fetch"
                sim = vpb.PyBulletSimulator(
                    str(robot_dir / "fetch_spherized.urdf"),
                    vamp.ROBOT_JOINTS["fetch"],
                    True,
                )

                # Add environment spheres
                for sphere in problem:
                    sim.add_sphere(radius, sphere)

                # Prepare and animate path
                simple.path.interpolate(vamp_module.resolution())
                sim.animate(simple.path)

        else:
            print(f"Failed to find a valid path! Time spent: {total_time:.2f}ms")
            print(f"Planning iterations: {result.iterations}")
            print(f"Graph size: {result.size}")

    except Exception as e:
        print(f"An error occurred during planning: {str(e)}")


if __name__ == "__main__":
    Fire(main)
