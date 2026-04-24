#pragma once

#include <map>
#include <cassert>

#include <vamp/collision/environment.hh>
#include <vamp/planning/simplify_settings.hh>
#include <vamp/planning/plan.hh>
#include <vamp/planning/validate.hh>
#include <vamp/planning/base_configuration.hh>
#include <vamp/random/rng.hh>
#include <vamp/vector.hh>

namespace vamp::planning
{
    // Helper function to create conservative collision checking environment
    template <std::size_t rake>
    inline collision::Environment<FloatVector<rake>> create_conservative_environment(
        const collision::Environment<FloatVector<rake>> &original_env,
        const SimplifySettings &settings)
    {
        if (!settings.use_conservative_collision_checking)
        {
            return original_env;
        }

        collision::Environment<FloatVector<rake>> conservative_env = original_env;

        // Only increase pointcloud radii, leave robot parameters unchanged
        for (auto &pointcloud : conservative_env.pointclouds)
        {
            pointcloud.r_point = settings.simplify_point_radius;
        }

        return conservative_env;
    }

    // Struct to hold the result of whole body planning
    template <typename Robot>
    struct WholeBodyPlanningResult
    {
        PlanningResult<Robot::dimension> arm_result;
        std::vector<MobileBaseConfiguration<Robot>> base_path;

        // Helper function to check if path lengths match
        bool validatePaths() const
        {
            return arm_result.path.size() == base_path.size();
        }
    };

    // Helper function to validate motion with base configurations
    template <typename Robot, std::size_t rake, std::size_t resolution>
    inline static bool validate_whole_body_motion(
        const typename Robot::Configuration &arm_start,
        const typename Robot::Configuration &arm_end,
        const MobileBaseConfiguration<Robot> &base_start,
        const MobileBaseConfiguration<Robot> &base_end,
        const collision::Environment<FloatVector<rake>> &environment)
    {
        // whole-body motion validation function. The `resolution` template
        // parameter now dictates the number of intermediate checks.
        return validate_motion_wb<Robot, rake, resolution>(
            arm_start, arm_end, base_start, base_end, environment);
    }

    // Subdivide function for synchronized arm and base paths
    template <typename Robot>
    inline static void
    subdivide_paths(Path<Robot::dimension> &arm_path, std::vector<MobileBaseConfiguration<Robot>> &base_path)
    {
        assert(arm_path.size() == base_path.size());

        if (arm_path.size() < 2)
        {
            return;
        }

        Path<Robot::dimension> new_arm_path;
        std::vector<MobileBaseConfiguration<Robot>> new_base_path;

        new_arm_path.push_back(arm_path.front());
        new_base_path.push_back(base_path.front());

        for (size_t i = 1; i < arm_path.size(); ++i)
        {
            // Add midpoint
            auto mid_arm = arm_path[i - 1].interpolate(arm_path[i], 0.5);
            auto mid_base = base_path[i - 1].interpolate(base_path[i], 0.5);

            new_arm_path.push_back(mid_arm);
            new_base_path.push_back(mid_base);

            // Add original point
            new_arm_path.push_back(arm_path[i]);
            new_base_path.push_back(base_path[i]);
        }

        arm_path = new_arm_path;
        base_path = new_base_path;

        assert(arm_path.size() == base_path.size());
    }

    // Struct to hold a segment of a whole-body path
    template <typename Robot>
    struct WholeBodyPathSegment
    {
        Path<Robot::dimension> arm_path;
        std::vector<MobileBaseConfiguration<Robot>> base_path;
    };

    // Function to split a whole-body path into segments based on direction changes
    template <typename Robot>
    static std::vector<WholeBodyPathSegment<Robot>> split_whole_body_path(
        const Path<Robot::dimension> &arm_path,
        const std::vector<MobileBaseConfiguration<Robot>> &base_path)
    {
        std::vector<WholeBodyPathSegment<Robot>> segments;
        if (arm_path.empty())
        {
            return segments;
        }

        WholeBodyPathSegment<Robot> current_segment;
        current_segment.arm_path.push_back(arm_path.front());
        current_segment.base_path.push_back(base_path.front());

        for (size_t i = 1; i < base_path.size(); ++i)
        {
            if (base_path[i].direction != current_segment.base_path.back().direction)
            {
                segments.push_back(current_segment);
                current_segment.arm_path.clear();
                current_segment.base_path.clear();
                current_segment.arm_path.push_back(arm_path[i - 1]);
                current_segment.base_path.push_back(base_path[i - 1]);
            }
            current_segment.arm_path.push_back(arm_path[i]);
            current_segment.base_path.push_back(base_path[i]);
        }
        segments.push_back(current_segment);

        return segments;
    }

    // Smooth B-Spline for whole body paths
    template <typename Robot, std::size_t rake, std::size_t resolution>
    inline static auto smooth_whole_body_bspline(
        Path<Robot::dimension> &arm_path,
        std::vector<MobileBaseConfiguration<Robot>> &base_path,
        const collision::Environment<FloatVector<rake>> &environment,
        const BSplineSettings &settings) -> bool
    {
        // Verify paths have same length
        assert(arm_path.size() == base_path.size());

        if (arm_path.size() < 3)
        {
            return false;
        }

        bool changed = false;
        for (auto step = 0U; step < settings.max_steps; ++step)
        {
            // Subdivide both paths
            subdivide_paths<Robot>(arm_path, base_path);

            bool updated = false;
            for (auto index = 2U; index < arm_path.size() - 1; index += 2)
            {
                // Calculate new midpoint for arm path
                const auto arm_temp_1 =
                    arm_path[index].interpolate(arm_path[index - 1], settings.midpoint_interpolation);
                const auto arm_temp_2 =
                    arm_path[index].interpolate(arm_path[index + 1], settings.midpoint_interpolation);
                const auto arm_midpoint = arm_temp_1.interpolate(arm_temp_2, 0.5);

                // Check if the change is significant enough
                if (arm_path[index].distance(arm_midpoint) > settings.min_change)
                {
                    // Check if motion is valid with the appropriate base configurations
                    if (validate_whole_body_motion<Robot, rake, resolution>(
                            arm_path[index - 1],
                            arm_midpoint,
                            base_path[index - 1],
                            base_path[index],
                            environment) &&
                        validate_whole_body_motion<Robot, rake, resolution>(
                            arm_midpoint,
                            arm_path[index + 1],
                            base_path[index],
                            base_path[index + 1],
                            environment))
                    {
                        // Update both arm and base paths with new midpoints
                        arm_path[index] = arm_midpoint;
                        changed |= (updated = true);
                    }
                }
            }

            if (!updated)
            {
                break;
            }
        }

        return changed;
    }

    // Reduce path vertices for whole body paths
    template <typename Robot, std::size_t rake, std::size_t resolution>
    inline static auto reduce_whole_body_path_vertices(
        Path<Robot::dimension> &arm_path,
        std::vector<MobileBaseConfiguration<Robot>> &base_path,
        const collision::Environment<FloatVector<rake>> &environment,
        const ReduceSettings &settings,
        const typename vamp::rng::RNG<Robot::dimension>::Ptr rng) -> bool
    {
        // Verify paths have same length
        assert(arm_path.size() == base_path.size());

        if (arm_path.size() < 3)
        {
            return false;
        }

        const auto max_steps = (not settings.max_steps) ? arm_path.size() : settings.max_steps;
        const auto max_empty_steps =
            (not settings.max_empty_steps) ? arm_path.size() : settings.max_empty_steps;

        bool result = false;
        for (auto i = 0U, no_change = 0U; i < max_steps or no_change < max_empty_steps; ++i, ++no_change)
        {
            int initial_size = arm_path.size();
            int max_n = initial_size - 1;

            int range = 1 + static_cast<int>(
                                std::floor(0.5F + static_cast<float>(initial_size) * settings.range_ratio));

            auto point_0 = rng->dist.uniform_integer(0, max_n);
            auto point_1 =
                rng->dist.uniform_integer(std::max(point_0 - range, 0), std::min(max_n, point_0 + range));

            if (std::abs(point_0 - point_1) < 2)
            {
                if (point_0 < max_n - 1)
                {
                    point_1 = point_0 + 2;
                }
                else if (point_0 > 1)
                {
                    point_1 = point_0 - 2;
                }
                else
                {
                    continue;
                }
            }

            if (point_0 > point_1)
            {
                std::swap(point_0, point_1);
            }

            // Check if direct motion is valid with appropriate base configurations
            if (validate_whole_body_motion<Robot, rake, resolution>(
                    arm_path[point_0],
                    arm_path[point_1],
                    base_path[point_0],
                    base_path[point_1],
                    environment))
            {
                // Remove intermediate waypoints from both arm and base paths
                arm_path.erase(arm_path.begin() + point_0 + 1, arm_path.begin() + point_1);
                base_path.erase(base_path.begin() + point_0 + 1, base_path.begin() + point_1);

                assert(arm_path.size() == base_path.size());

                no_change = 0;
                result = true;
            }
        }

        return result;
    }

    // Shortcut path for whole body paths
    template <typename Robot, std::size_t rake, std::size_t resolution>
    inline static auto shortcut_whole_body_path(
        Path<Robot::dimension> &arm_path,
        std::vector<MobileBaseConfiguration<Robot>> &base_path,
        const collision::Environment<FloatVector<rake>> &environment,
        const ShortcutSettings &settings) -> bool
    {
        // Verify paths have same length
        assert(arm_path.size() == base_path.size());

        if (arm_path.size() < 3)
        {
            return false;
        }

        bool result = false;
        for (auto i = 0U; i < arm_path.size() - 2; ++i)
        {
            for (auto j = arm_path.size() - 1; j > i + 1; --j)
            {
                // Apply relaxed collision checking as specified in requirements
                if (validate_whole_body_motion<Robot, rake, resolution>(
                        arm_path[i], arm_path[j], base_path[i], base_path[j], environment))
                {
                    // Remove intermediate waypoints from both arm and base paths
                    arm_path.erase(arm_path.begin() + i + 1, arm_path.begin() + j);
                    base_path.erase(base_path.begin() + i + 1, base_path.begin() + j);

                    assert(arm_path.size() == base_path.size());

                    result = true;
                    break;
                }
            }
        }

        return result;
    }

    // Perturb path for whole body paths
    template <typename Robot, std::size_t rake, std::size_t resolution>
    inline static auto perturb_whole_body_path(
        Path<Robot::dimension> &arm_path,
        std::vector<MobileBaseConfiguration<Robot>> &base_path,
        const collision::Environment<FloatVector<rake>> &environment,
        const PerturbSettings &settings,
        const typename vamp::rng::RNG<Robot::dimension>::Ptr rng) -> bool
    {
        // Verify paths have same length
        assert(arm_path.size() == base_path.size());

        if (arm_path.size() < 3)
        {
            return false;
        }

        const auto max_steps = (not settings.max_steps) ? arm_path.size() : settings.max_steps;
        const auto max_empty_steps =
            (not settings.max_empty_steps) ? arm_path.size() : settings.max_empty_steps;

        bool changed = false;
        for (auto step = 0U, no_change = 0U; step < max_steps and no_change < max_empty_steps;
             ++step, ++no_change)
        {
            auto to_perturb_idx = rng->dist.uniform_integer(1UL, arm_path.size() - 2);
            auto perturb_state = arm_path[to_perturb_idx];
            auto before_state = arm_path[to_perturb_idx - 1];
            auto after_state = arm_path[to_perturb_idx + 1];

            auto before_base = base_path[to_perturb_idx - 1];
            auto perturb_base = base_path[to_perturb_idx];
            auto after_base = base_path[to_perturb_idx + 1];

            float old_cost = perturb_state.distance(before_state) + perturb_state.distance(after_state);

            for (auto attempt = 0U; attempt < settings.perturbation_attempts; ++attempt)
            {
                auto perturbation = rng->next();
                Robot::scale_configuration(perturbation);

                const auto new_state = perturb_state.interpolate(perturbation, settings.range);
                float new_cost = new_state.distance(before_state) + new_state.distance(after_state);

                if (new_cost < old_cost)
                {
                    // Check if perturbed motion is valid
                    if (validate_whole_body_motion<Robot, rake, resolution>(
                            before_state, new_state, before_base, perturb_base, environment) &&
                        validate_whole_body_motion<Robot, rake, resolution>(
                            new_state, after_state, perturb_base, after_base, environment))
                    {
                        no_change = 0;
                        changed = true;
                        arm_path[to_perturb_idx] = new_state;
                        // Note: We're not changing the base configuration for this operation
                        // as mentioned in the requirements - "pertub_path involves very little with the base"
                        break;
                    }
                }
            }
        }

        return changed;
    }

    // Main simplify function for whole body paths
    template <typename Robot, std::size_t rake, std::size_t resolution>
    inline auto whole_body_simplify(
        const Path<Robot::dimension> &arm_path,
        const std::vector<MobileBaseConfiguration<Robot>> &base_path,
        const collision::Environment<FloatVector<rake>> &environment,
        const SimplifySettings &settings,
        const typename vamp::rng::RNG<Robot::dimension>::Ptr rng) -> WholeBodyPlanningResult<Robot>
    {
        auto start_time = std::chrono::steady_clock::now();

        // Create result structure
        WholeBodyPlanningResult<Robot> whole_body_result;
        auto &result = whole_body_result.arm_result;
        auto &base_result = whole_body_result.base_path;

        // Verify input paths have same length
        assert(arm_path.size() == base_path.size());

        // Split the path into segments
        auto path_segments = split_whole_body_path<Robot>(arm_path, base_path);

        for (const auto& segment : path_segments)
        {
            Path<Robot::dimension> current_simplified_arm = segment.arm_path;
            std::vector<MobileBaseConfiguration<Robot>> current_simplified_base = segment.base_path;

            // Create conservative environment for collision checking once
            auto conservative_env = create_conservative_environment(environment, settings);

            // Define operations for whole body simplification
            const auto bspline = [&current_simplified_arm, &current_simplified_base, &conservative_env, settings]()
            {
                return smooth_whole_body_bspline<Robot, rake, resolution>(
                    current_simplified_arm, current_simplified_base, conservative_env, settings.bspline);
            };

            const auto reduce = [&current_simplified_arm, &current_simplified_base, &conservative_env, settings, rng]()
            {
                return reduce_whole_body_path_vertices<Robot, rake, resolution>(
                    current_simplified_arm, current_simplified_base, conservative_env, settings.reduce, rng);
            };

            const auto shortcut = [&current_simplified_arm, &current_simplified_base, &conservative_env, settings]()
            {
                return shortcut_whole_body_path<Robot, rake, resolution>(
                    current_simplified_arm, current_simplified_base, conservative_env, settings.shortcut);
            };

            const auto perturb = [&current_simplified_arm, &current_simplified_base, &conservative_env, settings, rng]()
            {
                return perturb_whole_body_path<Robot, rake, resolution>(
                    current_simplified_arm, current_simplified_base, conservative_env, settings.perturb, rng);
            };

            const std::map<SimplifyRoutine, std::function<bool()>> operations = {
                {BSPLINE, bspline},
                {REDUCE, reduce},
                {SHORTCUT, shortcut},
                {PERTURB, perturb},
            };

            // Check if straight line is valid
            if (current_simplified_arm.size() < 3 ||
                validate_whole_body_motion<Robot, rake, resolution>(
                    current_simplified_arm.front(), current_simplified_arm.back(), current_simplified_base.front(), current_simplified_base.back(), environment))
            {
                if (!result.path.empty())
                {
                    result.path.pop_back();
                    base_result.pop_back();
                }
                result.path.push_back(current_simplified_arm.front());
                base_result.push_back(current_simplified_base.front());
                result.path.push_back(current_simplified_arm.back());
                base_result.push_back(current_simplified_base.back());
                continue;
            }

            // Interpolate if requested
            if (settings.interpolate)
            {
                // Create temporary paths with additional points
                Path<Robot::dimension> new_arm_path;
                std::vector<MobileBaseConfiguration<Robot>> new_base_path;

                for (size_t i = 1; i < current_simplified_arm.size(); ++i)
                {
                    new_arm_path.push_back(current_simplified_arm[i - 1]);
                    new_base_path.push_back(current_simplified_base[i - 1]);

                    for (size_t j = 1; j < settings.interpolate; ++j)
                    {
                        float t = static_cast<float>(j) / settings.interpolate;

                        auto interp_arm = current_simplified_arm[i - 1].interpolate(current_simplified_arm[i], t);
                        auto interp_base = current_simplified_base[i - 1].interpolate(current_simplified_base[i], t);

                        new_arm_path.push_back(interp_arm);
                        new_base_path.push_back(interp_base);
                    }
                }

                // Add last point
                new_arm_path.push_back(current_simplified_arm.back());
                new_base_path.push_back(current_simplified_base.back());

                current_simplified_arm = new_arm_path;
                current_simplified_base = new_base_path;
            }

            // Run simplification iterations
            if (current_simplified_arm.size() > 2)
            {
                for (auto i = 0U; i < settings.max_iterations; ++i)
                {
                    result.iterations++;
                    bool any = false;
                    for (const auto &op : settings.operations)
                    {
                        any |= operations.find(op)->second();
                        // Verify paths remain in sync after each operation
                        assert(current_simplified_arm.size() == current_simplified_base.size());
                    }

                    if (!any)
                    {
                        break;
                    }
                }
            }

            if (!result.path.empty())
            {
                result.path.pop_back();
                base_result.pop_back();
            }

            result.path.insert(result.path.end(), current_simplified_arm.begin(), current_simplified_arm.end());
            base_result.insert(base_result.end(), current_simplified_base.begin(), current_simplified_base.end());
        }

        // Calculate path cost
        result.cost = 0.0f;
        for (size_t i = 1; i < result.path.size(); ++i)
        {
            result.cost += result.path[i].distance(result.path[i - 1]);
        }

        result.nanoseconds = vamp::utils::get_elapsed_nanoseconds(start_time);
        return whole_body_result;
    }

    // Improved whole body interpolation function with uniform density
    template <typename Robot>
    inline void interpolate_whole_body_path(WholeBodyPlanningResult<Robot> &result, float density)
    {
        // Verify that arm and base paths have the same length
        if (!result.validatePaths())
        {
            throw std::runtime_error("Arm and base paths must have the same length for whole-body "
                                     "interpolation");
        }

        // Get references to the paths
        auto &arm_path = result.arm_result.path;
        auto &base_path = result.base_path;

        // Check if we have anything to interpolate
        if (arm_path.size() < 2 || density <= 0.0f)
        {
            return;  // Nothing to do
        }

        // Create new paths for interpolated results
        vamp::planning::Path<Robot::dimension> interpolated_arm_path;
        std::vector<vamp::planning::MobileBaseConfiguration<Robot>> interpolated_base_path;

        // Reserve memory to avoid reallocations
        interpolated_arm_path.reserve(arm_path.size() * 2);
        interpolated_base_path.reserve(base_path.size() * 2);

        // Add the first waypoint
        interpolated_arm_path.push_back(arm_path.front());
        interpolated_base_path.push_back(base_path.front());

        // For each segment in the original path
        for (std::size_t i = 0; i < arm_path.size() - 1; ++i)
        {
            const auto &arm_start = arm_path[i];
            const auto &arm_end = arm_path[i + 1];
            const auto &base_start = base_path[i];
            const auto &base_end = base_path[i + 1];

            // Calculate combined distance considering both translation and rotation
            // base_distance already includes rotation: sqrt(dx² + dy²) + 0.3 * dtheta
            float base_distance = base_start.arc_distance(base_end);  // includes rotation component
            float arm_distance = arm_start.distance(arm_end);     // L2 norm of all joint angles
            float arm_weight = 0.2f;  // Reduced weight for arm distance (joint rotations)
            float segment_length = base_distance + (arm_weight * arm_distance);

            if (segment_length > density)
            {
                std::size_t num_subdivisions = static_cast<std::size_t>(std::ceil(segment_length / density));

                for (std::size_t j = 1; j <= num_subdivisions; ++j)
                {
                    float t = static_cast<float>(j) / num_subdivisions;
                    interpolated_arm_path.push_back(arm_start.interpolate(arm_end, t));
                    interpolated_base_path.push_back(base_start.interpolate(base_end, t));
                }
            }
            else
            {
                interpolated_arm_path.push_back(arm_end);
                interpolated_base_path.push_back(base_end);
            }
        }

        // Update the original paths with the interpolated ones
        arm_path = std::move(interpolated_arm_path);
        base_path = std::move(interpolated_base_path);

        // Verify paths still have the same length
        assert(arm_path.size() == base_path.size());
    }
}  // namespace vamp::planning
