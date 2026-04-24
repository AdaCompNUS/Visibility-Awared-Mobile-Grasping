#pragma once

#include <map>
#include <cassert>
#include <functional>

#include <vamp/collision/environment.hh>
#include <vamp/planning/simplify_settings.hh>
#include <vamp/planning/plan.hh>
#include <vamp/planning/validate.hh>
#include <vamp/planning/base_configuration.hh>
#include <vamp/vector.hh>

namespace vamp::planning
{
    // Struct to hold the result of base planning
    template <typename Robot>
    struct BasePlanningResult
    {
        std::vector<MobileBaseConfiguration<Robot>> path;
        PlanningResult<Robot::dimension> info;  // For stats like iterations, time
    };

    template <typename Robot, std::size_t rake>
    inline static bool validate_base_motion(
        const MobileBaseConfiguration<Robot> &start,
        const MobileBaseConfiguration<Robot> &end,
        const collision::Environment<FloatVector<rake>> &environment,
        const typename Robot::Configuration &default_arm_config)
    {
        return validate_motion_wb<Robot, rake, 32>(
            default_arm_config, default_arm_config, start, end, environment);
    }

    template <typename Robot, std::size_t rake>
    inline static auto simplify_base_path(
        const std::vector<MobileBaseConfiguration<Robot>> &path,
        const collision::Environment<FloatVector<rake>> &environment,
        const SimplifySettings &settings,
        const typename Robot::Configuration &default_arm_config) -> BasePlanningResult<Robot>
    {
        auto start_time = std::chrono::steady_clock::now();
        BasePlanningResult<Robot> result;
        (void)settings;  // Silence unused variable warning

        // --- HARDCODED DENSITY as per request ---
        constexpr float max_segment_length = 0.25f;

        if (path.empty())
        {
            result.info.nanoseconds = vamp::utils::get_elapsed_nanoseconds(start_time);
            return result;
        }

        // --- STAGE 1: Resample path for uniform density ---
        std::vector<MobileBaseConfiguration<Robot>> resampled_path;
        if (path.size() >= 2)
        {
            resampled_path.reserve(path.size() * 2);
            resampled_path.push_back(path.front());

            for (std::size_t i = 0; i < path.size() - 1; ++i)
            {
                const auto &start_point = path[i];
                const auto &end_point = path[i + 1];
                const float segment_length = start_point.arc_distance(end_point);

                if (segment_length > max_segment_length)
                {
                    const std::size_t num_subdivisions =
                        static_cast<std::size_t>(std::ceil(segment_length / max_segment_length));
                    for (std::size_t j = 1; j <= num_subdivisions; ++j)
                    {
                        const float t = static_cast<float>(j) / static_cast<float>(num_subdivisions);
                        // MODIFIED: Calling the existing 2-argument version of interpolate
                        resampled_path.push_back(start_point.interpolate(end_point, t));
                    }
                }
                else
                {
                    resampled_path.push_back(end_point);
                }
            }
        }
        else  // path.size() == 1
        {
            resampled_path = path;
        }

        // --- STAGE 2: Ensure minimum path length ---
        auto &path_to_finalize = resampled_path;

        if (path_to_finalize.size() < 8)
        {
            std::vector<MobileBaseConfiguration<Robot>> interpolated_path;
            const int new_num_points = 8;
            interpolated_path.reserve(new_num_points);

            if (path_to_finalize.size() <= 1)
            {
                for (int i = 0; i < new_num_points; ++i)
                {
                    interpolated_path.push_back(path_to_finalize.front());
                }
            }
            else
            {
                for (int i = 0; i < new_num_points; ++i)
                {
                    float mapping = static_cast<float>(i) / (new_num_points - 1);
                    float old_index_float = mapping * (path_to_finalize.size() - 1);
                    int index1 = static_cast<int>(old_index_float);
                    int index2 = std::min(index1 + 1, static_cast<int>(path_to_finalize.size() - 1));
                    float t = old_index_float - index1;
                    // MODIFIED: Calling the existing 2-argument version of interpolate
                    interpolated_path.push_back(
                        path_to_finalize[index1].interpolate(path_to_finalize[index2], t));
                }
            }
            result.path = interpolated_path;
        }
        else
        {
            result.path = path_to_finalize;
        }

        result.info.cost = 0.0f;
        for (size_t i = 1; i < result.path.size(); ++i)
        {
            result.info.cost += result.path[i].distance(result.path[i - 1]);
        }

        result.info.nanoseconds = vamp::utils::get_elapsed_nanoseconds(start_time);
        return result;
    }
}  // namespace vamp::planning