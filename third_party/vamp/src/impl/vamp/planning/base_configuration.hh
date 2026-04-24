#pragma once

#include <vamp/vector.hh>
#include <array>
#include <cmath>
#include <vector>
#include <numeric>

// Include the separate Reeds-Shepp header.
#include "reeds_shepp.hh"

namespace vamp::planning
{
    template <typename Robot>
    struct alignas(FloatVectorAlignment) MobileBaseConfiguration
    {
        using Vector = FloatVector<Robot::base_dimension>;
        Vector config;
        int direction = 1;

        MobileBaseConfiguration() = default;

        MobileBaseConfiguration(const Vector &base_config, int dir = 1) : config(base_config), direction(dir)
        {
        }

        MobileBaseConfiguration(const typename Robot::BaseConfigurationArray &base_config, int dir = 1)
          : config(base_config), direction(dir)
        {
        }

        auto get_config_array() const -> typename Robot::BaseConfigurationArray;
        float distance(const MobileBaseConfiguration &other) const;
        float arc_distance(const MobileBaseConfiguration &other) const;
        float arc_distance_reverse_penalty(const MobileBaseConfiguration &other) const;
        MobileBaseConfiguration interpolate(const MobileBaseConfiguration &other, float t) const;

        template <std::size_t rake>
        auto interpolate_vector(const MobileBaseConfiguration &other, const FloatVector<rake> &t) const ->
            typename Robot::template BaseConfigurationBlock<rake>;
    };

    // --- Implementation of methods ---

    template <typename Robot>
    auto MobileBaseConfiguration<Robot>::get_config_array() const -> typename Robot::BaseConfigurationArray
    {
        typename Robot::BaseConfigurationArray arr;
        arr[0] = config.data[0][0];
        arr[1] = config.data[0][1];
        arr[2] = config.data[0][2];
        return arr;
    }

    // This is the heuristic distance (e.g., for A*), not the true path length.
    // It should remain as a fast, admissible estimate.
    template <typename Robot>
    float MobileBaseConfiguration<Robot>::distance(const MobileBaseConfiguration &other) const
    {
        float dx = config.data[0][0] - other.config.data[0][0];
        float dy = config.data[0][1] - other.config.data[0][1];
        float dtheta = std::abs(config.data[0][2] - other.config.data[0][2]);
        while (dtheta > M_PI)
        {
            dtheta = 2.0f * M_PI - dtheta;
        }
        return std::sqrt(dx * dx + dy * dy) + 0.3f * dtheta;
    }

    // This function now correctly calculates the TRUE Reeds-Shepp path length
    // by calling the library's distance function.
    template <typename Robot>
    float MobileBaseConfiguration<Robot>::arc_distance(const MobileBaseConfiguration<Robot> &other) const
    {
        double q0[3] = {
            static_cast<double>(config.data[0][0]),
            static_cast<double>(config.data[0][1]),
            static_cast<double>(config.data[0][2])};
        double q1[3] = {
            static_cast<double>(other.config.data[0][0]),
            static_cast<double>(other.config.data[0][1]),
            static_cast<double>(other.config.data[0][2])};

        // This hardcoded value should come from a Robot trait, e.g., Robot::turning_radius
        const double turning_radius = Robot::turning_radius;
        detail::reeds_shepp_gh::ReedsSheppStateSpace rs_space(turning_radius);

        return static_cast<float>(rs_space.distance(q0, q1));
    }

    template <typename Robot>
    float MobileBaseConfiguration<Robot>::arc_distance_reverse_penalty(
        const MobileBaseConfiguration<Robot> &other) const
    {
        double q0[3] = {
            static_cast<double>(config.data[0][0]),
            static_cast<double>(config.data[0][1]),
            static_cast<double>(config.data[0][2])};
        double q1[3] = {
            static_cast<double>(other.config.data[0][0]),
            static_cast<double>(other.config.data[0][1]),
            static_cast<double>(other.config.data[0][2])};

        const double turning_radius = Robot::turning_radius;
        detail::reeds_shepp_gh::ReedsSheppStateSpace rs_space(turning_radius);

        auto path = rs_space.reedsShepp(q0, q1);
        const double weighted_len_scaled =
            detail::reeds_shepp_gh::ReedsSheppStateSpace::weightedLength(path, Robot::reverse_penalty);

        return static_cast<float>(weighted_len_scaled * turning_radius);
    }

    template <typename Robot>
    MobileBaseConfiguration<Robot>
    MobileBaseConfiguration<Robot>::interpolate(const MobileBaseConfiguration<Robot> &other, float t) const
    {
        // Clamp t to be safe and handle edge cases
        t = std::max(0.0f, std::min(1.0f, t));
        if (t < 1e-6f)
        {
            return *this;
        }
        if (t > 1.0f - 1e-6f)
        {
            return other;
        }

        // Check for in-place rotation (negligible translation)
        float dx = other.config.data[0][0] - config.data[0][0];
        float dy = other.config.data[0][1] - config.data[0][1];
        if (dx * dx + dy * dy < 1e-6f)
        {
            MobileBaseConfiguration<Robot> result = *this;
            // Simple linear interpolation for angle
            float start_theta = config.data[0][2];
            float end_theta = other.config.data[0][2];
            
            // Handle angle wrap-around for shortest path
            float diff = end_theta - start_theta;
            while (diff > M_PI) diff -= 2.0f * M_PI;
            while (diff < -M_PI) diff += 2.0f * M_PI;
            
            result.config.data[0][0] = config.data[0][0]; // Position remains same
            result.config.data[0][1] = config.data[0][1];
            result.config.data[0][2] = start_theta + t * diff;
            
            // Normalize the resulting angle
            while (result.config.data[0][2] > M_PI) result.config.data[0][2] -= 2.0f * M_PI;
            while (result.config.data[0][2] < -M_PI) result.config.data[0][2] += 2.0f * M_PI;
            
            result.direction = other.direction;
            return result;
        }

        // Prepare start and end states in the format required by the library (double[3])
        double q0[3] = {
            static_cast<double>(config.data[0][0]),
            static_cast<double>(config.data[0][1]),
            static_cast<double>(config.data[0][2])};
        double q1[3] = {
            static_cast<double>(other.config.data[0][0]),
            static_cast<double>(other.config.data[0][1]),
            static_cast<double>(other.config.data[0][2])};

        // Instantiate the Reeds-Shepp state space with the robot's turning radius
        // TODO: This hardcoded value should come from a Robot trait, e.g., Robot::turning_radius
        const double turning_radius = Robot::turning_radius;
        detail::reeds_shepp_gh::ReedsSheppStateSpace rs_space(turning_radius);

        // Calculate the shortest Reeds-Shepp path
        auto path = rs_space.reedsShepp(q0, q1);
        double path_len_scaled = path.length();  // This is length in units of turning radius

        // Calculate the distance to sample along the path
        double sample_dist_scaled = t * path_len_scaled;

        // Use the library's interpolate function to get the new state
        double q_interp[3];
        rs_space.interpolate(q0, path, sample_dist_scaled, q_interp);

        // Create the resulting configuration, casting back to float
        MobileBaseConfiguration<Robot> result;
        result.config.data[0][0] = static_cast<float>(q_interp[0]);
        result.config.data[0][1] = static_cast<float>(q_interp[1]);
        result.config.data[0][2] = static_cast<float>(q_interp[2]);

        // Preserve the "end direction" from the original Hybrid A* segment
        result.direction = other.direction;

        return result;
    }

    template <typename Robot>
    template <std::size_t rake>
    auto MobileBaseConfiguration<Robot>::interpolate_vector(
        const MobileBaseConfiguration &other,
        const FloatVector<rake> &t_vec) const -> typename Robot::template BaseConfigurationBlock<rake>
    {
        typename Robot::template BaseConfigurationBlock<rake> result_block;

        // Check for in-place rotation
        float dx = other.config.data[0][0] - config.data[0][0];
        float dy = other.config.data[0][1] - config.data[0][1];
        if (dx * dx + dy * dy < 1e-6f)
        {
             // Loop through each element of the FloatVector
            for (std::size_t i = 0; i < rake; ++i)
            {
                float t = t_vec.data[0][i];
                t = std::max(0.0f, std::min(1.0f, t));  // Clamp t
                
                // Simple linear interpolation for angle
                float start_theta = config.data[0][2];
                float end_theta = other.config.data[0][2];
                
                float diff = end_theta - start_theta;
                while (diff > M_PI) diff -= 2.0f * M_PI;
                while (diff < -M_PI) diff += 2.0f * M_PI;

                float final_theta = start_theta + t * diff;
                 // Normalize
                while (final_theta > M_PI) final_theta -= 2.0f * M_PI;
                while (final_theta < -M_PI) final_theta += 2.0f * M_PI;

                result_block[0].data[0][i] = config.data[0][0];
                result_block[1].data[0][i] = config.data[0][1];
                result_block[2].data[0][i] = final_theta;
            }
            return result_block;
        }

        // Pre-calculate the single Reeds-Shepp path once for efficiency.
        double q0[3] = {
            static_cast<double>(config.data[0][0]),
            static_cast<double>(config.data[0][1]),
            static_cast<double>(config.data[0][2])};
        double q1[3] = {
            static_cast<double>(other.config.data[0][0]),
            static_cast<double>(other.config.data[0][1]),
            static_cast<double>(other.config.data[0][2])};

        // TODO: This hardcoded value should come from a Robot trait, e.g., Robot::turning_radius
        const double turning_radius = Robot::turning_radius;
        detail::reeds_shepp_gh::ReedsSheppStateSpace rs_space(turning_radius);

        auto path = rs_space.reedsShepp(q0, q1);
        double path_len_scaled = path.length();

        // Loop through each element of the FloatVector
        for (std::size_t i = 0; i < rake; ++i)
        {
            float t = t_vec.data[0][i];
            t = std::max(0.0f, std::min(1.0f, t));  // Clamp t

            double sample_dist_scaled = static_cast<double>(t) * path_len_scaled;

            double q_interp[3];
            rs_space.interpolate(q0, path, sample_dist_scaled, q_interp);

            // Assign the result to the correct 'lane' of the result block
            result_block[0].data[0][i] = static_cast<float>(q_interp[0]);
            result_block[1].data[0][i] = static_cast<float>(q_interp[1]);
            result_block[2].data[0][i] = static_cast<float>(q_interp[2]);
        }

        return result_block;
    }

}  // namespace vamp::planning