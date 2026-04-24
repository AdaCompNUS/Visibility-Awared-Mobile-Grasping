#pragma once

#include <cstdint>

#include <vamp/utils.hh>
#include <vamp/vector.hh>
#include <vamp/collision/environment.hh>
#include <vamp/planning/base_configuration.hh>

namespace vamp::planning
{
    template <std::size_t n, std::size_t... I>
    inline constexpr auto generate_percents(std::index_sequence<I...>) -> std::array<float, n>
    {
        return {(static_cast<void>(I), static_cast<float>(I + 1) / static_cast<float>(n))...};
    }

    template <std::size_t n>
    struct Percents
    {
        inline static constexpr auto percents = generate_percents<n>(std::make_index_sequence<n>());
    };

    template <typename Robot, std::size_t rake, std::size_t resolution>
    inline auto validate_motion_wb(
        const typename Robot::Configuration &start,
        const typename Robot::Configuration &goal,
        const planning::MobileBaseConfiguration<Robot> &base_start,
        const planning::MobileBaseConfiguration<Robot> &base_goal,
        const collision::Environment<FloatVector<rake>> &environment) -> bool
    {
        const auto arm_vector = goal - start;
        const float arm_distance = arm_vector.l2_norm();
        const float base_distance = base_start.arc_distance(base_goal);

        const std::size_t n_checks = std::max(
            {static_cast<size_t>(1),
             static_cast<size_t>(std::ceil(arm_distance * resolution)),
             static_cast<size_t>(std::ceil(base_distance * 4 * resolution))});

        if (n_checks <= 1)
        {
            // Perform a single check at the goal configuration
            typename Robot::template ConfigurationBlock<rake> arm_block;
            for (auto i = 0U; i < Robot::dimension; ++i)
            {
                arm_block[i] = goal.broadcast(i);
            }

            typename Robot::template BaseConfigurationBlock<rake> base_block;
            for (auto i = 0U; i < Robot::base_dimension; ++i)
            {
                base_block[i] = base_goal.config.broadcast(i);
            }

            return (environment.attachments) ?
                       Robot::template fkcc_attach_wb<rake>(environment, arm_block, base_block) :
                       Robot::template fkcc_wb<rake>(environment, arm_block, base_block);
        }

        const std::size_t n_batches = (n_checks + rake - 1) / rake;
        const float n_checks_float = static_cast<float>(n_checks);

        typename Robot::template ConfigurationBlock<rake> arm_block;
        typename Robot::template BaseConfigurationBlock<rake> base_block;
        std::array<float, rake> t_data;

        for (std::size_t i = 0; i < n_batches; ++i)
        {
            const size_t current_batch_size = (i == n_batches - 1) ? ((n_checks - 1) % rake + 1) : rake;

            // Create a vector of 't' values for this batch
            for (std::size_t j = 0; j < current_batch_size; ++j)
            {
                t_data[j] = static_cast<float>(i * rake + j + 1) / n_checks_float;
            }

            // If the last batch is not full, pad it by repeating the last valid point
            if (current_batch_size < rake)
            {
                for (std::size_t j = current_batch_size; j < rake; ++j)
                {
                    t_data[j] = t_data[current_batch_size - 1];
                }
            }
            const auto t_vector = FloatVector<rake>(t_data.data());

            // Generate a block of arm configurations
            for (auto dim = 0U; dim < Robot::dimension; ++dim)
            {
                arm_block[dim] = start.broadcast(dim) + (arm_vector.broadcast(dim) * t_vector);
            }

            // Generate a block of base configurations using the new vectorized method
            base_block = base_start.template interpolate_vector<rake>(base_goal, t_vector);

            // Perform the collision check for the entire block
            if (!((environment.attachments) ?
                      Robot::template fkcc_attach_wb<rake>(environment, arm_block, base_block) :
                      Robot::template fkcc_wb<rake>(environment, arm_block, base_block)))
            {
                return false;
            }
        }

        return true;
    }

    template <typename Robot, std::size_t rake, std::size_t resolution>
    inline constexpr auto validate_vector(
        const typename Robot::Configuration &start,
        const typename Robot::Configuration &vector,
        float distance,
        const collision::Environment<FloatVector<rake>> &environment) -> bool
    {
        // TODO: Fix use of reinterpret_cast in pack() so that this can be constexpr
        const auto percents = FloatVector<rake>(Percents<rake>::percents);

        typename Robot::template ConfigurationBlock<rake> block;

        // HACK: broadcast() implicitly assumes that the rake is exactly VectorWidth
        for (auto i = 0U; i < Robot::dimension; ++i)
        {
            block[i] = start.broadcast(i) + (vector.broadcast(i) * percents);
        }

        const std::size_t n = std::max(std::ceil(distance / static_cast<float>(rake) * resolution), 1.F);

        bool valid = (environment.attachments) ? Robot::template fkcc_attach<rake>(environment, block) :
                                                 Robot::template fkcc<rake>(environment, block);
        if (not valid or n == 1)
        {
            return valid;
        }

        const auto backstep = vector / (rake * n);
        for (auto i = 1U; i < n; ++i)
        {
            for (auto j = 0U; j < Robot::dimension; ++j)
            {
                block[j] = block[j] - backstep.broadcast(j);
            }

            if (not Robot::template fkcc<rake>(environment, block))
            {
                return false;
            }
        }

        return true;
    }

    template <typename Robot, std::size_t rake, std::size_t resolution>
    inline constexpr auto validate_motion(
        const typename Robot::Configuration &start,
        const typename Robot::Configuration &goal,
        const collision::Environment<FloatVector<rake>> &environment) -> bool
    {
        auto vector = goal - start;
        return validate_vector<Robot, rake, resolution>(start, vector, vector.l2_norm(), environment);
    }
}  // namespace vamp::planning
