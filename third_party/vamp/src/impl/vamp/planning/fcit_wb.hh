#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <unordered_set>
#include <vector>

#include <pdqsort.h>

#include <vamp/collision/environment.hh>
#include <vamp/planning/base_configuration.hh>
#include <vamp/planning/nn.hh>
#include <vamp/planning/plan.hh>
#include <vamp/planning/roadmap.hh>
#include <vamp/planning/utils.hh>
#include <vamp/planning/validate.hh>
#include <vamp/random/rng.hh>
#include <vamp/utils.hh>
#include <vamp/vector.hh>

namespace vamp::planning
{

    struct BaseBounds
    {
        float x_min{-1.0F};
        float x_max{1.0F};
        float y_min{-1.0F};
        float y_max{1.0F};
        float theta_min{-static_cast<float>(M_PI)};
        float theta_max{static_cast<float>(M_PI)};
    };

    namespace detail
    {
        template <typename Robot>
        inline auto
        scale_unit_to_base_config(const FloatVector<Robot::base_dimension> &unit, const BaseBounds &bounds)
            -> MobileBaseConfiguration<Robot>
        {
            typename Robot::BaseConfigurationArray arr{};
            std::array<float, Robot::base_dimension> tmp{};
            unit.to_array(tmp.data());

            const auto scale = [](float u, float lo, float hi) -> float { return lo + (hi - lo) * u; };

            arr[0] = scale(tmp[0], bounds.x_min, bounds.x_max);
            arr[1] = scale(tmp[1], bounds.y_min, bounds.y_max);
            if constexpr (Robot::base_dimension >= 3)
            {
                arr[2] = scale(tmp[2], bounds.theta_min, bounds.theta_max);
            }

            return MobileBaseConfiguration<Robot>(arr);
        }
    }  // namespace detail

    struct WBQueueEdge
    {
        unsigned int index;
        unsigned int parent;
        float cost;

        inline constexpr auto operator==(const WBQueueEdge &o) const noexcept -> bool
        {
            return index == o.index;
        }
    };

    struct FCITWBRoadmapNode
    {
    public:
        FCITWBRoadmapNode(
            unsigned int index,
            float g = std::numeric_limits<float>::infinity(),
            unsigned int sampleIdx = 0)
          : index(index), g(g), sampleIdx(sampleIdx)
        {
        }

        unsigned int index;
        float g;                 // best-known cost-to-come (arm + base)
        unsigned int sampleIdx;  // iterator start for neighbor generation

        struct Neighbor
        {
            unsigned int index;
            float distance;  // used as f-cost for ordering
        };

        std::vector<Neighbor> neighbors;
        std::vector<Neighbor>::iterator neighbor_iterator;
        std::unordered_set<int> invalidList;  // invalid parent-child relations
    };

    template <
        typename Robot,
        std::size_t rake,
        std::size_t resolution,
        typename NeighborParamsT = FCITStarNeighborParams>
    struct FCITWB
    {
        using Configuration = typename Robot::Configuration;  // arm configuration
        static constexpr auto dimension = Robot::dimension;
        using RNG = typename vamp::rng::RNG<Robot::dimension>;
        using BaseConfig = MobileBaseConfiguration<Robot>;

        struct Result
        {
            PlanningResult<dimension> arm_result;
            std::vector<BaseConfig> base_path;
            std::size_t nanoseconds{0};
        };

    public:
        // Compute Reeds-Shepp distance with a penalty applied to backward motion.
        inline static float
        rs_arc_distance_weighted_(const BaseConfig &a, const BaseConfig &b, double backwardPenalty);

        // Single-goal overload
        inline static auto solve(
            const Configuration &arm_start,
            const Configuration &arm_goal,
            const BaseConfig &base_start,
            const BaseConfig &base_goal,
            const collision::Environment<FloatVector<rake>> &environment,
            const RoadmapSettings<NeighborParamsT> &settings,
            const BaseBounds &base_bounds,
            typename RNG::Ptr &rng) noexcept -> Result
        {
            return solve(
                arm_start,
                std::vector<Configuration>{arm_goal},
                base_start,
                std::vector<BaseConfig>{base_goal},
                environment,
                settings,
                base_bounds,
                rng);
        }

        // Multi-goal overload
        inline static auto solve(
            const Configuration &arm_start,
            const std::vector<Configuration> &arm_goals,
            const BaseConfig &base_start,
            const std::vector<BaseConfig> &base_goals,
            const collision::Environment<FloatVector<rake>> &environment,
            const RoadmapSettings<NeighborParamsT> &settings,
            const BaseBounds &base_bounds,
            typename RNG::Ptr &rng) noexcept -> Result
        {
            auto start_time = std::chrono::steady_clock::now();

            Result wb_result;
            PlanningResult<dimension> &arm_result = wb_result.arm_result;

            NN<dimension> roadmap;  // arm space NN structure

            std::size_t iter = 0;
            auto states = std::unique_ptr<float>(
                vamp::utils::vector_alloc<float, FloatVectorAlignment, FloatVectorWidth>(
                    settings.max_samples * Configuration::num_scalars_rounded));

            std::vector<FCITWBRoadmapNode> nodes;
            nodes.reserve(settings.max_samples);
            std::vector<unsigned int> parents;
            parents.reserve(settings.max_samples);
            std::vector<BaseConfig> base_states;  // base state per node index
            base_states.reserve(settings.max_samples);

            const auto state_index = [&states](unsigned int index) -> float *
            { return states.get() + index * Configuration::num_scalars_rounded; };

            // Add start and goals to structures
            constexpr const unsigned int start_index = 0;

            float *start_state = state_index(start_index);
            arm_start.to_array(start_state);
            parents.emplace_back(std::numeric_limits<unsigned int>::max());
            nodes.emplace_back(start_index, 0.0F);
            roadmap.insert(NNNode<dimension>{start_index, {start_state}});
            auto &start_node = nodes[start_index];
            start_node.neighbor_iterator = start_node.neighbors.begin();
            base_states.emplace_back(base_start);

            // Insert arm+base goals consecutively starting at index 1
            for (std::size_t i = 0; i < arm_goals.size(); ++i)
            {
                const auto &goal_arm = arm_goals[i];
                const auto &goal_base = base_goals[std::min(i, base_goals.size() - 1)];
                std::size_t index = nodes.size();
                auto *goal_state = state_index(static_cast<unsigned int>(index));
                goal_arm.to_array(goal_state);
                parents.emplace_back(std::numeric_limits<unsigned int>::max());
                nodes.emplace_back(static_cast<unsigned int>(index));
                roadmap.insert(NNNode<dimension>{index, {goal_state}});
                base_states.emplace_back(goal_base);
            }

            Configuration temp_config;
            Configuration temp_config_self;
            std::vector<WBQueueEdge> open_set;

            // Search until initial solution (or optimized if requested)
            while (nodes.size() < settings.max_samples && iter++ < settings.max_iterations)
            {
                for (std::size_t i = 0; i < arm_goals.size(); ++i)
                {
                    const auto &goal_arm = arm_goals[i];
                    const auto &goal_base = base_states[i + 1];  // aligned with nodes
                    const auto &goal_node = nodes[i + 1];

                    // Iterate through neighbors and add all outgoing neighbors from start
                    for (auto it = nodes.begin() + start_node.sampleIdx; it != nodes.end(); ++it)
                    {
                        if (it->index == start_index)
                        {
                            start_node.sampleIdx++;
                            continue;
                        }

                        const auto neighbor_index = it->index;
                        temp_config = state_index(neighbor_index);
                        const auto arm_neighbor_distance = arm_start.distance(temp_config);
                        const auto base_neighbor_distance = rs_arc_distance_weighted_(
                                                            base_states[start_index], base_states[neighbor_index], settings.reverse_weight);
                        const float neighbor_total_cost = arm_neighbor_distance + base_neighbor_distance;
                        start_node.sampleIdx = std::max(neighbor_index, start_node.sampleIdx);

                        // g(p) + c^(p,c) < g(c)
                        if (neighbor_total_cost < it->g)
                        {
                            // f^(c) = g(p) + c^(p,c) + h^(c)
                            const auto cost =
                                neighbor_total_cost + goal_arm.distance(temp_config) +
                                rs_arc_distance_weighted_(base_states[neighbor_index], goal_base, settings.reverse_weight);
                            start_node.neighbors.emplace_back(
                                FCITWBRoadmapNode::Neighbor{neighbor_index, cost});
                        }
                        start_node.sampleIdx++;
                    }

                    pdqsort_branchless(
                        start_node.neighbors.begin(),
                        start_node.neighbors.end(),
                        [](const auto &a, const auto &b) { return a.distance < b.distance; });
                    start_node.neighbor_iterator = start_node.neighbors.begin();

                    if (start_node.neighbor_iterator != start_node.neighbors.end())
                    {
                        open_set.emplace_back(WBQueueEdge{
                            (*start_node.neighbor_iterator).index,
                            start_index,
                            (*start_node.neighbor_iterator).distance});
                        start_node.neighbor_iterator++;
                    }

                    while (!open_set.empty())
                    {
                        pdqsort_branchless(
                            open_set.begin(),
                            open_set.end(),
                            [](const auto &a, const auto &b) { return a.cost > b.cost; });

                        const auto current = open_set.back();
                        open_set.pop_back();

                        const auto current_index = current.index;
                        auto &current_node = nodes[current_index];
                        auto current_g = current_node.g;
                        auto current_p = current.parent;

                        auto &parent_node = nodes[current_p];

                        // Advance parent neighbor iterator if possible (best-first over neighbors)
                        while (parent_node.neighbor_iterator != parent_node.neighbors.end())
                        {
                            const auto &node = nodes[(*parent_node.neighbor_iterator).index];
                            const auto node_arm = Configuration(state_index(node.index));
                            const auto node_base = base_states[node.index];
                            const float h_estimate = goal_arm.distance(node_arm) +
                                                     rs_arc_distance_weighted_(node_base, goal_base, 20.0F);

                            if ((*parent_node.neighbor_iterator).distance < node.g + h_estimate)
                            {
                                open_set.emplace_back(WBQueueEdge{
                                    (*parent_node.neighbor_iterator).index,
                                    current_p,
                                    (*parent_node.neighbor_iterator).distance});
                                parent_node.neighbor_iterator++;
                                break;
                            }

                            parent_node.neighbor_iterator++;
                        }

                        if (parents[current_index] != current_p)
                        {
                            temp_config_self = state_index(current_index);
                            const auto base_current = base_states[current_index];
                            const auto dist_to_goal =
                                goal_arm.distance(temp_config_self) +
                                rs_arc_distance_weighted_(base_current, goal_base, settings.reverse_weight);

                            if (current.cost <= goal_node.g)
                            {
                                // If this edge could improve the path through this node
                                if (current.cost < current_g + dist_to_goal)
                                {
                                    bool valid = !current_node.invalidList.count(static_cast<int>(current_p));

                                    // If this edge hasn't already been found as invalid
                                    if (valid)
                                    {
                                        if (current_index != current_p)
                                        {
                                            temp_config = state_index(current_p);
                                            valid = validate_motion_wb<Robot, rake, resolution>(
                                                temp_config,
                                                temp_config_self,
                                                base_states[current_p],
                                                base_states[current_index],
                                                environment);
                                        }

                                        // If the edge is valid
                                        if (valid)
                                        {
                                            // Update the node's parent and g value (arm + base)
                                            parents[current_index] = current_p;
                                            const float arm_step = temp_config.distance(temp_config_self);
                                            const float base_step = rs_arc_distance_weighted_(
                                                base_states[current_p], base_states[current_index], settings.reverse_weight);
                                            current_g = parent_node.g + arm_step + base_step;
                                            current_node.g = current_g;
                                        }
                                        else
                                        {
                                            // Found to be invalid, add to necessary sets
                                            parent_node.invalidList.insert(static_cast<int>(current_index));
                                            current_node.invalidList.insert(static_cast<int>(current_p));

                                            (*(parent_node.neighbor_iterator - 1)).distance =
                                                std::numeric_limits<float>::max();
                                            continue;
                                        }
                                    }
                                }
                            }
                            else
                            {
                                break;
                            }
                        }

                        current_node.neighbors.reserve(nodes.size());

                        bool added_neighbors = false;
                        // Iterate through neighbors and add all outgoing neighbors
                        for (auto it = nodes.begin() + current_node.sampleIdx; it != nodes.end(); ++it)
                        {
                            if (it->index == current_index)
                            {
                                current_node.sampleIdx++;
                                continue;
                            }

                            const auto neighbor_index = it->index;
                            temp_config = state_index(neighbor_index);
                            const auto arm_neighbor_distance = temp_config_self.distance(temp_config);
                            const auto base_neighbor_distance = rs_arc_distance_weighted_(
                                base_states[current_index], base_states[neighbor_index], settings.reverse_weight);
                            current_node.sampleIdx = std::max(neighbor_index, current_node.sampleIdx);

                            // f^(c) = g(p) + c^(p,c) + h^(c)
                            const float cost =
                                current_g + arm_neighbor_distance + base_neighbor_distance +
                                goal_arm.distance(temp_config) +
                                rs_arc_distance_weighted_(base_states[neighbor_index], goal_base, settings.reverse_weight);

                            current_node.neighbors.emplace_back(
                                FCITWBRoadmapNode::Neighbor{neighbor_index, cost});
                            current_node.sampleIdx++;
                            added_neighbors = true;
                        }

                        // If any new neighbors were added
                        if (added_neighbors)
                        {
                            pdqsort_branchless(
                                current_node.neighbors.begin(),
                                current_node.neighbors.end(),
                                [](const auto &a, const auto &b) { return a.distance < b.distance; });
                            current_node.neighbor_iterator = current_node.neighbors.begin();

                            open_set.emplace_back(WBQueueEdge{
                                (*current_node.neighbor_iterator).index,
                                current_index,
                                (*current_node.neighbor_iterator).distance});
                            current_node.neighbor_iterator++;
                        }
                    }
                }

                // If we have a solution and just want an initial solution, break
                if (!settings.optimize && parents[1] != std::numeric_limits<unsigned int>::max())
                {
                    break;
                }

                // Batch sample new arm+base states
                for (auto new_samples = 0U;
                     new_samples < settings.batch_size && nodes.size() < settings.max_samples;)
                {
                    // Sample arm in unit hypercube then scale via Robot::scale_configuration
                    auto rng_temp = rng->next();
                    Robot::scale_configuration(rng_temp);

                    // Sample base in unit [0,1] then scale to bounds
                    FloatVector<Robot::base_dimension> unit_base;
                    // Use RNG's distribution for base dimensions
                    std::array<float, Robot::base_dimension> base_unit_arr{};
                    for (std::size_t d = 0; d < Robot::base_dimension; ++d)
                    {
                        base_unit_arr[d] = rng->dist.uniform_01();
                    }
                    unit_base = FloatVector<Robot::base_dimension>(base_unit_arr.data());
                    BaseConfig sampled_base =
                        detail::scale_unit_to_base_config<Robot>(unit_base, base_bounds);

                    // Whole-body validity check at the sampled configuration
                    // Convert rng_temp to a temporary array to reuse as Configuration
                    auto *temp_state_ptr = state_index(static_cast<unsigned int>(nodes.size()));
                    rng_temp.to_array(temp_state_ptr);
                    Configuration sampled_arm = temp_state_ptr;

                    // Whole-body validity check at the sampled configuration
                    if (!validate_motion_wb<Robot, rake, 1>(
                            sampled_arm, sampled_arm, sampled_base, sampled_base, environment))
                    {
                        continue;
                    }

                    // Insert valid state into graph structures
                    auto *state = temp_state_ptr;  // already filled

                    parents.emplace_back(std::numeric_limits<unsigned int>::max());
                    auto &node = nodes.emplace_back(static_cast<unsigned int>(nodes.size()));

                    auto road_node = NNNode<dimension>{node.index, {state}};
                    roadmap.insert(road_node);

                    // Store base state
                    base_states.emplace_back(sampled_base);

                    new_samples++;
                }
            }

            // Recover arm path
            utils::recover_path<Configuration>(parents, state_index, arm_result.path);
            arm_result.cost = nodes[1].g;
            arm_result.nanoseconds = vamp::utils::get_elapsed_nanoseconds(start_time);
            arm_result.iterations = iter;
            arm_result.size.emplace_back(roadmap.size());
            arm_result.size.emplace_back(0);

            // Recover base path using the same parent chain (build indices manually)
            {
                // Build indices path from parents vector
                constexpr const unsigned int s_idx = 0;
                constexpr const unsigned int g_idx = 1;
                std::vector<unsigned int> indices;
                if (parents.size() > g_idx && parents[g_idx] != std::numeric_limits<unsigned int>::max())
                {
                    unsigned int cur = g_idx;
                    while (parents[cur] != std::numeric_limits<unsigned int>::max())
                    {
                        indices.push_back(cur);
                        if (cur == s_idx)
                        {
                            break;
                        }
                        cur = parents[cur];
                    }
                    if (indices.empty() || indices.back() != s_idx)
                    {
                        indices.push_back(s_idx);
                    }
                    std::reverse(indices.begin(), indices.end());
                }

                wb_result.base_path.clear();
                wb_result.base_path.reserve(indices.size());
                for (auto idx : indices)
                {
                    wb_result.base_path.push_back(base_states[idx]);
                }
            }

            // Fill overall timings
            wb_result.nanoseconds = vamp::utils::get_elapsed_nanoseconds(start_time);

            return wb_result;
        }
    };

    // Private helper to compute RS distance with backward penalty
    template <typename Robot, std::size_t rake, std::size_t resolution, typename NeighborParamsT>
    inline float FCITWB<Robot, rake, resolution, NeighborParamsT>::rs_arc_distance_weighted_(
        const BaseConfig &a,
        const BaseConfig &b,
        double backwardPenalty)
    {
        // Clamp penalty
        if (backwardPenalty < 1.0)
        {
            backwardPenalty = 1.0;
        }
        double q0[3] = {
            static_cast<double>(a.config.data[0][0]),
            static_cast<double>(a.config.data[0][1]),
            static_cast<double>(a.config.data[0][2])};
        double q1[3] = {
            static_cast<double>(b.config.data[0][0]),
            static_cast<double>(b.config.data[0][1]),
            static_cast<double>(b.config.data[0][2])};

        // Keep this consistent with MobileBaseConfiguration::arc_distance
        constexpr double turning_radius = Robot::turning_radius;
        vamp::planning::detail::reeds_shepp_gh::ReedsSheppStateSpace rs_space(turning_radius);
        auto path = rs_space.reedsShepp(q0, q1);
        const double weighted_len_scaled =
            vamp::planning::detail::reeds_shepp_gh::ReedsSheppStateSpace::weightedLength(
                path, backwardPenalty);
        return static_cast<float>(weighted_len_scaled * turning_radius);
    }
}  // namespace vamp::planning
