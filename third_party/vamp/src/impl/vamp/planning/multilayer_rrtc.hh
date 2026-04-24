#pragma once

#include <memory>
#include <unordered_map>
#include <set>
#include <array>
#include <chrono>
#include <algorithm>
#include <limits>
#include <iostream>
#include <random>

#include <vamp/collision/environment.hh>
#include <vamp/planning/nn.hh>
#include <vamp/planning/plan.hh>
#include <vamp/planning/validate.hh>
#include <vamp/planning/rrtc_settings.hh>
#include <vamp/planning/base_configuration.hh>
#include <vamp/planning/hybrid_astar.hh>
#include <vamp/random/rng.hh>
#include <vamp/utils.hh>
#include <vamp/vector.hh>
#include <vamp/robots/fetch/fk.hh>

namespace vamp::planning
{
    // Define a struct to hold both planning result and base path
    template <typename Robot, std::size_t dimension>
    struct MultilayerPlanningResult
    {
        // Base planning result structure
        struct BaseResult
        {
            std::size_t nanoseconds{0};
            std::size_t iterations{0};
        };

        PlanningResult<dimension> arm_result;
        BaseResult base_result;
        std::vector<MobileBaseConfiguration<Robot>> base_path;
        std::size_t nanoseconds{0};  // Total time for the entire planning process

        // Helper methods to check if planning was successful
        bool isSuccessful() const
        {
            return !arm_result.path.empty();
        }

        // Get a message about the planning result
        std::string getMessage() const
        {
            if (isSuccessful())
            {
                return "Successfully found a path";
            }
            else if (base_path.empty())
            {
                return "Failed to find a base path";
            }
            else
            {
                return "Failed to connect trees";
            }
        }
    };

    template <typename Robot, std::size_t rake, std::size_t resolution>
    struct MultilayerRRTC
    {
        using Configuration = typename Robot::Configuration;
        static constexpr auto dimension = Robot::dimension;
        using BaseConfig = MobileBaseConfiguration<Robot>;
        using RNG = typename vamp::rng::RNG<Robot::dimension>;

        // Define our result type
        using MultilayerResult = MultilayerPlanningResult<Robot, dimension>;

        // Interface for solve method with single goal
        inline static auto solve(
            const Configuration &start,
            const Configuration &goal,
            const BaseConfig &base_start,
            const BaseConfig &base_goal,
            const collision::Environment<FloatVector<rake>> &environment,
            const RRTCSettings &settings,
            typename RNG::Ptr rng) noexcept -> MultilayerPlanningResult<Robot, dimension>
        {
            return solve(
                start,
                std::vector<Configuration>{goal},
                base_start,
                std::vector<BaseConfig>{base_goal},
                environment,
                settings,
                rng);
        }

        // Interface for solve method with multiple goals - optimized version with layer-balanced sampling
        inline static auto solve(
            const Configuration &start,
            const std::vector<Configuration> &goals,
            const BaseConfig &base_start,
            const std::vector<BaseConfig> &base_goals,
            const collision::Environment<FloatVector<rake>> &environment,
            const RRTCSettings &settings,
            typename RNG::Ptr rng) noexcept -> MultilayerPlanningResult<Robot, dimension>
        {
            MultilayerPlanningResult<Robot, dimension> multilayer_result;
            PlanningResult<dimension> &result = multilayer_result.arm_result;
            result.iterations = 0;
            result.cost = 0.0f;

            // Set up standard random number generation
            std::random_device rd;   // Obtain a random number from hardware
            std::mt19937 gen(rd());  // Seed the generator
            // NOTE: We will define the distribution range dynamically before each use

            // Overall planning start time
            auto total_start_time = std::chrono::high_resolution_clock::now();

            // 1. First plan a base path using hybrid A*
            std::vector<BaseConfig> base_path;
            HybridAStarConfig<Robot> base_config;

            // Measure time for hybrid A* base planning
            auto base_planning_start_time = std::chrono::high_resolution_clock::now();

            bool base_plan_success = HybridAStar<Robot, rake>::plan(
                base_start,
                base_goals[0],  // Using the first goal for simplicity
                environment,
                base_config,
                base_path);

            auto base_planning_end_time = std::chrono::high_resolution_clock::now();
            auto base_planning_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          base_planning_end_time - base_planning_start_time)
                                          .count();

            // Store base planning result for tracking
            multilayer_result.base_result.nanoseconds = base_planning_time;
            multilayer_result.base_result.iterations = 0;

            if (!base_plan_success)
            {
                // Return empty path to indicate failure
                auto end_time = std::chrono::high_resolution_clock::now();
                auto total_planning_time =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - total_start_time).count();

                // No RRTC planning was done, so set to 0
                result.nanoseconds = 0;
                multilayer_result.nanoseconds = total_planning_time;

                return multilayer_result;
            }
            // Print number of waypoints in base path
            std::cout << "Number of waypoints in base path: " << base_path.size() << std::endl;

            // Store base path
            multilayer_result.base_path = base_path;

            // Start timing for RRTC portion
            auto rrtc_planning_start_time = std::chrono::high_resolution_clock::now();

            // Number of layers = number of waypoints in the base path
            const int num_layers = base_path.size();

            // DEBUG PRINT 1: Number of layers
            // std::cout << "[Debug MultilayerRRTC] Number of layers from Hybrid A*: " << num_layers <<
            // std::endl;

            // OPTIMIZATION 1: Pre-compute base parameters for each layer
            std::vector<std::array<float, 3>> base_params_by_layer(num_layers);
            for (int i = 0; i < num_layers; ++i)
            {
                base_params_by_layer[i] = {
                    base_path[i].config.data[0][2],  // theta
                    base_path[i].config.data[0][0],  // x
                    base_path[i].config.data[0][1]   // y
                };
            }

            // 2. Set up memory-efficient data structures for RRTC
            // Allocate buffer for configurations
            auto buffer = std::unique_ptr<float, decltype(&free)>(
                vamp::utils::vector_alloc<float, FloatVectorAlignment, FloatVectorWidth>(
                    settings.max_samples * Configuration::num_scalars_rounded),
                &free);

            const auto buffer_index = [&buffer](std::size_t index) -> float *
            { return buffer.get() + index * Configuration::num_scalars_rounded; };

            // Create a single forward tree and a single backward tree
            NN<dimension> start_tree;
            NN<dimension> goal_tree;

            // Track parent-child relationships and layer information
            std::vector<std::size_t> parents(settings.max_samples);
            std::vector<int> node_layers(settings.max_samples);
            std::vector<bool> in_start_tree(settings.max_samples);

            // OPTIMIZATION 2: Create indexed structure to quickly find nodes by layer
            std::vector<std::vector<std::size_t>> forward_nodes_by_layer(num_layers);
            std::vector<std::vector<std::size_t>> backward_nodes_by_layer(num_layers);

            // Initialize with free index after start and goals
            std::size_t free_index = 1 + goals.size();

            // Add start node to forward tree with layer 0
            start.to_array(buffer_index(0));
            start_tree.insert(NNNode<dimension>{0, {buffer_index(0)}});
            parents[0] = 0;      // Self-parent for root
            node_layers[0] = 0;  // Layer 0
            in_start_tree[0] = true;
            forward_nodes_by_layer[0].push_back(0);

            // Add goal nodes to backward tree with layer (num_layers - 1)
            for (size_t i = 0; i < goals.size(); ++i)
            {
                const size_t goal_idx = i + 1;
                goals[i].to_array(buffer_index(goal_idx));
                goal_tree.insert(NNNode<dimension>{goal_idx, {buffer_index(goal_idx)}});
                parents[goal_idx] = goal_idx;            // Self-parent for root
                node_layers[goal_idx] = num_layers - 1;  // Last layer
                in_start_tree[goal_idx] = false;
                backward_nodes_by_layer[num_layers - 1].push_back(goal_idx);
            }

            // 3. Main RRTC loop with layer-balanced sampling
            bool trees_connected = false;
            std::size_t connection_forward_idx = 0;
            std::size_t connection_backward_idx = 0;

            // Track the last used layer to avoid redundant base parameter updates
            int last_layer_used = -1;

            for (int iter = 0;
                 iter < settings.max_iterations && !trees_connected && free_index < settings.max_samples;
                 ++iter)
            {
                result.iterations++;

                // Decide which tree to extend (forward or backward)
                bool extend_forward = (iter % 2 == 0);

                // --- MODIFIED Layer Sampling Logic ---
                int sampled_layer = -1;
                int target_layer = -1;
                std::vector<int> valid_sampling_layers;

                if (extend_forward)
                {
                    // Find non-empty layers eligible for forward expansion (layers 0 to num_layers - 2)
                    for (int l = 0; l < num_layers - 1; ++l)
                    {
                        if (!forward_nodes_by_layer[l].empty())
                        {
                            valid_sampling_layers.push_back(l);
                        }
                    }
                    if (valid_sampling_layers.empty())
                    {
                        // No valid layer to expand from in the forward direction, try backward next iteration
                        continue;
                    }
                    // Randomly select a layer to sample from
                    std::uniform_int_distribution<int> dist(0, valid_sampling_layers.size() - 1);
                    sampled_layer = valid_sampling_layers[dist(gen)];
                    target_layer = sampled_layer + 1;
                }
                else  // extend_backward
                {
                    // Find non-empty layers eligible for backward expansion (layers 1 to num_layers - 1)
                    for (int l = 1; l < num_layers; ++l)
                    {
                        if (!backward_nodes_by_layer[l].empty())
                        {
                            valid_sampling_layers.push_back(l);
                        }
                    }
                    if (valid_sampling_layers.empty())
                    {
                        // No valid layer to expand from in the backward direction, try forward next iteration
                        continue;
                    }
                    // Randomly select a layer to sample from
                    std::uniform_int_distribution<int> dist(0, valid_sampling_layers.size() - 1);
                    sampled_layer = valid_sampling_layers[dist(gen)];
                    target_layer = sampled_layer - 1;
                }
                // --- End MODIFIED Layer Sampling ---

                // Sample a random configuration as usual
                auto random_config = rng->next();
                Robot::scale_configuration(random_config);

                typename Robot::ConfigurationBuffer random_array;
                random_config.to_array(random_array.data());

                // NEW: Find nearest node in the sampled layer
                float min_distance = std::numeric_limits<float>::max();
                std::size_t nearest_node_idx = 0;

                const auto &layer_nodes = extend_forward ? forward_nodes_by_layer[sampled_layer] :
                                                           backward_nodes_by_layer[sampled_layer];

                for (std::size_t idx : layer_nodes)
                {
                    float *node_config = buffer_index(idx);
                    float dist = 0.0f;
                    for (std::size_t j = 0; j < dimension; ++j)
                    {
                        float diff = random_array[j] - node_config[j];
                        dist += diff * diff;
                    }
                    dist = std::sqrt(dist);

                    if (dist < min_distance)
                    {
                        min_distance = dist;
                        nearest_node_idx = idx;
                    }
                }

                // Get the nearest configuration as a vector
                const auto nearest_configuration = Configuration(buffer_index(nearest_node_idx));

                // Calculate extension vector (standard RRTC extension)
                auto extension_vector = random_config - nearest_configuration;
                bool reach = min_distance < settings.range;
                if (!reach)
                {
                    extension_vector = extension_vector * (settings.range / min_distance);
                }

                const auto new_configuration = nearest_configuration + extension_vector;

                // Validate the extension
                bool is_valid_motion = false;
                if (extend_forward)
                {
                    // For the forward tree, the motion is from the existing node to the new node.
                    // The base also moves from the sampled layer to the target layer.
                    is_valid_motion = validate_motion_wb<Robot, rake, resolution>(
                        nearest_configuration,     // from arm config
                        new_configuration,         // to arm config
                        base_path[sampled_layer],  // from base config (e.g., layer i)
                        base_path[target_layer],   // to base config (e.g., layer i+1)
                        environment);
                }
                else  // extend_backward
                {
                    // For the backward tree, the motion segment in the final path runs
                    // FROM the new node TO the existing node. We must validate in that direction.
                    // The base must also move forward, from target_layer to sampled_layer.
                    is_valid_motion = validate_motion_wb<Robot, rake, resolution>(
                        new_configuration,         // from arm config (at target_layer)
                        nearest_configuration,     // to arm config (at sampled_layer)
                        base_path[target_layer],   // from base config (e.g., layer i-1)
                        base_path[sampled_layer],  // to base config (e.g., layer i)
                        environment);
                }

                if (is_valid_motion)
                {
                    // Add the new node to the tree
                    float *new_config_buffer = buffer_index(free_index);
                    auto new_configuration = nearest_configuration + extension_vector;
                    new_configuration.to_array(new_config_buffer);

                    if (extend_forward)
                    {
                        start_tree.insert(NNNode<dimension>{free_index, {new_config_buffer}});
                        forward_nodes_by_layer[target_layer].push_back(free_index);
                    }
                    else
                    {
                        goal_tree.insert(NNNode<dimension>{free_index, {new_config_buffer}});
                        backward_nodes_by_layer[target_layer].push_back(free_index);
                    }

                    parents[free_index] = nearest_node_idx;
                    node_layers[free_index] = target_layer;
                    in_start_tree[free_index] = extend_forward;

                    // Look for nodes in the opposite tree with the same layer (target layer)
                    const auto &opposite_layer_nodes = extend_forward ?
                                                           backward_nodes_by_layer[target_layer] :
                                                           forward_nodes_by_layer[target_layer];

                    if (!opposite_layer_nodes.empty())
                    {
                        // Find nearest node in opposite tree with matching layer
                        float min_connect_distance = std::numeric_limits<float>::max();
                        std::size_t nearest_opposite_idx = 0;
                        bool found_connection_candidate = false;

                        for (std::size_t idx : opposite_layer_nodes)
                        {
                            float *opposite_config = buffer_index(idx);
                            float dist = 0.0f;
                            for (std::size_t j = 0; j < dimension; ++j)
                            {
                                float diff = new_config_buffer[j] - opposite_config[j];
                                dist += diff * diff;
                            }
                            dist = std::sqrt(dist);

                            if (dist < min_connect_distance)
                            {
                                min_connect_distance = dist;
                                nearest_opposite_idx = idx;
                                found_connection_candidate = true;
                            }
                        }

                        if (found_connection_candidate && min_connect_distance <= settings.connect_distance)
                        {
                            // Use incremental connection strategy
                            const auto opposite_configuration =
                                Configuration(buffer_index(nearest_opposite_idx));
                            auto connect_vector = opposite_configuration - new_configuration;
                            const float connect_distance = min_connect_distance;

                            // Try to connect the trees using multiple smaller steps
                            const std::size_t n_extensions = std::ceil(connect_distance / settings.range);
                            const float increment_length =
                                connect_distance / static_cast<float>(n_extensions);
                            auto increment = connect_vector * (1.0F / static_cast<float>(n_extensions));

                            std::size_t i_extension = 0;
                            auto prior = new_configuration;
                            std::size_t last_add_idx = free_index;

                            for (; i_extension < n_extensions && free_index < settings.max_samples;
                                 ++i_extension)
                            {
                                // For the final step, we don't need to validate or add a new node
                                if (i_extension == n_extensions - 1)
                                {
                                    trees_connected = true;
                                    break;
                                }

                                // For intermediate steps, validate and add nodes
                                if (validate_motion_wb<Robot, rake, resolution>(
                                        prior,
                                        prior + increment,
                                        base_path[target_layer],
                                        base_path[target_layer],
                                        environment))
                                {
                                    // Add intermediate connection node
                                    auto next = prior + increment;
                                    float *next_index = buffer_index(free_index + 1);
                                    next.to_array(next_index);

                                    if (extend_forward)
                                    {
                                        start_tree.insert(NNNode<dimension>{free_index + 1, {next_index}});
                                        forward_nodes_by_layer[target_layer].push_back(free_index + 1);
                                    }
                                    else
                                    {
                                        goal_tree.insert(NNNode<dimension>{free_index + 1, {next_index}});
                                        backward_nodes_by_layer[target_layer].push_back(free_index + 1);
                                    }

                                    parents[free_index + 1] = last_add_idx;
                                    node_layers[free_index + 1] = target_layer;
                                    in_start_tree[free_index + 1] = extend_forward;

                                    last_add_idx = free_index + 1;
                                    free_index++;
                                    prior = next;
                                }
                                else
                                {
                                    // Connection failed
                                    break;
                                }
                            }

                            if (i_extension == n_extensions - 1)
                            {
                                // Trees connected!
                                trees_connected = true;
                                connection_forward_idx = extend_forward ? last_add_idx : nearest_opposite_idx;
                                connection_backward_idx =
                                    extend_forward ? nearest_opposite_idx : last_add_idx;
                                // std::cout << "[Debug MultilayerRRTC] Trees connected at layer "
                                //           << target_layer << " with indices: " << connection_forward_idx
                                //           << " (forward), " << connection_backward_idx << " (backward)"
                                //           << std::endl;
                                break;
                            }
                        }
                    }

                    free_index++;
                }
            }

            // DEBUG PRINT 2 & 3: Nodes per layer and Total nodes vs Iterations
            std::cout << "[Debug MultilayerRRTC] RRTC Loop Finished." << std::endl;
            std::cout << "[Debug MultilayerRRTC] Total Iterations: " << result.iterations << std::endl;
            // Note: free_index includes start (1) + goal(s) nodes. Added nodes = free_index - (1 +
            // goals.size())
            std::cout << "[Debug MultilayerRRTC] Total Nodes Allocated (free_index): " << free_index
                      << " (includes start + goals)" << std::endl;
            std::cout << "[Debug MultilayerRRTC] Nodes added during iterations: "
                      << (free_index - (1 + goals.size())) << std::endl;
            std::cout << "[Debug MultilayerRRTC] Nodes per layer:" << std::endl;
            for (int i = 0; i < num_layers; ++i)
            {
                std::cout << "  Layer " << i << ": Forward=" << forward_nodes_by_layer[i].size()
                          << ", Backward=" << backward_nodes_by_layer[i].size() << std::endl;
            }
            // --- End Debug Prints ---

            // 4. If trees connected, extract the solution path
            if (trees_connected)
            {
                // Extract the forward path
                std::vector<std::pair<int, std::size_t>> path_nodes;  // (layer, node_idx)
                std::size_t current = connection_forward_idx;

                while (true)
                {
                    path_nodes.push_back({node_layers[current], current});
                    if (parents[current] == current)
                    {
                        break;  // Reached root
                    }
                    current = parents[current];
                }

                std::reverse(path_nodes.begin(), path_nodes.end());

                // Extract the backward path
                current = connection_backward_idx;

                while (true)
                {
                    path_nodes.push_back({node_layers[current], current});
                    if (parents[current] == current)
                    {
                        break;  // Reached root
                    }
                    current = parents[current];
                }

                // Build complete path by layer
                std::vector<Configuration> configurations_by_layer(num_layers);
                std::vector<bool> layer_covered(num_layers, false);

                // Fill in the layers we have
                for (const auto &[layer, node_idx] : path_nodes)
                {
                    if (!layer_covered[layer])
                    {
                        configurations_by_layer[layer] = Configuration(buffer_index(node_idx));
                        layer_covered[layer] = true;
                    }
                }

                // Fill in any missing layers by interpolation
                for (int i = 0; i < num_layers; ++i)
                {
                    if (!layer_covered[i])
                    {
                        // Find previous and next valid layers
                        int prev_valid = -1;
                        for (int j = i - 1; j >= 0; --j)
                        {
                            if (layer_covered[j])
                            {
                                prev_valid = j;
                                break;
                            }
                        }

                        int next_valid = -1;
                        for (int j = i + 1; j < num_layers; ++j)
                        {
                            if (layer_covered[j])
                            {
                                next_valid = j;
                                break;
                            }
                        }

                        if (prev_valid != -1 && next_valid != -1)
                        {
                            // Interpolate between prev_valid and next_valid
                            float t = static_cast<float>(i - prev_valid) / (next_valid - prev_valid);

                            const Configuration &prev_config = configurations_by_layer[prev_valid];
                            const Configuration &next_config = configurations_by_layer[next_valid];

                            Configuration interp_config;
                            for (size_t j = 0; j < dimension; ++j)
                            {
                                interp_config[j] = prev_config[j] * (1.0f - t) + next_config[j] * t;
                            }

                            configurations_by_layer[i] = interp_config;
                        }
                        else if (prev_valid != -1)
                        {
                            // Use previous configuration
                            configurations_by_layer[i] = configurations_by_layer[prev_valid];
                        }
                        else if (next_valid != -1)
                        {
                            // Use next configuration
                            configurations_by_layer[i] = configurations_by_layer[next_valid];
                        }
                    }
                }

                // Build the final path
                Path<dimension> complete_path;
                for (int i = 0; i < num_layers; ++i)
                {
                    complete_path.push_back(configurations_by_layer[i]);
                }

                // Calculate path cost
                float path_cost = 0.0f;
                for (size_t i = 1; i < complete_path.size(); ++i)
                {
                    path_cost += complete_path[i].distance(complete_path[i - 1]);
                }

                // Set results
                result.path = complete_path;
                result.cost = path_cost;
                result.size.push_back(start_tree.size());
                result.size.push_back(goal_tree.size());
            }
            else
            {
                // Failed to find a path - return empty path
                result.size.push_back(start_tree.size());
                result.size.push_back(goal_tree.size());
            }

            auto end_time = std::chrono::high_resolution_clock::now();

            // Calculate RRTC planning time (excludes A*)
            auto rrtc_planning_time =
                std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - rrtc_planning_start_time)
                    .count();

            std::cout << "RRTC planning time: " << rrtc_planning_time / 1000 << " μs" << std::endl;

            // Calculate total planning time
            auto total_planning_time =
                std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - total_start_time).count();

            std::cout << "Total planning time: " << total_planning_time / 1000 << " μs" << std::endl;

            // Store timing results
            result.nanoseconds = rrtc_planning_time;              // Only RRTC portion
            multilayer_result.nanoseconds = total_planning_time;  // Total time

            return multilayer_result;
        }
    };
}  // namespace vamp::planning
