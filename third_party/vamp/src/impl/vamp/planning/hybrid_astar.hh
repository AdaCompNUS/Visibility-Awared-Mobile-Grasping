#pragma once

#include <vector>
#include <array>
#include <unordered_map>
#include <unordered_set>  // Added for the closed set
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <queue>
#include <memory>

#include <vamp/collision/environment.hh>
#include <vamp/robots/fetch/fk.hh>
#include <vamp/planning/base_configuration.hh>
#include <vamp/planning/hybrid_astar_settings.hh>
#include <vamp/planning/base_simplify.hh>

namespace vamp::planning
{
    // Main HybridAStar implementation
    template <typename Robot, std::size_t rake>
    class HybridAStar
    {
    public:
        using BaseConfig = MobileBaseConfiguration<Robot>;
        using Config = HybridAStarConfig<Robot>;

    private:
        struct CostMap2D
        {
            int width;
            int height;
            float resolution;
            float min_x;
            float min_y;
            std::vector<float> costs;

            float getCost(float x, float y) const
            {
                int ix = static_cast<int>(std::floor((x - min_x) / resolution));
                int iy = static_cast<int>(std::floor((y - min_y) / resolution));
                if (ix < 0 || ix >= width || iy < 0 || iy >= height)
                {
                    return std::numeric_limits<float>::infinity();
                }
                return costs[iy * width + ix];
            }
        };

        static typename Robot::Configuration getDefaultConfig()
        {
            std::array<float, Robot::dimension> config_arr;
            std::vector<float> config_values = {0.3f, 1.32f, 1.4f, -0.2f, 1.72f, 0.0f, 1.66f, 0.0f};

            for (size_t i = 0; i < std::min(config_values.size(), (size_t)Robot::dimension); i++)
            {
                config_arr[i] = config_values[i];
            }
            for (size_t i = config_values.size(); i < Robot::dimension; i++)
            {
                config_arr[i] = 0.0f;
            }

            return typename Robot::Configuration(config_arr);
        }

        struct AStarNode
        {
            BaseConfig config;
            std::array<int, 3> grid_pos;
            float g_cost;
            float f_cost;
            size_t parent_index;
            int direction;  // 1 for forward, -1 for backward

            AStarNode()
              : g_cost(std::numeric_limits<float>::max())
              , f_cost(std::numeric_limits<float>::max())
              , parent_index(std::numeric_limits<size_t>::max())
              , direction(1)
            {
            }

            AStarNode(const BaseConfig &cfg, const std::array<int, 3> &grid)
              : config(cfg)
              , grid_pos(grid)
              , g_cost(std::numeric_limits<float>::max())
              , f_cost(std::numeric_limits<float>::max())
              , parent_index(std::numeric_limits<size_t>::max())
              , direction(1)
            {
            }

            bool operator>(const AStarNode &other) const
            {
                return f_cost > other.f_cost;
            }
        };

        struct PathSegment
        {
            std::vector<BaseConfig> points;
            float cost = 0.0f;
        };

        static float normalizeAngle(float angle)
        {
            angle = std::fmod(angle, 2.0f * M_PI);
            if (angle > M_PI)
            {
                angle -= 2.0f * M_PI;
            }
            else if (angle < -M_PI)
            {
                angle += 2.0f * M_PI;
            }
            return angle;
        }

        static std::array<int, 3>
        toGridPosition(const BaseConfig &config, float cell_size, float heading_resolution)
        {
            std::array<int, 3> grid_pos;
            grid_pos[0] = static_cast<int>(std::floor(config.config.data[0][0] / cell_size));
            grid_pos[1] = static_cast<int>(std::floor(config.config.data[0][1] / cell_size));
            float theta = normalizeAngle(config.config.data[0][2]);
            grid_pos[2] = static_cast<int>(std::floor((theta + M_PI) / heading_resolution));
            return grid_pos;
        }

        static std::vector<std::pair<BaseConfig, float>>
        generateMotionPrimitives(const BaseConfig &config, const Config &settings)
        {
            std::vector<std::pair<BaseConfig, float>> primitives;
            std::vector<float> steering_angles = {-settings.steering_angle, 0.0f, settings.steering_angle};
            std::vector<int> directions = {1};
            if (settings.allow_reverse)
            {
                directions.push_back(-1);
            }

            for (int dir : directions)
            {
                for (float phi : steering_angles)
                {
                    BaseConfig next_config = config;
                    float step = settings.motion_resolution;
                    float curvature = std::tan(phi) / settings.wheel_base;

                    next_config.config.data[0][0] += dir * step * std::cos(config.config.data[0][2]);
                    next_config.config.data[0][1] += dir * step * std::sin(config.config.data[0][2]);
                    next_config.config.data[0][2] =
                        normalizeAngle(config.config.data[0][2] + dir * step * curvature);
                    next_config.direction = dir;

                    float cost = step;
                    if (dir == -1)
                    {
                        cost *= settings.reverse_penalty;
                    }
                    if (std::abs(phi) > 1e-6)
                    {
                        cost *= settings.turn_penalty;
                    }

                    primitives.emplace_back(next_config, cost);
                }
            }

            if (settings.allow_inplace_rotation)
            {
                BaseConfig rotate_right_config = config;
                rotate_right_config.config.data[0][2] =
                    normalizeAngle(config.config.data[0][2] - settings.inplace_rotation_step);
                primitives.emplace_back(rotate_right_config, settings.inplace_rotation_cost);

                BaseConfig rotate_left_config = config;
                rotate_left_config.config.data[0][2] =
                    normalizeAngle(config.config.data[0][2] + settings.inplace_rotation_step);
                primitives.emplace_back(rotate_left_config, settings.inplace_rotation_cost);
            }

            return primitives;
        }

        static float calculateHeuristic(
            const BaseConfig &current,
            const BaseConfig &goal,
            const CostMap2D *costmap = nullptr)
        {
            // Use arc_distance_reverse_penalty to properly account for reverse motion cost
            float reeds_shepp_dist = current.arc_distance_reverse_penalty(goal);
            float heuristic = reeds_shepp_dist;

            if (costmap)
            {
                float dx = current.config.data[0][0];
                float dy = current.config.data[0][1];
                float dijkstra_dist = costmap->getCost(dx, dy);
                if (std::isfinite(dijkstra_dist))
                {
                    heuristic = std::max(heuristic, dijkstra_dist);
                }
                // If dijkstra_dist is infinite, it implies the 2D grid cell is unreachable.
                // However, due to conservative inflation, valid continuous configs might fall into these cells.
                // In such cases, we fallback to the Reeds-Shepp distance (already set in `heuristic`)
                // rather than returning infinity, to allow the search to proceed.
            }

            return heuristic;
        }

        struct GridIndexHash
        {
            std::size_t operator()(const std::array<int, 3> &idx) const
            {
                std::size_t h1 = std::hash<int>{}(idx[0]);
                std::size_t h2 = std::hash<int>{}(idx[1]);
                std::size_t h3 = std::hash<int>{}(idx[2]);
                return h1 ^ (h2 << 1) ^ (h3 << 2);
            }
        };

        struct GridIndexEqual
        {
            bool operator()(const std::array<int, 3> &a, const std::array<int, 3> &b) const
            {
                return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
            }
        };

        static CostMap2D buildCostMap(
            const collision::Environment<FloatVector<rake>> &env,
            const Config &config,
            const BaseConfig &start,
            const BaseConfig &goal)
        {
            CostMap2D map;

            // 1. Determine bounds
            float min_x = std::min(start.config.data[0][0], goal.config.data[0][0]);
            float max_x = std::max(start.config.data[0][0], goal.config.data[0][0]);
            float min_y = std::min(start.config.data[0][1], goal.config.data[0][1]);
            float max_y = std::max(start.config.data[0][1], goal.config.data[0][1]);

            for (const auto &pc : env.pointclouds)
            {
                for (size_t i = 0; i < pc.affordances[0].size(); ++i)
                {
                    auto xs = pc.affordances[0][i].to_array();
                    auto ys = pc.affordances[1][i].to_array();
                    for (size_t k = 0; k < xs.size(); ++k)
                    {
                        float x = xs[k];
                        float y = ys[k];
                        if (std::isfinite(x) && std::isfinite(y))
                        {
                            min_x = std::min(min_x, x);
                            max_x = std::max(max_x, x);
                            min_y = std::min(min_y, y);
                            max_y = std::max(max_y, y);
                        }
                    }
                }
            }

            // Padding
            float padding = 5.0f;  // 5 meters padding
            min_x -= padding;
            min_y -= padding;
            max_x += padding;
            max_y += padding;

            map.resolution = config.cell_size;
            map.min_x = std::floor(min_x / map.resolution) * map.resolution;
            map.min_y = std::floor(min_y / map.resolution) * map.resolution;
            map.width = static_cast<int>(std::ceil((max_x - map.min_x) / map.resolution)) + 1;
            map.height = static_cast<int>(std::ceil((max_y - map.min_y) / map.resolution)) + 1;

            // 2. Mark raw occupancy
            std::vector<bool> raw_occupancy(map.width * map.height, false);

            for (const auto &pc : env.pointclouds)
            {
                for (size_t i = 0; i < pc.affordances[0].size(); ++i)
                {
                    auto xs = pc.affordances[0][i].to_array();
                    auto ys = pc.affordances[1][i].to_array();
                    for (size_t k = 0; k < xs.size(); ++k)
                    {
                        float x = xs[k];
                        float y = ys[k];
                        if (std::isfinite(x) && std::isfinite(y))
                        {
                            int ix = static_cast<int>((x - map.min_x) / map.resolution);
                            int iy = static_cast<int>((y - map.min_y) / map.resolution);
                            if (ix >= 0 && ix < map.width && iy >= 0 && iy < map.height)
                            {
                                raw_occupancy[iy * map.width + ix] = true;
                            }
                        }
                    }
                }
            }

            // 3. Efficient Inflation
            std::vector<bool> occupancy = raw_occupancy;
            float obstacle_radius = 0.3;
            obstacle_radius = std::max(obstacle_radius, config.cell_size);
            int inflation_cells = static_cast<int>(std::ceil(obstacle_radius / map.resolution));

            std::vector<int> occupied_indices;
            occupied_indices.reserve(map.width * map.height / 10);
            for (size_t i = 0; i < raw_occupancy.size(); ++i)
            {
                if (raw_occupancy[i])
                {
                    occupied_indices.push_back(i);
                }
            }

            for (int idx : occupied_indices)
            {
                int cy = idx / map.width;
                int cx = idx % map.width;

                for (int dy = -inflation_cells; dy <= inflation_cells; ++dy)
                {
                    for (int dx = -inflation_cells; dx <= inflation_cells; ++dx)
                    {
                        int nx = cx + dx;
                        int ny = cy + dy;
                        if (nx >= 0 && nx < map.width && ny >= 0 && ny < map.height)
                        {
                            occupancy[ny * map.width + nx] = true;
                        }
                    }
                }
            }

            // 4. Dijkstra
            map.costs.assign(map.width * map.height, std::numeric_limits<float>::infinity());
            std::queue<std::pair<int, int>> q;

            int goal_ix = static_cast<int>(std::floor((goal.config.data[0][0] - map.min_x) / map.resolution));
            int goal_iy = static_cast<int>(std::floor((goal.config.data[0][1] - map.min_y) / map.resolution));

            if (goal_ix >= 0 && goal_ix < map.width && goal_iy >= 0 && goal_iy < map.height)
            {
                if (!occupancy[goal_iy * map.width + goal_ix])
                {
                    map.costs[goal_iy * map.width + goal_ix] = 0.0f;
                    q.push({goal_ix, goal_iy});
                }
            }

            int dx[] = {1, -1, 0, 0, 1, 1, -1, -1};
            int dy[] = {0, 0, 1, -1, 1, -1, 1, -1};
            float dists[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.414f, 1.414f, 1.414f, 1.414f};

            while (!q.empty())
            {
                auto [cx, cy] = q.front();
                q.pop();

                float current_cost = map.costs[cy * map.width + cx];

                for (int i = 0; i < 8; ++i)
                {
                    int nx = cx + dx[i];
                    int ny = cy + dy[i];

                    if (nx >= 0 && nx < map.width && ny >= 0 && ny < map.height)
                    {
                        if (!occupancy[ny * map.width + nx])
                        {
                            float new_cost = current_cost + dists[i] * map.resolution;
                            if (new_cost < map.costs[ny * map.width + nx])
                            {
                                map.costs[ny * map.width + nx] = new_cost;
                                q.push({nx, ny});
                            }
                        }
                    }
                }
            }

            return map;
        }

    public:
        static bool plan(
            const BaseConfig &start,
            const BaseConfig &goal,
            const collision::Environment<FloatVector<rake>> &environment,
            const Config &config,
            std::vector<BaseConfig> &path,
            bool smooth_path = true)
        {
            (void)smooth_path;

            auto t_start = std::chrono::high_resolution_clock::now();
            double time_costmap_ms = 0;
            double time_collision_ms = 0;
            double time_heuristic_ms = 0;
            double time_search_ms = 0;  // Total search loop time
            double time_setup_ms = 0;
            double time_validation_ms = 0;
            size_t nodes_expanded = 0;

            auto t_setup_start = std::chrono::high_resolution_clock::now();
            const auto &planning_environment = environment;
            const auto &safety_environment = environment;

            auto t_setup_end = std::chrono::high_resolution_clock::now();
            time_setup_ms = std::chrono::duration<double, std::milli>(t_setup_end - t_setup_start).count();

            const auto arm_config = getDefaultConfig();

            auto t_val_start = std::chrono::high_resolution_clock::now();
            if (!validate_motion_wb<Robot, rake, 1>(
                    arm_config, arm_config, start, start, planning_environment))
            {
                return false;
            }

            if (!validate_motion_wb<Robot, rake, 1>(arm_config, arm_config, goal, goal, planning_environment))
            {
                return false;
            }
            auto t_val_end = std::chrono::high_resolution_clock::now();
            time_validation_ms = std::chrono::duration<double, std::milli>(t_val_end - t_val_start).count();

            // Build 2D CostMap for Dual Heuristic
            auto t_map_start = std::chrono::high_resolution_clock::now();
            CostMap2D costmap = buildCostMap(planning_environment, config, start, goal);
            auto t_map_end = std::chrono::high_resolution_clock::now();
            time_costmap_ms = std::chrono::duration<double, std::milli>(t_map_end - t_map_start).count();

            using PriorityQueue =
                std::priority_queue<AStarNode, std::vector<AStarNode>, std::greater<AStarNode>>;
            PriorityQueue open_queue;
            std::unordered_map<std::array<int, 3>, size_t, GridIndexHash, GridIndexEqual> grid_to_node_index;
            std::vector<AStarNode> node_list;
            std::unordered_set<std::array<int, 3>, GridIndexHash, GridIndexEqual> closed_set;

            auto t_h_start = std::chrono::high_resolution_clock::now();
            float start_h = calculateHeuristic(start, goal, &costmap);
            auto t_h_end = std::chrono::high_resolution_clock::now();
            time_heuristic_ms += std::chrono::duration<double, std::milli>(t_h_end - t_h_start).count();

            AStarNode start_node(start, toGridPosition(start, config.cell_size, config.heading_resolution));
            start_node.g_cost = 0.0f;
            start_node.f_cost = start_h;
            node_list.push_back(start_node);
            grid_to_node_index[start_node.grid_pos] = 0;
            open_queue.push(start_node);

            const int max_iterations = config.max_iterations;
            const float goal_threshold = config.cell_size;
            int iterations = 0;
            bool path_found = false;
            size_t final_node_idx = std::numeric_limits<size_t>::max();

            // DEBUG: Failure Diagnosis
            float min_h_seen = std::numeric_limits<float>::max();
            BaseConfig best_config_seen;
            bool reached_goal_region = false;

            auto t_search_start = std::chrono::high_resolution_clock::now();

            // In HybridAStar::plan function...
            while (!open_queue.empty() && iterations < max_iterations)
            {
                iterations++;
                nodes_expanded++;

                AStarNode current_from_queue = open_queue.top();
                open_queue.pop();

                size_t current_node_idx = grid_to_node_index[current_from_queue.grid_pos];
                AStarNode current_node_in_list =
                    node_list[current_node_idx];  // Get a reference to the canonical node

                // DEBUG: Update best seen
                float current_h = current_node_in_list.f_cost - current_node_in_list.g_cost;
                if (current_h < min_h_seen)
                {
                    min_h_seen = current_h;
                    best_config_seen = current_node_in_list.config;
                }

                if (current_from_queue.f_cost > current_node_in_list.f_cost)
                {
                    continue;
                }

                if (closed_set.count(current_from_queue.grid_pos))
                {
                    continue;
                }

                closed_set.insert(current_from_queue.grid_pos);

                if (current_node_in_list.config.arc_distance(goal) < config.shot_distance)
                {
                    // Use a higher resolution for the shot to be safe.
                    constexpr std::size_t shot_resolution = 64;
                    auto t_col_start = std::chrono::high_resolution_clock::now();
                    bool shot_clear = validate_motion_wb<Robot, rake, shot_resolution>(
                        arm_config, arm_config, current_node_in_list.config, goal, safety_environment);
                    auto t_col_end = std::chrono::high_resolution_clock::now();
                    time_collision_ms +=
                        std::chrono::duration<double, std::milli>(t_col_end - t_col_start).count();

                    if (shot_clear)
                    {
                        // SUCCESS! Found a direct, collision-free path.
                        path_found = true;

                        float total_length = current_node_in_list.config.arc_distance(goal);
                        int steps = std::max(1, static_cast<int>(std::ceil(total_length / config.cell_size)));

                        size_t last_node_idx = current_node_idx;

                        for (int i = 1; i <= steps; ++i)
                        {
                            float t = static_cast<float>(i) / steps;
                            BaseConfig intermediate_config = current_node_in_list.config.interpolate(goal, t);

                            size_t new_idx = node_list.size();
                            // Grid position is not used for final path reconstruction of these nodes
                            AStarNode node(intermediate_config, {});
                            node.parent_index = last_node_idx;
                            node.g_cost = current_node_in_list.g_cost + (total_length * t);
                            node.direction = intermediate_config.direction;

                            node_list.push_back(node);
                            last_node_idx = new_idx;
                        }

                        final_node_idx = last_node_idx;
                        break;  // Exit the search loop.
                    }
                }

                float dx = current_node_in_list.config.config.data[0][0] - goal.config.data[0][0];
                float dy = current_node_in_list.config.config.data[0][1] - goal.config.data[0][1];
                float dtheta = std::abs(
                    normalizeAngle(current_node_in_list.config.config.data[0][2] - goal.config.data[0][2]));

                // Hardcoded tolerances: loose distance (2x threshold), strict angle (0.05 rad)
                if (std::sqrt(dx * dx + dy * dy) < goal_threshold && dtheta < 0.2f)
                {
                    path_found = true;
                    final_node_idx = current_node_idx;
                    break;
                }

                // --- Expansion Logic (also slightly refactored for clarity) ---
                // We expand from the best-known version of the node, which is 'current_node_in_list'.
                auto primitives = generateMotionPrimitives(current_node_in_list.config, config);

                for (auto &[next_config, motion_cost] : primitives)
                {
                    float updated_motion_cost = motion_cost;

                    auto t_col_start = std::chrono::high_resolution_clock::now();

                    // Check for hard collision.
                    bool hard_collision = !validate_motion_wb<Robot, rake, 8>(
                        arm_config,
                        arm_config,
                        current_node_in_list.config,
                        next_config,
                        planning_environment);
                    auto t_col_end = std::chrono::high_resolution_clock::now();
                    time_collision_ms +=
                        std::chrono::duration<double, std::milli>(t_col_end - t_col_start).count();

                    if (hard_collision)
                    {
                        continue;
                    }

                    std::array<int, 3> next_grid_pos =
                        toGridPosition(next_config, config.cell_size, config.heading_resolution);
                    if (closed_set.count(next_grid_pos))
                    {
                        continue;
                    }

                    float new_g_cost = current_node_in_list.g_cost + updated_motion_cost;

                    auto t_h_ex_start = std::chrono::high_resolution_clock::now();
                    float new_h = calculateHeuristic(next_config, goal, &costmap);
                    auto t_h_ex_end = std::chrono::high_resolution_clock::now();
                    time_heuristic_ms +=
                        std::chrono::duration<double, std::milli>(t_h_ex_end - t_h_ex_start).count();

                    auto it = grid_to_node_index.find(next_grid_pos);
                    if (it == grid_to_node_index.end())  // Node has not been seen before
                    {
                        size_t new_idx = node_list.size();
                        grid_to_node_index[next_grid_pos] = new_idx;

                        AStarNode new_node(next_config, next_grid_pos);
                        new_node.g_cost = new_g_cost;
                        new_node.f_cost = new_g_cost + new_h;
                        new_node.parent_index = current_node_idx;
                        new_node.direction = next_config.direction;

                        node_list.push_back(new_node);
                        open_queue.push(new_node);
                    }
                    else  // Node has been seen before, check if this is a better path
                    {
                        size_t existing_node_idx = it->second;
                        if (new_g_cost < node_list[existing_node_idx].g_cost)
                        {
                            AStarNode &existing_node = node_list[existing_node_idx];
                            existing_node.g_cost = new_g_cost;
                            existing_node.f_cost = new_g_cost + new_h;
                            existing_node.parent_index = current_node_idx;
                            existing_node.config = next_config;  // Update config in case of minor variations
                            existing_node.direction = next_config.direction;

                            // Push the updated node to the queue. The old, more expensive one will be ignored
                            // later.
                            open_queue.push(existing_node);
                        }
                    }
                }
            }

            auto t_search_end = std::chrono::high_resolution_clock::now();
            time_search_ms = std::chrono::duration<double, std::milli>(t_search_end - t_search_start).count();
            auto t_end = std::chrono::high_resolution_clock::now();
            double total_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

            std::cout << "--- Hybrid A* Performance Stats ---" << std::endl;
            std::cout << "Total Time: " << total_time_ms << " ms" << std::endl;
            std::cout << "  Setup (Env Copy): " << time_setup_ms << " ms" << std::endl;
            std::cout << "  Start/Goal Validation: " << time_validation_ms << " ms" << std::endl;
            std::cout << "  CostMap Build: " << time_costmap_ms << " ms" << std::endl;
            std::cout << "  Search Loop: " << time_search_ms << " ms" << std::endl;
            std::cout << "    Collision Checks: " << time_collision_ms << " ms" << std::endl;
            std::cout << "    Heuristic Calc: " << time_heuristic_ms << " ms" << std::endl;
            std::cout << "  Nodes Expanded: " << nodes_expanded << std::endl;
            std::cout << "  Iterations: " << iterations << std::endl;
            std::cout << "-----------------------------------" << std::endl;

            // Re-check path_found condition logic after profiling output block
            if (path_found)
            {
                std::vector<BaseConfig> raw_path;
                size_t node_idx = final_node_idx;

                // Check if the final node is the exact goal (from Reeds-Shepp shot)
                // or an approximate goal (from distance threshold)
                bool final_node_is_exact_goal = (node_list[final_node_idx].config.distance(goal) < 1e-6f);

                while (node_idx != std::numeric_limits<size_t>::max())
                {
                    raw_path.push_back(node_list[node_idx].config);
                    raw_path.back().direction = node_list[node_idx].direction;
                    node_idx = node_list[node_idx].parent_index;
                }
                std::reverse(raw_path.begin(), raw_path.end());

                auto simplified_result = simplify_base_path<Robot, rake>(
                    raw_path, planning_environment, config.simplify_settings, getDefaultConfig());
                path = simplified_result.path;
                // path = raw_path;
            }
            // std::cout << "\n[Hybrid A* DEBUG]" << std::endl;
            // if (open_queue.empty())
            // {
            //     std::cout << "Cause: Open Set Empty (No reachable nodes left)" << std::endl;
            // }
            // else if (iterations >= max_iterations)
            // {
            //     std::cout << "Cause: Max Iterations Reached (" << max_iterations << ")" << std::endl;
            // }

            std::cout << "Closest Approach (Heuristic Distance): " << min_h_seen << std::endl;
            std::cout << "Best Config Location: x=" << best_config_seen.config.data[0][0]
                      << ", y=" << best_config_seen.config.data[0][1]
                      << ", theta=" << best_config_seen.config.data[0][2] << std::endl;
            std::cout << "Goal Location: x=" << goal.config.data[0][0] << ", y=" << goal.config.data[0][1]
                      << ", theta=" << goal.config.data[0][2] << std::endl;
            std::cout << "CostMap Value at Best Config: "
                      << costmap.getCost(
                             best_config_seen.config.data[0][0], best_config_seen.config.data[0][1])
                      << std::endl;
            std::cout << "---------------------------------------" << std::endl;

            return path_found;
        }
    };
}  // namespace vamp::planning