#include <vamp/bindings/common.hh>
#include <vamp/bindings/init.hh>
#include <vamp/robots/fetch.hh>
#include <vamp/planning/multilayer_rrtc.hh>
#include <vamp/planning/hybrid_astar.hh>
#include <vamp/planning/base_configuration.hh>
#include <vamp/planning/hybrid_astar_settings.hh>
#include <vamp/planning/whole_body_simplify.hh>
#include <vamp/planning/fcit_wb.hh>
#include <vamp/collision/capt.hh>
#include <vector>
#include <iostream>
#include <iterator>

void vamp::binding::init_fetch(nanobind::module_ &pymodule)
{
    static constexpr const std::size_t rake = vamp::FloatVectorWidth;

    namespace nb = nanobind;
    using namespace nb::literals;

    // Initialize the robot
    auto submodule = vamp::binding::init_robot<vamp::robots::Fetch>(pymodule);

    // Add base parameter bindings
    submodule.def(
        "get_base_theta",
        []() { return vamp::robots::Fetch::getBaseTheta(); },
        "Get the base rotation theta parameter (in radians).");

    submodule.def(
        "get_base_x", []() { return vamp::robots::Fetch::getBaseX(); }, "Get the base x position parameter.");

    submodule.def(
        "get_base_y", []() { return vamp::robots::Fetch::getBaseY(); }, "Get the base y position parameter.");

    submodule.def(
        "set_base_params",
        [](double theta, double x, double y) { vamp::robots::Fetch::setBaseParams(theta, x, y); },
        "theta"_a,
        "x"_a,
        "y"_a,
        "Set the base parameters (theta in radians, x, y).");

    submodule.def(
        "reset_base_params",
        []() { vamp::robots::Fetch::resetBaseParams(); },
        "Reset the base parameters to their default values.");

    // Add multilayer_rrtc binding
    using MultilayerRRTC = vamp::planning::MultilayerRRTC<vamp::robots::Fetch, vamp::binding::rake, 32>;
    using BaseConfig = vamp::planning::MobileBaseConfiguration<vamp::robots::Fetch>;
    using MultilayerResult =
        vamp::planning::MultilayerPlanningResult<vamp::robots::Fetch, vamp::robots::Fetch::dimension>;
    using HybridAStarConfig = vamp::planning::HybridAStarConfig<vamp::robots::Fetch>;
    using HybridAStar = vamp::planning::HybridAStar<vamp::robots::Fetch, vamp::binding::rake>;
    using FCITWB = vamp::planning::FCITWB<vamp::robots::Fetch, vamp::binding::rake, vamp::robots::Fetch::resolution>;
    using FCITNeighborSettings = vamp::planning::RoadmapSettings<vamp::planning::FCITStarNeighborParams>;

    // Add MobileBaseConfiguration class binding
    nb::class_<BaseConfig>(submodule, "MobileBaseConfiguration")
        .def(nb::init<>())
        .def(nb::init<const std::array<float, 3> &>())
        .def_prop_ro("config", &BaseConfig::get_config_array)
        .def("distance", &BaseConfig::distance);

    // Add HybridAStarConfig class binding
    nb::class_<HybridAStarConfig>(submodule, "HybridAStarConfig")
        .def(nb::init<>())
        .def_rw("cell_size", &HybridAStarConfig::cell_size)
        .def_rw("heading_resolution", &HybridAStarConfig::heading_resolution)
        .def_rw("num_headings", &HybridAStarConfig::num_headings)
        .def_rw("steering_angle", &HybridAStarConfig::steering_angle)
        .def_rw("point_radius", &HybridAStarConfig::point_radius)
        .def_rw("reverse_penalty", &HybridAStarConfig::reverse_penalty)
        .def_rw("turn_penalty", &HybridAStarConfig::turn_penalty)
        .def_rw("safety_margin_radius", &HybridAStarConfig::safety_margin_radius)
        .def_rw("safety_margin_penalty", &HybridAStarConfig::safety_margin_penalty);

    // Define the BaseResult class for Python
    nb::class_<MultilayerResult::BaseResult>(submodule, "BaseResult")
        .def(nb::init<>())
        .def_rw(
            "nanoseconds",
            &MultilayerResult::BaseResult::nanoseconds,
            "Time taken for base planning in nanoseconds")
        .def_rw(
            "iterations",
            &MultilayerResult::BaseResult::iterations,
            "Number of iterations taken for base planning");

    // Define a MultilayerPlanningResult class for Python
    nb::class_<MultilayerResult>(submodule, "MultilayerPlanningResult")
        .def(nb::init<>())
        .def_rw("arm_result", &MultilayerResult::arm_result, "The planning result for the arm component")
        .def_rw("base_result", &MultilayerResult::base_result, "The planning result for the base component")
        .def_rw("base_path", &MultilayerResult::base_path, "The path for the base component")
        .def_rw(
            "nanoseconds",
            &MultilayerResult::nanoseconds,
            "Total time taken for the entire planning process in nanoseconds")
        .def("is_successful", &MultilayerResult::isSuccessful, "Check if planning was successful")
        .def(
            "get_message",
            &MultilayerResult::getMessage,
            "Get a descriptive message about the planning result");

    // Add HybridAStar class binding with static plan method
    nb::class_<HybridAStar>(submodule, "HybridAStar")
        .def_static(
            "plan",
            [](const BaseConfig &start,
               const BaseConfig &goal,
               const vamp::collision::Environment<float> &environment,
               const HybridAStarConfig &config,
               nb::list path)
            {
                // Convert environment to the expected type
                vamp::collision::Environment<vamp::FloatVector<rake>> env_converted(environment);
                // Copy relevant data from environment to env_converted
                // This is a simplified conversion - in a real implementation, you would need to copy all
                // relevant data

                std::vector<BaseConfig> cpp_path;
                bool result = HybridAStar::plan(start, goal, env_converted, config, cpp_path);

                // Convert C++ vector to Python list
                for (const auto &config : cpp_path)
                {
                    path.append(config);
                }

                return result;
            },
            "start"_a,
            "goal"_a,
            "environment"_a,
            "config"_a,
            "path"_a);

    // Updated multilayer_rrtc binding with single goal
    submodule.def(
        "multilayer_rrtc",
        [](const std::vector<float> &start,
           const std::vector<float> &goal,
           const std::vector<float> &base_start,
           const std::vector<float> &base_goal,
           const vamp::collision::Environment<float> &environment,
           const vamp::planning::RRTCSettings &settings,
           typename vamp::rng::RNG<vamp::robots::Fetch::dimension>::Ptr rng) -> MultilayerResult
        {
            // Convert vectors to arrays
            std::array<float, vamp::robots::Fetch::dimension> start_arr;
            std::array<float, vamp::robots::Fetch::dimension> goal_arr;
            std::array<float, 3> base_start_arr;
            std::array<float, 3> base_goal_arr;

            // Copy values, with bounds checking
            for (size_t i = 0; i < std::min(start.size(), (size_t)vamp::robots::Fetch::dimension); i++)
            {
                start_arr[i] = start[i];
            }
            for (size_t i = start.size(); i < vamp::robots::Fetch::dimension; i++)
            {
                start_arr[i] = 0.0f;  // Zero-pad if input is too short
            }

            for (size_t i = 0; i < std::min(goal.size(), (size_t)vamp::robots::Fetch::dimension); i++)
            {
                goal_arr[i] = goal[i];
            }
            for (size_t i = goal.size(); i < vamp::robots::Fetch::dimension; i++)
            {
                goal_arr[i] = 0.0f;  // Zero-pad if input is too short
            }

            for (size_t i = 0; i < std::min(base_start.size(), (size_t)3); i++)
            {
                base_start_arr[i] = base_start[i];
            }
            for (size_t i = base_start.size(); i < 3; i++)
            {
                base_start_arr[i] = 0.0f;  // Zero-pad if input is too short
            }

            for (size_t i = 0; i < std::min(base_goal.size(), (size_t)3); i++)
            {
                base_goal_arr[i] = base_goal[i];
            }
            for (size_t i = base_goal.size(); i < 3; i++)
            {
                base_goal_arr[i] = 0.0f;  // Zero-pad if input is too short
            }

            // Create base configurations
            BaseConfig base_start_config(base_start_arr);
            BaseConfig base_goal_config(base_goal_arr);

            // Create arm configurations
            vamp::robots::Fetch::Configuration start_config(start_arr);
            vamp::robots::Fetch::Configuration goal_config(goal_arr);

            // Convert environment to the expected type
            vamp::collision::Environment<vamp::FloatVector<vamp::binding::rake>> env_converted(environment);

            // Call the multilayer_rrtc solver
            return MultilayerRRTC::solve(
                start_config, goal_config, base_start_config, base_goal_config, env_converted, settings, rng);
        },
        "start"_a,
        "goal"_a,
        "base_start"_a,
        "base_goal"_a,
        "environment"_a,
        "settings"_a,
        "rng"_a,
        "Solve the motion planning problem with MultilayerRRTC, which plans for both base and arm.");

    // Updated multilayer_rrtc binding with multiple goals
    submodule.def(
        "multilayer_rrtc",
        [](const std::vector<float> &start,
           const std::vector<std::vector<float>> &goals,
           const std::vector<float> &base_start,
           const std::vector<std::vector<float>> &base_goals,
           const vamp::collision::Environment<float> &environment,
           const vamp::planning::RRTCSettings &settings,
           typename vamp::rng::RNG<vamp::robots::Fetch::dimension>::Ptr rng) -> MultilayerResult
        {
            // Convert start vectors to arrays
            std::array<float, vamp::robots::Fetch::dimension> start_arr;
            std::array<float, 3> base_start_arr;

            // Copy values, with bounds checking
            for (size_t i = 0; i < std::min(start.size(), (size_t)vamp::robots::Fetch::dimension); i++)
            {
                start_arr[i] = start[i];
            }
            for (size_t i = start.size(); i < vamp::robots::Fetch::dimension; i++)
            {
                start_arr[i] = 0.0f;  // Zero-pad if input is too short
            }

            for (size_t i = 0; i < std::min(base_start.size(), (size_t)3); i++)
            {
                base_start_arr[i] = base_start[i];
            }
            for (size_t i = base_start.size(); i < 3; i++)
            {
                base_start_arr[i] = 0.0f;  // Zero-pad if input is too short
            }

            // Create base configurations
            BaseConfig base_start_config(base_start_arr);
            std::vector<BaseConfig> base_goal_configs;
            base_goal_configs.reserve(base_goals.size());

            for (const auto &base_goal : base_goals)
            {
                std::array<float, 3> base_goal_arr;

                // Copy values, with bounds checking
                for (size_t i = 0; i < std::min(base_goal.size(), (size_t)3); i++)
                {
                    base_goal_arr[i] = base_goal[i];
                }
                for (size_t i = base_goal.size(); i < 3; i++)
                {
                    base_goal_arr[i] = 0.0f;  // Zero-pad if input is too short
                }

                base_goal_configs.emplace_back(base_goal_arr);
            }

            // Create arm configurations
            vamp::robots::Fetch::Configuration start_config(start_arr);
            std::vector<vamp::robots::Fetch::Configuration> goal_configs;
            goal_configs.reserve(goals.size());

            for (const auto &goal : goals)
            {
                std::array<float, vamp::robots::Fetch::dimension> goal_arr;

                // Copy values, with bounds checking
                for (size_t i = 0; i < std::min(goal.size(), (size_t)vamp::robots::Fetch::dimension); i++)
                {
                    goal_arr[i] = goal[i];
                }
                for (size_t i = goal.size(); i < vamp::robots::Fetch::dimension; i++)
                {
                    goal_arr[i] = 0.0f;  // Zero-pad if input is too short
                }

                goal_configs.emplace_back(goal_arr);
            }

            // Convert environment to the expected type
            vamp::collision::Environment<vamp::FloatVector<vamp::binding::rake>> env_converted(environment);

            // Call the multilayer_rrtc solver
            return MultilayerRRTC::solve(
                start_config,
                goal_configs,
                base_start_config,
                base_goal_configs,
                env_converted,
                settings,
                rng);
        },
        "start"_a,
        "goals"_a,
        "base_start"_a,
        "base_goals"_a,
        "environment"_a,
        "settings"_a,
        "rng"_a,
        "Solve the motion planning problem with MultilayerRRTC with multiple goals, which plans for both "
        "base and arm.");

    // Whole-body FCIT bindings (plan arm and base together)
    submodule.def(
        "fcit_wb",
        [](const std::vector<float> &start,
           const std::vector<float> &goal,
           const std::vector<float> &base_start,
           const std::vector<float> &base_goal,
           const vamp::collision::Environment<float> &environment,
           const FCITNeighborSettings &settings,
           typename vamp::rng::RNG<vamp::robots::Fetch::dimension>::Ptr rng,
           float x_min,
           float x_max,
           float y_min,
           float y_max,
           float theta_min,
           float theta_max) -> vamp::planning::WholeBodyPlanningResult<vamp::robots::Fetch>
        {
            // Convert arm start/goal
            std::array<float, vamp::robots::Fetch::dimension> start_arr{};
            std::array<float, vamp::robots::Fetch::dimension> goal_arr{};
            for (size_t i = 0; i < std::min(start.size(), (size_t)vamp::robots::Fetch::dimension); ++i)
                start_arr[i] = start[i];
            for (size_t i = 0; i < std::min(goal.size(), (size_t)vamp::robots::Fetch::dimension); ++i)
                goal_arr[i] = goal[i];

            vamp::robots::Fetch::Configuration arm_start(start_arr);
            vamp::robots::Fetch::Configuration arm_goal(goal_arr);

            // Convert base start/goal
            std::array<float, 3> base_start_arr{0.f, 0.f, 0.f};
            std::array<float, 3> base_goal_arr{0.f, 0.f, 0.f};
            for (size_t i = 0; i < std::min(base_start.size(), (size_t)3); ++i)
                base_start_arr[i] = base_start[i];
            for (size_t i = 0; i < std::min(base_goal.size(), (size_t)3); ++i)
                base_goal_arr[i] = base_goal[i];

            BaseConfig base_start_cfg(base_start_arr);
            BaseConfig base_goal_cfg(base_goal_arr);

            // Environment conversion
            vamp::collision::Environment<vamp::FloatVector<vamp::binding::rake>> env_converted(environment);

            // Bounds
            vamp::planning::BaseBounds bounds{.x_min = x_min,
                                              .x_max = x_max,
                                              .y_min = y_min,
                                              .y_max = y_max,
                                              .theta_min = theta_min,
                                              .theta_max = theta_max};

            // Solve
            auto res = FCITWB::solve(
                arm_start, arm_goal, base_start_cfg, base_goal_cfg, env_converted, settings, bounds, rng);

            // Pack into WholeBodyPlanningResult
            vamp::planning::WholeBodyPlanningResult<vamp::robots::Fetch> wb;
            wb.arm_result = res.arm_result;
            wb.base_path = std::move(res.base_path);
            return wb;
        },
        "start"_a,
        "goal"_a,
        "base_start"_a,
        "base_goal"_a,
        "environment"_a,
        "settings"_a,
        "rng"_a,
        "x_min"_a,
        "x_max"_a,
        "y_min"_a,
        "y_max"_a,
        "theta_min"_a = -static_cast<float>(M_PI),
        "theta_max"_a = static_cast<float>(M_PI),
        "Solve the whole-body motion planning problem with FCIT*, sampling arm and base together.");

    submodule.def(
        "fcit_wb",
        [](const std::vector<float> &start,
           const std::vector<std::vector<float>> &goals,
           const std::vector<float> &base_start,
           const std::vector<std::vector<float>> &base_goals,
           const vamp::collision::Environment<float> &environment,
           const FCITNeighborSettings &settings,
           typename vamp::rng::RNG<vamp::robots::Fetch::dimension>::Ptr rng,
           float x_min,
           float x_max,
           float y_min,
           float y_max,
           float theta_min,
           float theta_max) -> vamp::planning::WholeBodyPlanningResult<vamp::robots::Fetch>
        {
            // Convert arm start
            std::array<float, vamp::robots::Fetch::dimension> start_arr{};
            for (size_t i = 0; i < std::min(start.size(), (size_t)vamp::robots::Fetch::dimension); ++i)
                start_arr[i] = start[i];
            vamp::robots::Fetch::Configuration arm_start(start_arr);

            // Convert arm goals
            std::vector<vamp::robots::Fetch::Configuration> arm_goals;
            arm_goals.reserve(goals.size());
            for (const auto &g : goals)
            {
                std::array<float, vamp::robots::Fetch::dimension> g_arr{};
                for (size_t i = 0; i < std::min(g.size(), (size_t)vamp::robots::Fetch::dimension); ++i)
                    g_arr[i] = g[i];
                arm_goals.emplace_back(g_arr);
            }

            // Convert base start
            std::array<float, 3> base_start_arr{0.f, 0.f, 0.f};
            for (size_t i = 0; i < std::min(base_start.size(), (size_t)3); ++i)
                base_start_arr[i] = base_start[i];
            BaseConfig base_start_cfg(base_start_arr);

            // Convert base goals
            std::vector<BaseConfig> base_goal_cfgs;
            base_goal_cfgs.reserve(base_goals.size());
            for (const auto &bg : base_goals)
            {
                std::array<float, 3> bg_arr{0.f, 0.f, 0.f};
                for (size_t i = 0; i < std::min(bg.size(), (size_t)3); ++i)
                    bg_arr[i] = bg[i];
                base_goal_cfgs.emplace_back(bg_arr);
            }

            // Environment conversion
            vamp::collision::Environment<vamp::FloatVector<vamp::binding::rake>> env_converted(environment);

            // Bounds
            vamp::planning::BaseBounds bounds{.x_min = x_min,
                                              .x_max = x_max,
                                              .y_min = y_min,
                                              .y_max = y_max,
                                              .theta_min = theta_min,
                                              .theta_max = theta_max};

            // Solve
            auto res = FCITWB::solve(
                arm_start, arm_goals, base_start_cfg, base_goal_cfgs, env_converted, settings, bounds, rng);

            // Pack into WholeBodyPlanningResult
            vamp::planning::WholeBodyPlanningResult<vamp::robots::Fetch> wb;
            wb.arm_result = res.arm_result;
            wb.base_path = std::move(res.base_path);
            return wb;
        },
        "start"_a,
        "goals"_a,
        "base_start"_a,
        "base_goals"_a,
        "environment"_a,
        "settings"_a,
        "rng"_a,
        "x_min"_a,
        "x_max"_a,
        "y_min"_a,
        "y_max"_a,
        "theta_min"_a = -static_cast<float>(M_PI),
        "theta_max"_a = static_cast<float>(M_PI),
        "Solve the whole-body motion planning problem with FCIT* for multiple goals.");

    // Add whole_body_simplify binding
    using WholeBodyResult = vamp::planning::WholeBodyPlanningResult<vamp::robots::Fetch>;

    nb::class_<WholeBodyResult>(submodule, "WholeBodyPlanningResult")
        .def(nb::init<>())
        .def_rw("arm_result", &WholeBodyResult::arm_result, "The planning result for the arm component")
        .def_rw("base_path", &WholeBodyResult::base_path, "The path for the base component")
        .def(
            "validate_paths",
            &WholeBodyResult::validatePaths,
            "Check if arm and base paths have the same length")
        .def(
            "interpolate",
            [](WholeBodyResult &result, float density)
            { vamp::planning::interpolate_whole_body_path<vamp::robots::Fetch>(result, density); },
            "density"_a,
            "Interpolate both arm and base paths to ensure the distance between adjacent waypoints "
            "is no more than the given density.");

    submodule.def(
        "whole_body_simplify",
        [](const std::vector<std::vector<float>> &arm_path,
           const std::vector<vamp::planning::MobileBaseConfiguration<vamp::robots::Fetch>> &base_path,
           const vamp::collision::Environment<float> &environment,
           const vamp::planning::SimplifySettings &settings,
           typename vamp::rng::RNG<vamp::robots::Fetch::dimension>::Ptr rng) -> WholeBodyResult
        {
            // Convert arm path to Path<Fetch::dimension>
            vamp::planning::Path<vamp::robots::Fetch::dimension> arm_path_converted;

            for (const auto &config : arm_path)
            {
                std::array<float, vamp::robots::Fetch::dimension> config_arr;

                // Copy values with bounds checking
                for (size_t i = 0; i < std::min(config.size(), (size_t)vamp::robots::Fetch::dimension); i++)
                {
                    config_arr[i] = config[i];
                }
                for (size_t i = config.size(); i < vamp::robots::Fetch::dimension; i++)
                {
                    config_arr[i] = 0.0f;  // Zero-pad if input is too short
                }

                arm_path_converted.emplace_back(config_arr);
            }

            // Convert environment to the expected type
            vamp::collision::Environment<vamp::FloatVector<vamp::binding::rake>> env_converted(environment);

            // Call the whole_body_simplify function
            return vamp::planning::whole_body_simplify<
                vamp::robots::Fetch,
                vamp::binding::rake,
                vamp::robots::Fetch::resolution>(
                arm_path_converted, base_path, env_converted, settings, rng);
        },
        "arm_path"_a,
        "base_path"_a,
        "environment"_a,
        "settings"_a,
        "rng"_a,
        "Simplify a whole-body path (arm and base) for the Fetch robot.");

    submodule.def(
        "filter_fetch_from_pointcloud",
        [](const std::vector<vamp::collision::Point> &pc,
           const std::vector<float> &arm_config_vec,
           const std::vector<float> &base_config_vec,
           const vamp::collision::Environment<float> &environment,
           float point_radius) -> std::vector<vamp::collision::Point>
        {
            using Robot = vamp::robots::Fetch;
            using EnvironmentVector = vamp::collision::Environment<vamp::FloatVector<rake>>;

            EnvironmentVector ev(environment);

            typename Robot::template Spheres<1> out;
            typename Robot::template ConfigurationBlock<1> block;

            std::array<float, Robot::dimension> arm_config;
            for (size_t i = 0; i < std::min(arm_config_vec.size(), (size_t)Robot::dimension); ++i)
                arm_config[i] = arm_config_vec[i];
            for (size_t i = arm_config_vec.size(); i < Robot::dimension; ++i)
                arm_config[i] = 0.0f;

            for (auto i = 0U; i < Robot::dimension; ++i)
            {
                block[i] = arm_config[i];
            }

            Robot::template sphere_fk<1>(block, out);

            std::vector<vamp::collision::Point> filtered;
            filtered.reserve(pc.size());

            float base_x = 0.f, base_y = 0.f, base_theta = 0.f;
            if (base_config_vec.size() > 0)
                base_x = base_config_vec[0];
            if (base_config_vec.size() > 1)
                base_y = base_config_vec[1];
            if (base_config_vec.size() > 2)
                base_theta = base_config_vec[2];

            const auto cos_theta = std::cos(base_theta);
            const auto sin_theta = std::sin(base_theta);

            for (const auto &point : pc)
            {
                const auto px = point[0];
                const auto py = point[1];
                const auto pz = point[2];
                const auto pr = point_radius;

                bool valid = true;
                
                for (auto i = 0U; i < Robot::n_spheres; ++i)
                {
                    const auto sx = out.x[{i, 0}];
                    const auto sy = out.y[{i, 0}];
                    const auto sz = out.z[{i, 0}];
                    const auto sr = out.r[{i, 0}];

                    const auto sx_rot = sx * cos_theta - sy * sin_theta;
                    const auto sy_rot = sx * sin_theta + sy * cos_theta;

                    const auto sx_world = sx_rot + base_x;
                    const auto sy_world = sy_rot + base_y;
                    const auto sz_world = sz;

                    if (vamp::collision::sphere_sphere_sql2(
                            sx_world, sy_world, sz_world, sr, px, py, pz, pr) < 0)
                    {
                        valid = false;
                        break;
                    }
                }

                if (valid)
                {
                    filtered.emplace_back(point);
                }
            }

            return filtered;
        },
        "pointcloud"_a,
        "arm_configuration"_a,
        "base_configuration"_a,
        "environment"_a,
        "point_radius"_a,
        "Filters all colliding points from a point cloud for the Fetch robot, "
        "considering its mobile base transformation.");

    submodule.def(
        "check_whole_body_collisions",
        [](const vamp::collision::Environment<float> &environment,
           const std::vector<std::vector<float>> &arm_configs,
           const std::vector<std::vector<float>> &base_configs) -> bool
        {
            // Ensure arm and base configs have the same number of points
            if (arm_configs.size() != base_configs.size())
            {
                return true; // Invalid input, treat as collision
            }

            // If there are no points, no collision
            if (arm_configs.empty())
            {
                return false;
            }

            // Convert arm path to Path<Fetch::dimension>
            vamp::planning::Path<vamp::robots::Fetch::dimension> arm_path_converted;
            arm_path_converted.reserve(arm_configs.size());

            for (const auto &config : arm_configs)
            {
                std::array<float, vamp::robots::Fetch::dimension> config_arr;

                // Copy values with bounds checking
                for (size_t i = 0;
                     i < std::min(config.size(), (size_t)vamp::robots::Fetch::dimension);
                     i++)
                {
                    config_arr[i] = config[i];
                }
                for (size_t i = config.size(); i < vamp::robots::Fetch::dimension; i++)
                {
                    config_arr[i] = 0.0f; // Zero-pad if input is too short
                }

                arm_path_converted.emplace_back(config_arr);
            }

            // Convert base path to vector<MobileBaseConfiguration<Fetch>>
            std::vector<vamp::planning::MobileBaseConfiguration<vamp::robots::Fetch>> base_path_converted;
            base_path_converted.reserve(base_configs.size());

            for (const auto &base_config : base_configs)
            {
                std::array<float, 3> base_config_arr{0.0f, 0.0f, 0.0f};

                // Copy values with bounds checking
                for (size_t i = 0; i < std::min(base_config.size(), (size_t)3); i++)
                {
                    base_config_arr[i] = base_config[i];
                }

                base_path_converted.emplace_back(base_config_arr);
            }

            // Convert environment to the expected type
            vamp::collision::Environment<vamp::FloatVector<vamp::binding::rake>> env_converted(
                environment);

            // If there's only one point, check that single configuration for collision
            if (arm_path_converted.size() == 1)
            {
                return !vamp::planning::validate_whole_body_motion<
                    vamp::robots::Fetch,
                    vamp::binding::rake,
                    vamp::robots::Fetch::resolution>(
                    arm_path_converted[0],
                    arm_path_converted[0],
                    base_path_converted[0],
                    base_path_converted[0],
                    env_converted);
            }

            // Check each segment of the path for collisions
            for (size_t i = 0; i < arm_path_converted.size() - 1; ++i)
            {
                if (!vamp::planning::validate_whole_body_motion<
                        vamp::robots::Fetch,
                        vamp::binding::rake,
                        vamp::robots::Fetch::resolution>(
                        arm_path_converted[i],
                        arm_path_converted[i + 1],
                        base_path_converted[i],
                        base_path_converted[i + 1],
                        env_converted))
                {
                    std::cout << "Collision between waypoint " << i << " and " << i + 1 << std::endl;
                    return true; // Collision found
                }
            }

            return false; // No collisions found
        },
        "environment"_a,
        "arm_configs"_a,
        "base_configs"_a,
        "Check a batch of whole-body configurations for collision against the environment.");

    submodule.def(
        "validate_whole_body_config",
        [](const std::vector<float>& arm_config_vec,
           const std::vector<float>& base_config_vec,
           const vamp::collision::Environment<float>& environment) -> bool
        {
            using Robot = vamp::robots::Fetch;
            
            // Convert arm config
            std::array<float, Robot::dimension> arm_config_arr{};
            std::copy_n(arm_config_vec.begin(), std::min(arm_config_vec.size(), (size_t)Robot::dimension), arm_config_arr.begin());
            Robot::Configuration arm_config(arm_config_arr);

            // Convert base config
            std::array<float, 3> base_config_arr{};
            std::copy_n(base_config_vec.begin(), std::min(base_config_vec.size(), (size_t)3), base_config_arr.begin());
            vamp::planning::MobileBaseConfiguration<Robot> base_config(base_config_arr);

            // Convert environment
            vamp::collision::Environment<vamp::FloatVector<vamp::binding::rake>> env_converted(environment);

            return vamp::planning::validate_motion_wb<
                Robot,
                vamp::binding::rake,
                1>(
                arm_config,
                arm_config,
                base_config,
                base_config,
                env_converted);
        },
        "arm_config"_a,
        "base_config"_a,
        "environment"_a,
        "Check a single whole-body configuration for collision."
    );
}
