#pragma once

#include <cmath>

#include <vamp/planning/simplify_settings.hh>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace vamp::planning
{
    // Configuration parameters for Hybrid A* algorithm
    template <typename Robot>
    struct HybridAStarConfig
    {
        float cell_size;               // Size of grid cells for discretization
        float heading_resolution;      // Resolution of heading discretization
        int num_headings;              // Number of discrete headings
        float steering_angle;          // Maximum steering angle
        float turning_radius;          // Minimum turning radius
        float motion_resolution;       // Step size for motion primitives
        bool allow_reverse;            // Whether to allow reverse motion
        float point_radius;            // Point radius for pointcloud collision checking during planning
        float wheel_base = 0.35f;      // Distance between front and rear axles (for bicycle model)
        float reverse_penalty = 1.5f;  // Multiplier for cost of reverse motion
        float turn_penalty = 1.1f;     // Multiplier for cost of turning motion
        int max_iterations = 50000;    // Max iterations for the planner
        float shot_distance = 1.0f;    // Distance for shot planning

        // --- New parameters for in-place rotation ---
        bool allow_inplace_rotation;  // Whether to allow in-place rotation primitives
        float inplace_rotation_step;  // The angle for a single in-place rotation primitive (radians)
        float inplace_rotation_cost;  // The cost associated with a single in-place rotation step

        // --- Safety margin parameters ---
        float safety_margin_radius;   // Radius for safety margin collision checking
        float safety_margin_penalty;  // Cost penalty multiplier for paths in safety margin

        // --- Settings for path post-processing ---
        SimplifySettings simplify_settings;

        HybridAStarConfig()
          : cell_size(0.1f)
          , heading_resolution(M_PI / 18.0f)  // 10 degrees
          , num_headings(19)
          , steering_angle(M_PI / 4.0f)
          , turning_radius(0.3f)
          , motion_resolution(0.2f)
          , allow_reverse(false)
          , point_radius(0.03f)
          , wheel_base(0.45f)
          , reverse_penalty(20.0f)  // Match Robot::reverse_penalty to discourage backward motion
          , turn_penalty(1.1f)
          , max_iterations(50000)
          // --- Default values for new parameters ---
          , allow_inplace_rotation(true)
          , inplace_rotation_step(M_PI / 18.0f)  // 10 degrees
          , inplace_rotation_cost(10.0f)         // Increased to discourage spinning
          , safety_margin_radius(0.2f)           // Default safety margin radius
          , safety_margin_penalty(3.0f)          // Default safety margin penalty multiplier
          , shot_distance(1.0f)
        {
            // Configure default simplification settings
            simplify_settings.operations = {BSPLINE, SHORTCUT};
            simplify_settings.bspline.max_steps = 3;
            simplify_settings.interpolate = 64;  // Corresponds to waypoints per segment
        }
    };
}  // namespace vamp::planning