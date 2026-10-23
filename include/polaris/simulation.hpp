#pragma once

#include "polaris/grid_map.hpp"
#include "polaris/mppc.hpp"

#include <functional>
#include <span>
#include <string>
#include <string_view>

namespace polaris
{
struct Scenario
{
  GridMap map;
  ReferencePath reference;
  State start;
};

struct Frame
{
  int step{};
  double time{};
  State state;
  Control previous;
  std::span<Vec2 const> scan;
  std::span<Vec2 const> trail;
  std::reference_wrapper<Plan const> plan;
};

struct SimulationResult
{
  bool reached_goal{};
  bool collision{};
  int steps{};
  int braking_steps{};
  double final_error{};
  double min_clearance{};
  double max_tracking_error{};
  double mean_solve_ms{};
  double max_solve_ms{};
  State final_state;
  Control final_control;
  std::string status;
};

[[nodiscard]] Scenario
make_scenario(std::string_view name);
[[nodiscard]] SimulationResult
simulate(Scenario const& scenario,
         MppcPlanner& planner,
         int max_steps,
         std::function<void(Frame const&)> const& on_frame = {});
} // namespace polaris
