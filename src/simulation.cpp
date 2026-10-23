#include "polaris/simulation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace polaris
{
Scenario
make_scenario(std::string_view name)
{
  GridMap map(180, 120, 0.1);
  map.add_boundary();
  if(name == "avoidance")
  {
    map.fill_rectangle({7.0, 4.8}, {8.2, 7.2});
    map.fill_rectangle({11.0, 7.5}, {12.5, 9.5});
    map.fill_rectangle({12.0, 2.5}, {13.5, 4.0});
  }
  else if(name == "blocked")
  {
    map.fill_rectangle({7.0, 0.0}, {7.5, 12.0});
  }
  else if(name != "empty")
  {
    throw std::invalid_argument("Unknown scenario: " + std::string(name));
  }
  return {
      std::move(map),
      ReferencePath({{2.0, 6.0}, {16.0, 6.0}}),
      {2.0, 6.0, 0.0},
  };
}

SimulationResult
simulate(Scenario const& scenario,
         MppcPlanner& planner,
         int max_steps,
         std::function<void(Frame const&)> const& on_frame)
{
  if(max_steps <= 0)
  {
    throw std::invalid_argument("Step count must be positive");
  }
  SimulationResult result;
  result.min_clearance = std::numeric_limits<double>::infinity();
  result.status = "step_limit";
  auto state = scenario.start;
  Control previous;
  auto const vehicle = planner.vehicle();
  auto const& config = planner.config();
  auto const goal = scenario.reference.sample(scenario.reference.length());
  std::vector<Vec2> trail{{state.x, state.y}};
  double total_solve_ms{};
  int stationary_steps{};
  for(int step = 0; step < max_steps; ++step)
  {
    if(scenario.map.collides(state, vehicle))
    {
      result.collision = true;
      result.status = "collision";
      break;
    }
    auto const points = scenario.map.scan(state, {});
    auto const plan =
        planner.solve(state, previous, points, scenario.reference);
    total_solve_ms += plan.solve_ms;
    result.max_solve_ms = std::max(result.max_solve_ms, plan.solve_ms);
    result.steps = step + 1;
    result.braking_steps += plan.braking ? 1 : 0;
    auto const measured_clearance = clearance(state, vehicle, points);
    result.min_clearance = std::min(result.min_clearance, measured_clearance);
    auto const projection = scenario.reference.sample(
        scenario.reference.project({state.x, state.y}));
    result.max_tracking_error =
        std::max(result.max_tracking_error,
                 std::hypot(state.x - projection.x, state.y - projection.y));
    if(on_frame)
    {
      on_frame({step, step * config.dt, state, previous, points, trail, plan});
    }
    if(!plan.feasible)
    {
      result.status = plan.status;
      break;
    }
    // The map is ground truth for the simulator only. The planner never sees it.
    // Inspect the entire Euler transition, including rotation of the footprint.
    bool collision{};
    for(int sample = 1; sample <= 20; ++sample)
    {
      auto const intermediate =
          advance(state, plan.command, config.dt * sample / 20.0);
      collision = collision || scenario.map.collides(intermediate, vehicle);
      result.min_clearance = std::min(result.min_clearance,
                                      clearance(intermediate, vehicle, points));
    }
    if(collision)
    {
      result.collision = true;
      result.status = "collision_during_transition";
      break;
    }
    if(std::hypot(state.x - goal.x, state.y - goal.y) < 0.18 &&
       std::abs(wrap_angle(state.yaw - goal.yaw)) < 0.2 &&
       std::abs(previous.speed) < 0.05 && std::abs(previous.yaw_rate) < 0.05)
    {
      result.reached_goal = true;
      result.status = "goal_reached";
      break;
    }
    auto const next = advance(state, plan.command, config.dt);
    stationary_steps = std::hypot(next.x - state.x, next.y - state.y) < 0.001
                           ? stationary_steps + 1
                           : 0;
    state = next;
    previous = plan.command;
    trail.push_back({state.x, state.y});
    if(stationary_steps >= 25)
    {
      result.status = "blocked_or_local_minimum";
      break;
    }
  }
  result.final_state = state;
  result.final_control = previous;
  result.final_error = std::hypot(state.x - goal.x, state.y - goal.y);
  result.mean_solve_ms = total_solve_ms / std::max(result.steps, 1);
  return result;
}
} // namespace polaris
