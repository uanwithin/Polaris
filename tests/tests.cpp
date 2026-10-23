#include "polaris/simulation.hpp"

#include <cmath>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>

namespace
{
void
require(bool condition, std::string const& message)
{
  if(!condition)
  {
    throw std::runtime_error(message);
  }
}

void
close(double actual,
      double expected,
      double tolerance,
      std::string const& message)
{
  require(std::abs(actual - expected) <= tolerance, message);
}

void
test_geometry()
{
  polaris::Vehicle const vehicle{2.0, 1.0};
  polaris::State const origin;
  close(polaris::point_rectangle_distance({3.0, 0.0}, origin, vehicle).value,
        2.0,
        1e-12,
        "face distance");
  close(polaris::point_rectangle_distance({2.0, 1.5}, origin, vehicle).value,
        std::numbers::sqrt2,
        1e-12,
        "corner distance");
  require(polaris::point_rectangle_distance({0.0, 0.0}, origin, vehicle).value <
              0.0,
          "interior distance");
  close(polaris::point_rectangle_distance({0.0, 3.0},
                                          {0.0, 0.0, std::numbers::pi / 2.0},
                                          vehicle)
            .value,
        2.0,
        1e-12,
        "rotated rectangle");
  for(auto const point : {polaris::Vec2{3.0, 2.0},
                          polaris::Vec2{0.2, 0.1},
                          polaris::Vec2{-2.0, 0.2}})
  {
    polaris::State const state{0.1, -0.2, 0.3};
    auto const distance =
        polaris::point_rectangle_distance(point, state, vehicle);
    for(std::size_t axis = 0; axis < 3; ++axis)
    {
      auto plus = state;
      auto minus = state;
      constexpr double k_epsilon = 1e-6;
      if(axis == 0)
      {
        plus.x += k_epsilon;
        minus.x -= k_epsilon;
      }
      else if(axis == 1)
      {
        plus.y += k_epsilon;
        minus.y -= k_epsilon;
      }
      else
      {
        plus.yaw += k_epsilon;
        minus.yaw -= k_epsilon;
      }
      auto const numerical =
          (polaris::point_rectangle_distance(point, plus, vehicle).value -
           polaris::point_rectangle_distance(point, minus, vehicle).value) /
          (2 * k_epsilon);
      close(distance.gradient.at(axis),
            numerical,
            1e-6,
            "analytic distance gradient");
    }
  }
}

void
test_map_and_lidar()
{
  polaris::GridMap map(100, 100, 0.1);
  map.fill_rectangle({5.0, 1.0}, {5.1, 9.0});
  auto const points = map.scan({2.0, 5.0, 0.0}, {4, 4.0});
  require(points.size() == 2,
          "lidar only returns obstacle or map-boundary hits");
  close(points.front().x, 5.0, 1e-12, "first ray exact wall hit");
  close(points.front().y, 5.0, 1e-12, "first ray y");
  require(map.collides({4.6, 5.0, 0.0}, {}), "vehicle face hits thin wall");
  require(!map.collides({4.0, 5.0, 0.0}, {}), "separated vehicle");
  require(map.collides({0.1, 5.0, 0.0}, {}), "outside grid is occupied");
  require(map.scan({2.0, 5.0, 0.0}, {4, 0.5}).empty(),
          "no synthetic max-range returns");
}

void
test_reference()
{
  polaris::ReferencePath const path({{0.0, 0.0}, {2.0, 0.0}, {2.0, 3.0}});
  close(path.project({1.0, 1.0}), 1.0, 1e-12, "reference projection");
  close(path.sample(4.0).y, 2.0, 1e-12, "reference interpolation");
  close(path.sample(9.0).y, 3.0, 1e-12, "reference endpoint clamp");
}

void
test_plan()
{
  polaris::MppcPlanner planner;
  polaris::ReferencePath const reference({{0.0, 0.0}, {8.0, 0.0}});
  polaris::State const origin;
  polaris::Control const stopped;
  std::vector<polaris::Vec2> const no_points;
  auto const plan = planner.solve(origin, stopped, no_points, reference);
  require(plan.feasible, "empty world plan feasible");
  require(plan.command.speed > 0.0, "empty world advances");
  polaris::Control previous;
  for(std::size_t i = 0; i < plan.controls.size(); ++i)
  {
    auto const control = plan.controls[i];
    require(std::abs(control.speed - previous.speed) <= 0.20001,
            "acceleration bound");
    require(std::abs(control.yaw_rate - previous.yaw_rate) <= 0.50001,
            "angular acceleration bound");
    auto const expected = polaris::advance(plan.states[i], control, 0.2);
    close(plan.states[i + 1].x, expected.x, 1e-12, "shooting dynamics");
    close(plan.states[i + 1].y, expected.y, 1e-12, "shooting lateral dynamics");
    previous = control;
  }
  std::vector<polaris::Vec2> const colliding{{0.0, 0.0}};
  require(!planner.solve(origin, stopped, colliding, reference).feasible,
          "initial collision is infeasible");
}

void
test_simulation(std::string_view name)
{
  polaris::MppcPlanner planner;
  auto const scenario = polaris::make_scenario(name);
  auto const& config = planner.config();
  auto const result = polaris::simulate(
      scenario,
      planner,
      300,
      [&](polaris::Frame const& frame)
      {
        auto const& plan = frame.plan.get();
        if(!plan.feasible)
        {
          return;
        }
        auto previous = frame.previous;
        for(std::size_t h = 0; h < plan.controls.size(); ++h)
        {
          auto const control = plan.controls[h];
          require(control.speed >= -1e-5 &&
                      control.speed <= config.max_speed + 1e-5,
                  "speed bound in closed loop");
          require(std::abs(control.yaw_rate) <= config.max_yaw_rate + 1e-5,
                  "yaw rate bound in closed loop");
          require(std::abs(control.speed - previous.speed) <=
                      config.max_acceleration * config.dt + 1e-5,
                  "acceleration bound in closed loop");
          require(std::abs(control.yaw_rate - previous.yaw_rate) <=
                      config.max_yaw_acceleration * config.dt + 1e-5,
                  "yaw acceleration bound in closed loop");
          require(polaris::clearance(plan.states[h + 1],
                                     planner.vehicle(),
                                     frame.scan) >=
                      config.safety_distance - 1e-5,
                  "all observed points satisfy predicted clearance");
          previous = control;
        }
      });
  require(!result.collision, "no ground-truth grid collision");
  require(result.min_clearance >= config.safety_distance - 1e-5,
          "executed clearance");
  if(name == "blocked")
  {
    require(!result.reached_goal, "blocked scene must not report arrival");
    require(result.status == "blocked_or_local_minimum",
            "blocked scene stops safely");
    require(std::abs(result.final_control.speed) < 0.01,
            "blocked scene settles to rest");
  }
  else
  {
    require(result.reached_goal && result.final_error < 0.18,
            "closed-loop arrival");
    require(std::abs(result.final_control.speed) < 0.05, "arrival speed");
    require(std::abs(result.final_control.yaw_rate) < 0.05, "arrival yaw rate");
    if(name == "avoidance")
    {
      require(result.max_tracking_error > 1.0,
              "actually detours around obstructed reference");
    }
  }
  std::cout << name << ": " << result.status << " steps=" << result.steps
            << " clearance=" << result.min_clearance
            << " error=" << result.final_error << '\n';
}
} // namespace

int
main(int argc, char const* const* argv)
{
  try
  {
    if(argc == 2)
    {
      test_simulation(argv[1]);
      return 0;
    }
    test_geometry();
    test_map_and_lidar();
    test_reference();
    test_plan();
    std::cout << "Geometry, distance gradients, ray casting, path, and MPC "
                 "checks passed\n";
    return 0;
  }
  catch(std::exception const& error)
  {
    std::cerr << "Test failed: " << error.what() << '\n';
    return 1;
  }
}
