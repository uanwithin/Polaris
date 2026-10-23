#pragma once

#include <array>
#include <span>

namespace polaris
{
struct Vec2
{
  double x{};
  double y{};
};

struct State
{
  double x{};
  double y{};
  double yaw{};
};

struct Control
{
  double speed{};
  double yaw_rate{};
};

struct Vehicle
{
  double length{1.0};
  double width{0.6};
};

struct Distance
{
  // Signed extension of Eq. (8): identical to Euclidean distance outside,
  // negative inside. Its positive-clearance feasible set is unchanged.
  double value{};
  std::array<double, 3> gradient{};
};

[[nodiscard]] double
squared_distance(Vec2 a, Vec2 b);
[[nodiscard]] double
wrap_angle(double angle);
[[nodiscard]] State
advance(State state, Control control, double dt);
[[nodiscard]] Distance
point_rectangle_distance(Vec2 point, State state, Vehicle vehicle);
[[nodiscard]] std::array<Vec2, 5>
footprint(State state, Vehicle vehicle);
[[nodiscard]] double
clearance(State state, Vehicle vehicle, std::span<Vec2 const> points);
} // namespace polaris
