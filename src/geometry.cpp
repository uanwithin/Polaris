#include "polaris/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

namespace polaris
{
double
squared_distance(Vec2 a, Vec2 b)
{
  return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
}

double
wrap_angle(double angle)
{
  return std::remainder(angle, 2.0 * std::numbers::pi);
}

State
advance(State state, Control control, double dt)
{
  // Eq. (1), forward Euler differential-drive kinematics. The same model
  // is used by both shooting and the simulated plant.
  return {
      state.x + dt * control.speed * std::cos(state.yaw),
      state.y + dt * control.speed * std::sin(state.yaw),
      wrap_angle(state.yaw + dt * control.yaw_rate),
  };
}

Distance
point_rectangle_distance(Vec2 point, State state, Vehicle vehicle)
{
  auto const c = std::cos(state.yaw);
  auto const s = std::sin(state.yaw);
  auto const dx = point.x - state.x;
  auto const dy = point.y - state.y;
  auto const local_x = c * dx + s * dy;
  auto const local_y = -s * dx + c * dy;
  auto const qx = std::abs(local_x) - vehicle.length / 2.0;
  auto const qy = std::abs(local_y) - vehicle.width / 2.0;
  auto const ox = std::max(qx, 0.0);
  auto const oy = std::max(qy, 0.0);
  auto const outside = std::hypot(ox, oy);
  auto const sign_x = std::copysign(1.0, local_x);
  auto const sign_y = std::copysign(1.0, local_y);
  double gx{};
  double gy{};
  if(outside > 1e-12)
  {
    gx = sign_x * ox / outside;
    gy = sign_y * oy / outside;
  }
  else if(qx >= qy)
  {
    gx = sign_x;
  }
  else
  {
    gy = sign_y;
  }
  return {
      outside + std::min(std::max(qx, qy), 0.0),
      {-c * gx + s * gy, -s * gx - c * gy, local_y * gx - local_x * gy},
  };
}

std::array<Vec2, 5>
footprint(State state, Vehicle vehicle)
{
  auto const hx = vehicle.length / 2.0;
  auto const hy = vehicle.width / 2.0;
  std::array<Vec2, 5> corners{
      {{hx, hy}, {-hx, hy}, {-hx, -hy}, {hx, -hy}, {hx, hy}},
  };
  auto const c = std::cos(state.yaw);
  auto const s = std::sin(state.yaw);
  for(auto& corner : corners)
  {
    corner = {
        state.x + c * corner.x - s * corner.y,
        state.y + s * corner.x + c * corner.y,
    };
  }
  return corners;
}

double
clearance(State state, Vehicle vehicle, std::span<Vec2 const> points)
{
  auto result = std::numeric_limits<double>::infinity();
  for(auto const point : points)
  {
    result =
        std::min(result, point_rectangle_distance(point, state, vehicle).value);
  }
  return result;
}
} // namespace polaris
