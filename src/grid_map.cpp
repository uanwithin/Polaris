#include "polaris/grid_map.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace polaris
{
GridMap::GridMap(int width, int height, double resolution)
  : width_(width)
  , height_(height)
  , resolution_(resolution)
{
  if(width < 3 || height < 3 || !std::isfinite(resolution) || resolution <= 0.0)
  {
    throw std::invalid_argument("Invalid grid dimensions or resolution");
  }
  cells_.resize(static_cast<std::size_t>(width) *
                static_cast<std::size_t>(height));
}

int
GridMap::width() const
{
  return width_;
}
int
GridMap::height() const
{
  return height_;
}
double
GridMap::resolution() const
{
  return resolution_;
}

bool
GridMap::occupied(int col, int row) const
{
  if(col < 0 || row < 0 || col >= width_ || row >= height_)
  {
    return true;
  }
  return cells_[static_cast<std::size_t>(row) *
                    static_cast<std::size_t>(width_) +
                static_cast<std::size_t>(col)] != 0;
}

void
GridMap::fill_rectangle(Vec2 minimum, Vec2 maximum)
{
  if(!std::isfinite(minimum.x) || !std::isfinite(minimum.y) ||
     !std::isfinite(maximum.x) || !std::isfinite(maximum.y) ||
     minimum.x > maximum.x || minimum.y > maximum.y)
  {
    throw std::invalid_argument("Invalid obstacle rectangle");
  }
  for(int row = 0; row < height_; ++row)
  {
    for(int col = 0; col < width_; ++col)
    {
      auto const x = (static_cast<double>(col) + 0.5) * resolution_;
      auto const y = (static_cast<double>(row) + 0.5) * resolution_;
      if(x >= minimum.x && x < maximum.x && y >= minimum.y && y < maximum.y)
      {
        cells_[static_cast<std::size_t>(row) *
                   static_cast<std::size_t>(width_) +
               static_cast<std::size_t>(col)] = 1;
      }
    }
  }
}

void
GridMap::add_boundary()
{
  auto const w = static_cast<double>(width_) * resolution_;
  auto const h = static_cast<double>(height_) * resolution_;
  fill_rectangle({0.0, 0.0}, {w, resolution_});
  fill_rectangle({0.0, h - resolution_}, {w, h});
  fill_rectangle({0.0, 0.0}, {resolution_, h});
  fill_rectangle({w - resolution_, 0.0}, {w, h});
}

std::vector<Vec2>
GridMap::occupied_centers() const
{
  std::vector<Vec2> points;
  for(int row = 0; row < height_; ++row)
  {
    for(int col = 0; col < width_; ++col)
    {
      if(occupied(col, row))
      {
        points.push_back({
            (static_cast<double>(col) + 0.5) * resolution_,
            (static_cast<double>(row) + 0.5) * resolution_,
        });
      }
    }
  }
  return points;
}

std::vector<Vec2>
GridMap::scan(State state, LidarConfig config) const
{
  if(config.beams <= 0 || !std::isfinite(config.max_range) ||
     config.max_range <= 0.0)
  {
    throw std::invalid_argument("Invalid lidar configuration");
  }
  std::vector<Vec2> points;
  points.reserve(static_cast<std::size_t>(config.beams));
  double const delte_theta = 2.0 * std::numbers::pi / config.beams;
  for(int beam = 0; beam < config.beams; ++beam)
  {
    auto const angle = state.yaw + static_cast<double>(beam) * delte_theta;
    auto const dx = std::cos(angle);
    auto const dy = std::sin(angle);
    auto col = static_cast<int>(std::floor(state.x / resolution_));
    auto row = static_cast<int>(std::floor(state.y / resolution_));
    auto const step_x = dx >= 0.0 ? 1 : -1;
    auto const step_y = dy >= 0.0 ? 1 : -1;
    auto const infinity = std::numeric_limits<double>::infinity();
    auto const delta_x =
        std::abs(dx) > 1e-12 ? resolution_ / std::abs(dx) : infinity;
    auto const delta_y =
        std::abs(dy) > 1e-12 ? resolution_ / std::abs(dy) : infinity;
    auto next_x =
        std::abs(dx) > 1e-12
            ? ((col + (step_x > 0 ? 1 : 0)) * resolution_ - state.x) / dx
            : infinity;
    auto next_y =
        std::abs(dy) > 1e-12
            ? ((row + (step_y > 0 ? 1 : 0)) * resolution_ - state.y) / dy
            : infinity;
    double range{};
    // Exact grid traversal: report the first occupied-cell boundary. No hit at
    // maximum range means no point, not a fictitious obstacle on a range circle.
    while(range <= config.max_range)
    {
      if(occupied(col, row))
      {
        points.push_back({state.x + range * dx, state.y + range * dy});
        break;
      }
      if(next_x < next_y)
      {
        range = next_x;
        next_x += delta_x;
        col += step_x;
      }
      else
      {
        range = next_y;
        next_y += delta_y;
        row += step_y;
      }
    }
  }
  return points;
}

bool
GridMap::collides(State state, Vehicle vehicle) const
{
  auto const corners = footprint(state, vehicle);
  auto min_x = corners.front().x;
  auto max_x = min_x;
  auto min_y = corners.front().y;
  auto max_y = min_y;
  for(auto const corner : corners)
  {
    min_x = std::min(min_x, corner.x);
    max_x = std::max(max_x, corner.x);
    min_y = std::min(min_y, corner.y);
    max_y = std::max(max_y, corner.y);
  }
  if(min_x <= 0.0 || min_y <= 0.0 || max_x >= width_ * resolution_ ||
     max_y >= height_ * resolution_)
  {
    return true;
  }
  auto const c = std::cos(state.yaw);
  auto const s = std::sin(state.yaw);
  auto const half = resolution_ / 2.0;
  // Separating-axis test between oriented vehicle and each solid grid cell.
  for(int row = static_cast<int>(min_y / resolution_);
      row <= static_cast<int>(max_y / resolution_);
      ++row)
  {
    for(int col = static_cast<int>(min_x / resolution_);
        col <= static_cast<int>(max_x / resolution_);
        ++col)
    {
      if(!occupied(col, row))
      {
        continue;
      }
      auto const dx = (col + 0.5) * resolution_ - state.x;
      auto const dy = (row + 0.5) * resolution_ - state.y;
      auto const cell_radius = half * (std::abs(c) + std::abs(s));
      if(std::abs(dx) <= half + std::abs(c) * vehicle.length / 2.0 +
                             std::abs(s) * vehicle.width / 2.0 &&
         std::abs(dy) <= half + std::abs(s) * vehicle.length / 2.0 +
                             std::abs(c) * vehicle.width / 2.0 &&
         std::abs(c * dx + s * dy) <= vehicle.length / 2.0 + cell_radius &&
         std::abs(-s * dx + c * dy) <= vehicle.width / 2.0 + cell_radius)
      {
        return true;
      }
    }
  }
  return false;
}
} // namespace polaris
