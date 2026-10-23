#pragma once

#include "polaris/geometry.hpp"

#include <cstdint>
#include <vector>

namespace polaris
{
struct LidarConfig
{
  int beams{180};
  double max_range{7.0};
};

class GridMap
{
public:
  GridMap(int width, int height, double resolution);
  [[nodiscard]] int
  width() const;
  [[nodiscard]] int
  height() const;
  [[nodiscard]] double
  resolution() const;
  [[nodiscard]] bool
  occupied(int col, int row) const;
  void
  fill_rectangle(Vec2 minimum, Vec2 maximum);
  void
  add_boundary();
  [[nodiscard]] std::vector<Vec2>
  occupied_centers() const;
  [[nodiscard]] std::vector<Vec2>
  scan(State state, LidarConfig config) const;
  [[nodiscard]] bool
  collides(State state, Vehicle vehicle) const;

private:
  int width_;
  int height_;
  double resolution_;
  std::vector<std::uint8_t> cells_;
};
} // namespace polaris
