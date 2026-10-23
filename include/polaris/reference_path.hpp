#pragma once

#include "polaris/geometry.hpp"

#include <span>
#include <vector>

namespace polaris
{
class ReferencePath
{
public:
  explicit ReferencePath(std::vector<Vec2> points);
  [[nodiscard]] double
  length() const;
  [[nodiscard]] double
  project(Vec2 point) const;
  [[nodiscard]] State
  sample(double arc_length) const;
  [[nodiscard]] std::span<Vec2 const>
  points() const;

private:
  std::vector<Vec2> points_;
  std::vector<double> arc_lengths_;
};
} // namespace polaris
