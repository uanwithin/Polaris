#include "polaris/reference_path.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace polaris
{
ReferencePath::ReferencePath(std::vector<Vec2> points)
  : points_(std::move(points))
{
  if(points_.size() < 2)
  {
    throw std::invalid_argument("Path needs two points");
  }
  arc_lengths_.push_back(0.0);
  for(auto const point : points_)
  {
    if(!std::isfinite(point.x) || !std::isfinite(point.y))
    {
      throw std::invalid_argument("Path points must be finite");
    }
  }
  for(std::size_t i = 1; i < points_.size(); ++i)
  {
    auto const segment =
        std::sqrt(squared_distance(points_[i], points_[i - 1]));
    if(segment < 1e-9)
    {
      throw std::invalid_argument("Repeated path point");
    }
    arc_lengths_.push_back(arc_lengths_.back() + segment);
  }
}

double
ReferencePath::length() const
{
  return arc_lengths_.back();
}
std::span<Vec2 const>
ReferencePath::points() const
{
  return points_;
}

double
ReferencePath::project(Vec2 point) const
{
  auto best_distance = std::numeric_limits<double>::infinity();
  double result{};
  for(std::size_t i = 1; i < points_.size(); ++i)
  {
    auto const a = points_[i - 1];
    auto const b = points_[i];
    auto const fraction = std::clamp(
        ((point.x - a.x) * (b.x - a.x) + (point.y - a.y) * (b.y - a.y)) /
            squared_distance(a, b),
        0.0,
        1.0);
    auto const distance = squared_distance(
        point,
        {a.x + fraction * (b.x - a.x), a.y + fraction * (b.y - a.y)});
    if(distance < best_distance)
    {
      best_distance = distance;
      result = arc_lengths_[i - 1] +
               fraction * (arc_lengths_[i] - arc_lengths_[i - 1]);
    }
  }
  return result;
}

State
ReferencePath::sample(double arc_length) const
{
  arc_length = std::clamp(arc_length, 0.0, length());
  auto const upper = std::ranges::upper_bound(arc_lengths_, arc_length);
  auto const index =
      std::min(static_cast<std::size_t>(upper - arc_lengths_.begin()),
               points_.size() - 1);
  auto const a = points_[index - 1];
  auto const b = points_[index];
  auto const fraction = (arc_length - arc_lengths_[index - 1]) /
                        (arc_lengths_[index] - arc_lengths_[index - 1]);
  return {
      a.x + fraction * (b.x - a.x),
      a.y + fraction * (b.y - a.y),
      std::atan2(b.y - a.y, b.x - a.x),
  };
}
} // namespace polaris
