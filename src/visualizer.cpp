#include "polaris/visualizer.hpp"

#include <array>
#include <cmath>
#include <numbers>
#include <sstream>
#include <vector>

namespace polaris
{
namespace
{
using PlotPoint = std::array<float, 2>;

std::vector<PlotPoint>
plot_points(std::span<Vec2 const> points)
{
  std::vector<PlotPoint> result;
  result.reserve(points.size());
  for(auto const point : points)
  {
    result.push_back(
        {static_cast<float>(point.x), static_cast<float>(point.y)});
  }
  return result;
}

std::vector<PlotPoint>
plot_states(std::span<State const> states)
{
  std::vector<PlotPoint> result;
  result.reserve(states.size());
  for(auto const state : states)
  {
    result.push_back(
        {static_cast<float>(state.x), static_cast<float>(state.y)});
  }
  return result;
}

std::vector<PlotPoint>
safety_contour(State state, Vehicle vehicle, double margin)
{
  std::vector<PlotPoint> result;
  auto const c = std::cos(state.yaw);
  auto const s = std::sin(state.yaw);
  for(int corner = 0; corner < 4; ++corner)
  {
    auto const x =
        (corner == 0 || corner == 3 ? 1.0 : -1.0) * vehicle.length / 2.0;
    auto const y = (corner < 2 ? 1.0 : -1.0) * vehicle.width / 2.0;
    for(int sample = 0; sample <= 8; ++sample)
    {
      auto const angle = (corner + sample / 8.0) * std::numbers::pi / 2.0;
      auto const px = x + margin * std::cos(angle);
      auto const py = y + margin * std::sin(angle);
      result.push_back({
          static_cast<float>(state.x + c * px - s * py),
          static_cast<float>(state.y + s * px + c * py),
      });
    }
  }
  result.push_back(result.front());
  return result;
}
} // namespace

Visualizer::Visualizer(Scenario const& scenario,
                       Vehicle vehicle,
                       double safety_distance,
                       std::string const& recording,
                       bool spawn)
  : recording_("polaris_mppc")
  , vehicle_(vehicle)
  , safety_distance_(safety_distance)
{
  if(spawn)
  {
    recording_.spawn().throw_on_failure();
  }
  else
  {
    recording_.save(recording).throw_on_failure();
  }
  recording_.try_log_file_from_path(POLARIS_BLUEPRINT_PATH).throw_on_failure();
  auto const cells = plot_points(scenario.map.occupied_centers());
  auto const half = static_cast<float>(scenario.map.resolution() / 2.0);
  recording_.log_static("world/map",
                        rerun::Boxes2D::from_half_sizes(
                            std::vector<PlotPoint>(cells.size(), {half, half}))
                            .with_centers(cells)
                            .with_colors(rerun::Color(88, 98, 114)));
  auto const reference = plot_points(scenario.reference.points());
  recording_.log_static("world/reference",
                        rerun::LineStrips2D(rerun::LineStrip2D(reference))
                            .with_colors(rerun::Color(80, 140, 255))
                            .with_radii(0.025F));
  recording_.log_static("world/start",
                        rerun::Points2D({reference.front()})
                            .with_colors(rerun::Color(80, 220, 140))
                            .with_radii(0.12F)
                            .with_labels({"start"}));
  recording_.log_static("world/goal",
                        rerun::Points2D({reference.back()})
                            .with_colors(rerun::Color(220, 110, 255))
                            .with_radii(0.14F)
                            .with_labels({"goal"}));
  recording_.log_static(
      "legend",
      rerun::TextDocument(
          "Polaris MPPC | numerical baseline\n\n"
          "Blue: reference path\nGreen: executed trajectory\nYellow: optimized "
          "horizon\n"
          "Cyan: rectangular vehicle\nCyan outline: Euclidean safety margin\n"
          "Red: current lidar returns\nGray: ground-truth occupied cells "
          "(simulator only)\n\n"
          "Timeline: sim_time (seconds). Units: meters, radians.\n"
          "Planner input: current point cloud, state, previous control, "
          "reference.\n"
          "NLopt SLSQP solves paper Eqs. (1)-(11); no learned component."));
}

void
Visualizer::log(Frame const& frame) const
{
  recording_.set_time_duration_secs("sim_time", frame.time);
  recording_.log("world/lidar",
                 rerun::Points2D(plot_points(frame.scan))
                     .with_colors(rerun::Color(255, 95, 90))
                     .with_radii(0.035F));
  recording_.log(
      "world/trail",
      rerun::LineStrips2D(rerun::LineStrip2D(plot_points(frame.trail)))
          .with_colors(rerun::Color(65, 215, 135))
          .with_radii(0.035F));
  recording_.log("world/prediction",
                 rerun::LineStrips2D(
                     rerun::LineStrip2D(plot_states(frame.plan.get().states)))
                     .with_colors(rerun::Color(255, 202, 72))
                     .with_radii(0.025F));
  recording_.log("world/vehicle",
                 rerun::LineStrips2D(rerun::LineStrip2D(plot_points(
                                         footprint(frame.state, vehicle_))))
                     .with_colors(rerun::Color(50, 220, 240))
                     .with_radii(0.045F));
  recording_.log(
      "world/safety",
      rerun::LineStrips2D(
          rerun::LineStrip2D(
              safety_contour(frame.state, vehicle_, safety_distance_)))
          .with_colors(rerun::Color(50, 220, 240, 120))
          .with_radii(0.012F));
  std::vector<std::vector<PlotPoint>> predicted_bodies;
  for(std::size_t h = 3; h < frame.plan.get().states.size(); h += 3)
  {
    predicted_bodies.push_back(
        plot_points(footprint(frame.plan.get().states[h], vehicle_)));
  }
  recording_.log("world/predicted_bodies",
                 rerun::LineStrips2D(predicted_bodies)
                     .with_colors(rerun::Color(255, 202, 72, 90))
                     .with_radii(0.01F));
  std::array<Vec2, 2> const heading{
      {
          {frame.state.x, frame.state.y},
          {
              frame.state.x + 0.65 * std::cos(frame.state.yaw),
              frame.state.y + 0.65 * std::sin(frame.state.yaw),
          },
      },
  };
  recording_.log("world/heading",
                 rerun::LineStrips2D(rerun::LineStrip2D(plot_points(heading)))
                     .with_colors(rerun::Color(50, 220, 240))
                     .with_radii(0.025F));
  recording_.log("control/speed_mps",
                 rerun::Scalars(frame.plan.get().command.speed));
  recording_.log("control/yaw_rate_rps",
                 rerun::Scalars(frame.plan.get().command.yaw_rate));
  recording_.log("safety/predicted_clearance_m",
                 rerun::Scalars(frame.plan.get().min_clearance));
  recording_.log("safety/required_clearance_m",
                 rerun::Scalars(safety_distance_));
  recording_.log("solver/time_ms", rerun::Scalars(frame.plan.get().solve_ms));
  recording_.log("solver/objective",
                 rerun::Scalars(frame.plan.get().objective));
  recording_.log("solver/status", rerun::TextLog(frame.plan.get().status));
}

void
Visualizer::finish(SimulationResult const& result) const
{
  std::ostringstream summary;
  summary << result.status << " | goal error: " << result.final_error
          << " m | measured minimum clearance: " << result.min_clearance
          << " m";
  recording_.log("solver/status", rerun::TextLog(summary.str()));
  recording_.flush_blocking().throw_on_failure();
}
} // namespace polaris
