#pragma once

#include "polaris/geometry.hpp"
#include "polaris/reference_path.hpp"

#include <span>
#include <string>
#include <vector>

namespace polaris
{
struct MppcConfig
{
  int horizon{18};
  double dt{0.2};
  double desired_speed{1.0};
  double max_speed{1.2};
  double max_yaw_rate{1.2};
  double max_acceleration{1.0};
  double max_yaw_acceleration{2.5};
  double safety_distance{0.22};
  double sampling_margin{0.04};
  double position_weight{3.0};
  double heading_weight{0.15};
  double speed_weight{0.4};
  int max_evaluations{65};
  double feasibility_tolerance{1e-5};
};

struct Plan
{
  Control command;
  std::vector<State> states;
  std::vector<Control> controls;
  bool feasible{};
  bool braking{};
  double objective{};
  double min_clearance{};
  double solve_ms{};
  int evaluations{};
  std::string status;
};

// A numerical baseline of paper Eqs. (1)-(11). No map, learned features,
// obstacle clustering, A*, DWA, or neural policy is used by this class.
class MppcPlanner
{
public:
  explicit MppcPlanner(Vehicle vehicle = Vehicle(),
                       MppcConfig config = MppcConfig());
  [[nodiscard]] Plan
  solve(State state,
        Control previous,
        std::span<Vec2 const> points,
        ReferencePath const& reference);
  [[nodiscard]] MppcConfig const&
  config() const;
  [[nodiscard]] Vehicle
  vehicle() const;

private:
  Vehicle vehicle_;
  MppcConfig config_;
  std::vector<double> warm_start_;
};
} // namespace polaris
