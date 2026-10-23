#pragma once

#include "polaris/simulation.hpp"

#include <rerun.hpp>

#include <string>

namespace polaris
{
class Visualizer
{
public:
  Visualizer(Scenario const& scenario,
             Vehicle vehicle,
             double safety_distance,
             std::string const& recording,
             bool spawn);
  void
  log(Frame const& frame) const;
  void
  finish(SimulationResult const& result) const;

private:
  rerun::RecordingStream recording_;
  Vehicle vehicle_;
  double safety_distance_;
};
} // namespace polaris
