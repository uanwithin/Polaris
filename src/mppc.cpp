#include "polaris/mppc.hpp"

#include <nlopt.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace polaris
{
namespace
{
class ShootingProblem
{
public:
  ShootingProblem(State initial,
                  Control previous,
                  Vehicle vehicle,
                  MppcConfig const& config,
                  std::span<Vec2 const> points,
                  ReferencePath const& reference)
    : initial_(initial)
    , previous_(previous)
    , vehicle_(vehicle)
    , config_(config)
    , points_(points)
    , dimension_(static_cast<std::size_t>(2 * config.horizon))
  {
    auto const progress = reference.project({initial.x, initial.y});
    auto const goal = reference.sample(reference.length());
    auto const goal_distance =
        std::hypot(initial.x - goal.x, initial.y - goal.y);
    // Arrival taper is the only modification to the constant desired speed in
    // Eq. (10); it allows settling at the goal rather than driving through it.
    target_speed_ = std::min(config.desired_speed, goal_distance * 0.9);
    for(int h = 0; h < config.horizon; ++h)
    {
      references_.push_back(
          reference.sample(progress + (h + 1) * config.dt * target_speed_));
    }
  }

  [[nodiscard]] std::size_t
  constraint_count() const
  {
    return static_cast<std::size_t>(config_.horizon) * (points_.size() * 4 + 4);
  }

  void
  rollout(std::span<double const> controls)
  {
    if(std::ranges::equal(cached_, controls))
    {
      return;
    }
    cached_.assign(controls.begin(), controls.end());
    states_.assign(static_cast<std::size_t>(config_.horizon) + 1, initial_);
    jacobians_.assign(states_.size() * 3 * dimension_, 0.0);
    for(int h = 0; h < config_.horizon; ++h)
    {
      auto const step = static_cast<std::size_t>(h);
      auto const state = states_[step];
      auto const speed = controls[2 * step];
      auto const c = std::cos(state.yaw);
      auto const s = std::sin(state.yaw);
      states_[step + 1] =
          advance(state, {speed, controls[2 * step + 1]}, config_.dt);
      for(std::size_t j = 0; j < dimension_; ++j)
      {
        jacobian(step + 1, 0, j) =
            jacobian(step, 0, j) -
            config_.dt * speed * s * jacobian(step, 2, j);
        jacobian(step + 1, 1, j) =
            jacobian(step, 1, j) +
            config_.dt * speed * c * jacobian(step, 2, j);
        jacobian(step + 1, 2, j) = jacobian(step, 2, j);
      }
      jacobian(step + 1, 0, 2 * step) += config_.dt * c;
      jacobian(step + 1, 1, 2 * step) += config_.dt * s;
      jacobian(step + 1, 2, 2 * step + 1) += config_.dt;
    }
  }

  double
  objective(std::span<double const> controls, std::span<double> gradient)
  {
    rollout(controls);
    std::ranges::fill(gradient, 0.0);
    double cost{};
    for(std::size_t h = 0; h < references_.size(); ++h)
    {
      auto const state = states_[h + 1];
      auto const reference = references_[h];
      std::array<double, 3> const errors{
          state.x - reference.x,
          state.y - reference.y,
          wrap_angle(state.yaw - reference.yaw),
      };
      std::array<double, 3> const weights{
          config_.position_weight,
          config_.position_weight,
          config_.heading_weight,
      };
      for(std::size_t axis = 0; axis < 3; ++axis)
      {
        cost += weights.at(axis) * errors.at(axis) * errors.at(axis);
        for(std::size_t j = 0; j < gradient.size(); ++j)
        {
          gradient[j] += 2.0 * weights.at(axis) * errors.at(axis) *
                         jacobian(h + 1, axis, j);
        }
      }
      auto const speed_error = controls[2 * h] - target_speed_;
      cost += config_.speed_weight * speed_error * speed_error;
      if(!gradient.empty())
      {
        gradient[2 * h] += 2.0 * config_.speed_weight * speed_error;
      }
    }
    return cost;
  }

  void
  constraints(std::span<double const> controls,
              std::span<double> values,
              std::span<double> gradients)
  {
    rollout(controls);
    std::ranges::fill(gradients, 0.0);
    std::size_t row{};
    for(std::size_t h = 0; h < references_.size(); ++h)
    {
      for(double const fraction : {0.25, 0.5, 0.75, 1.0})
      {
        auto const pose = advance(states_[h],
                                  {controls[2 * h], controls[2 * h + 1]},
                                  config_.dt * fraction);
        for(auto const point : points_)
        {
          auto const distance = point_rectangle_distance(point, pose, vehicle_);
          values[row] = config_.safety_distance + config_.sampling_margin -
                        distance.value;
          if(!gradients.empty())
          {
            for(std::size_t j = 0; j < dimension_; ++j)
            {
              for(std::size_t axis = 0; axis < 3; ++axis)
              {
                auto const derivative =
                    (1.0 - fraction) * jacobian(h, axis, j) +
                    fraction * jacobian(h + 1, axis, j);
                gradients[row * dimension_ + j] -=
                    distance.gradient.at(axis) * derivative;
              }
            }
          }
          ++row;
        }
      }
      // Eq. (3), including the first control relative to the executed command.
      for(std::size_t axis = 0; axis < 2; ++axis)
      {
        auto const index = 2 * h + axis;
        auto const initial_control =
            axis == 0 ? previous_.speed : previous_.yaw_rate;
        auto const previous = h == 0 ? initial_control : controls[index - 2];
        auto const limit =
            config_.dt * (axis == 0 ? config_.max_acceleration
                                    : config_.max_yaw_acceleration);
        for(double const sign : {-1.0, 1.0})
        {
          values[row] = sign * (controls[index] - previous) - limit;
          if(!gradients.empty())
          {
            gradients[row * dimension_ + index] = sign;
            if(h > 0)
            {
              gradients[row * dimension_ + index - 2] = -sign;
            }
          }
          ++row;
        }
      }
    }
  }

  [[nodiscard]] bool
  validate(std::span<double const> controls)
  {
    for(std::size_t i = 0; i < controls.size(); ++i)
    {
      auto const low = i % 2 == 0 ? 0.0 : -config_.max_yaw_rate;
      auto const high = i % 2 == 0 ? config_.max_speed : config_.max_yaw_rate;
      if(!std::isfinite(controls[i]) ||
         controls[i] < low - config_.feasibility_tolerance ||
         controls[i] > high + config_.feasibility_tolerance)
      {
        return false;
      }
    }
    std::vector<double> values(constraint_count());
    constraints(controls, values, {});
    auto const per_step = points_.size() * 4 + 4;
    for(std::size_t row = 0; row < values.size(); ++row)
    {
      auto const tolerance =
          row % per_step < points_.size() * 4
              ? config_.sampling_margin + config_.feasibility_tolerance
              : config_.feasibility_tolerance;
      if(!std::isfinite(values[row]) || values[row] > tolerance)
      {
        return false;
      }
    }
    // Denser independent validation catches inter-knot penetration. This is
    // sampled validation, not a proof of continuous-time collision avoidance.
    for(std::size_t h = 0; h < references_.size(); ++h)
    {
      for(int sample = 1; sample <= 10; ++sample)
      {
        auto const state = advance(states_[h],
                                   {controls[2 * h], controls[2 * h + 1]},
                                   config_.dt * sample / 10.0);
        if(clearance(state, vehicle_, points_) <
           config_.safety_distance - config_.feasibility_tolerance)
        {
          return false;
        }
      }
    }
    return true;
  }

  [[nodiscard]] std::vector<double>
  seed(double turn, bool braking) const
  {
    std::vector<double> controls(dimension_);
    auto previous = previous_;
    for(std::size_t h = 0; h < references_.size(); ++h)
    {
      double target_turn{};
      if(h < references_.size() / 3)
      {
        target_turn = turn;
      }
      else if(h < 2 * references_.size() / 3)
      {
        target_turn = -turn;
      }
      previous.speed +=
          std::clamp((braking ? 0.0 : target_speed_) - previous.speed,
                     -config_.max_acceleration * config_.dt,
                     config_.max_acceleration * config_.dt);
      previous.yaw_rate +=
          std::clamp((braking ? 0.0 : target_turn) - previous.yaw_rate,
                     -config_.max_yaw_acceleration * config_.dt,
                     config_.max_yaw_acceleration * config_.dt);
      controls[2 * h] = previous.speed;
      controls[2 * h + 1] = previous.yaw_rate;
    }
    return controls;
  }

  [[nodiscard]] std::vector<State> const&
  states() const
  {
    return states_;
  }

  static double
  objective_callback(unsigned n, double const* x, double* gradient, void* data)
  {
    auto& problem = *static_cast<ShootingProblem*>(data);
    return problem.objective({x, n}, {gradient, gradient == nullptr ? 0U : n});
  }

  static void
  constraints_callback(unsigned m,
                       double* result,
                       unsigned n,
                       double const* x,
                       double* gradient,
                       void* data)
  {
    auto& problem = *static_cast<ShootingProblem*>(data);
    problem.constraints(
        {x, n},
        {result, m},
        {gradient, gradient == nullptr ? 0 : static_cast<std::size_t>(m) * n});
  }

private:
  double&
  jacobian(std::size_t h, std::size_t axis, std::size_t j)
  {
    return jacobians_[(h * 3 + axis) * dimension_ + j];
  }

  State initial_;
  Control previous_;
  Vehicle vehicle_;
  MppcConfig config_;
  std::span<Vec2 const> points_;
  std::size_t dimension_;
  double target_speed_{};
  std::vector<State> references_;
  std::vector<State> states_;
  std::vector<double> cached_;
  std::vector<double> jacobians_;
};
} // namespace

MppcPlanner::MppcPlanner(Vehicle vehicle, MppcConfig config)
  : vehicle_(vehicle)
  , config_(config)
{
  std::array const positives{
      vehicle.length,
      vehicle.width,
      config.dt,
      config.desired_speed,
      config.max_speed,
      config.max_yaw_rate,
      config.max_acceleration,
      config.max_yaw_acceleration,
      config.safety_distance,
      config.position_weight,
      config.heading_weight,
      config.speed_weight,
      config.feasibility_tolerance,
  };
  if(config.horizon < 3 || config.horizon > 100 || config.max_evaluations < 1 ||
     config.desired_speed > config.max_speed ||
     !std::isfinite(config.sampling_margin) || config.sampling_margin < 0.0 ||
     std::ranges::any_of(
         positives,
         [](double value) { return !std::isfinite(value) || value <= 0.0; }))
  {
    throw std::invalid_argument("Invalid MPPC configuration");
  }
}

MppcConfig const&
MppcPlanner::config() const
{
  return config_;
}
Vehicle
MppcPlanner::vehicle() const
{
  return vehicle_;
}

Plan
MppcPlanner::solve(State state,
                   Control previous,
                   std::span<Vec2 const> points,
                   ReferencePath const& reference)
{
  auto const start = std::chrono::steady_clock::now();
  Plan result;
  result.objective = std::numeric_limits<double>::infinity();
  // Drop only points provably unreachable over this horizon, using the maximum
  // translation plus the circumscribed footprint radius and safety distance.
  auto const radius = config_.horizon * config_.dt * config_.max_speed +
                      std::hypot(vehicle_.length, vehicle_.width) / 2.0 +
                      config_.safety_distance + config_.sampling_margin;
  std::vector<Vec2> relevant;
  for(auto const point : points)
  {
    if(!std::isfinite(point.x) || !std::isfinite(point.y))
    {
      throw std::invalid_argument("Non-finite lidar point");
    }
    if(squared_distance(point, {state.x, state.y}) <= radius * radius)
    {
      relevant.push_back(point);
    }
  }
  ShootingProblem problem(state,
                          previous,
                          vehicle_,
                          config_,
                          relevant,
                          reference);
  auto const dimension = static_cast<unsigned>(2 * config_.horizon);
  std::vector<double> lower(dimension);
  std::vector<double> upper(dimension);
  for(unsigned j = 0; j < dimension; ++j)
  {
    lower[j] = j % 2 == 0 ? 0.0 : -config_.max_yaw_rate;
    upper[j] = j % 2 == 0 ? config_.max_speed : config_.max_yaw_rate;
  }
  std::vector<std::vector<double>> seeds;
  if(!warm_start_.empty())
  {
    seeds.push_back(warm_start_);
  }
  seeds.push_back(problem.seed(0.0, false));
  seeds.push_back(problem.seed(0.75, false));
  seeds.push_back(problem.seed(-0.75, false));
  std::vector<double> best;
  auto const initial_clearance = clearance(state, vehicle_, points);
  if(initial_clearance >=
     config_.safety_distance - config_.feasibility_tolerance)
  {
    for(auto controls : seeds)
    {
      if(problem.validate(controls))
      {
        auto const seed_cost = problem.objective(controls, {});
        if(seed_cost < result.objective)
        {
          best = controls;
          result.objective = seed_cost;
          result.status = "feasible_seed";
        }
      }
      nlopt::opt optimizer(nlopt::LD_SLSQP, dimension);
      optimizer.set_lower_bounds(lower);
      optimizer.set_upper_bounds(upper);
      optimizer.set_min_objective(ShootingProblem::objective_callback,
                                  &problem);
      optimizer.add_inequality_mconstraint(
          ShootingProblem::constraints_callback,
          &problem,
          std::vector<double>(problem.constraint_count(),
                              config_.feasibility_tolerance));
      optimizer.set_ftol_rel(1e-5);
      optimizer.set_xtol_rel(1e-5);
      optimizer.set_maxeval(config_.max_evaluations);
      double cost{};
      int solver_code{};
      try
      {
        solver_code = optimizer.optimize(controls, cost);
      }
      catch(nlopt::roundoff_limited const&)
      {
        solver_code = nlopt::ROUNDOFF_LIMITED;
      }
      catch(std::runtime_error const&)
      {
        solver_code = nlopt::FAILURE;
      }
      result.evaluations += optimizer.get_numevals();
      if(problem.validate(controls))
      {
        cost = problem.objective(controls, {});
        if(cost < result.objective)
        {
          best = controls;
          result.objective = cost;
          result.status = "slsqp:" + std::to_string(solver_code);
        }
      }
      // A good warm start needs no multi-start search. Near an obstructed
      // reference, left/right seeds help escape the symmetric stopping minimum.
      if(!best.empty() && result.objective < 2.0)
      {
        break;
      }
    }
  }
  if(best.empty())
  {
    best = problem.seed(0.0, true);
    result.braking = true;
    result.status = "braking:no_feasible_optimized_sequence";
  }
  result.feasible = initial_clearance >= config_.safety_distance -
                                             config_.feasibility_tolerance &&
                    problem.validate(best);
  if(!result.feasible)
  {
    result.status = "infeasible:braking_not_safe";
  }
  result.objective = problem.objective(best, {});
  result.states = problem.states();
  result.min_clearance = initial_clearance;
  for(std::size_t h = 0; h < best.size() / 2; ++h)
  {
    result.controls.push_back({best[2 * h], best[2 * h + 1]});
    result.min_clearance =
        std::min(result.min_clearance,
                 clearance(result.states[h + 1], vehicle_, points));
  }
  result.command = result.controls.front();
  if(result.feasible)
  {
    warm_start_.assign(best.begin() + 2, best.end());
    warm_start_.push_back(best[best.size() - 2]);
    warm_start_.push_back(best.back());
  }
  else
  {
    warm_start_.clear();
  }
  result.solve_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start)
                        .count();
  return result;
}
} // namespace polaris
