#include "polaris/simulation.hpp"

#ifdef POLARIS_WITH_RERUN
#include "polaris/visualizer.hpp"
#endif

#include <charconv>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

namespace
{
struct Options
{
  std::string scenario{"avoidance"};
  std::string output;
  std::string csv;
  int steps{300};
  bool headless{};
  bool realtime{};
  bool help{};
};

Options
parse_options(std::span<char const* const> args)
{
  Options options;
  for(std::size_t i = 1; i < args.size(); ++i)
  {
    std::string_view const arg(args[i]);
    if(arg == "--headless")
    {
      options.headless = true;
    }
    else if(arg == "--realtime")
    {
      options.realtime = true;
    }
    else if(arg == "--help" || arg == "-h")
    {
      options.help = true;
    }
    else if(arg == "--scenario" || arg == "--output" || arg == "--csv" ||
            arg == "--steps")
    {
      if(++i >= args.size())
      {
        throw std::invalid_argument("Missing value for " + std::string(arg));
      }
      std::string_view const value(args[i]);
      if(arg == "--scenario")
      {
        options.scenario = value;
      }
      else if(arg == "--output")
      {
        options.output = value;
      }
      else if(arg == "--csv")
      {
        options.csv = value;
      }
      else
      {
        auto const [end, error] = std::from_chars(value.data(),
                                                  value.data() + value.size(),
                                                  options.steps);
        if(error != std::errc{} || end != value.data() + value.size() ||
           options.steps <= 0)
        {
          throw std::invalid_argument("--steps needs a positive integer");
        }
      }
    }
    else
    {
      throw std::invalid_argument("Unknown option: " + std::string(arg));
    }
  }
  return options;
}

void
create_parent(std::string const& filename)
{
  auto const parent = std::filesystem::path(filename).parent_path();
  if(!parent.empty())
  {
    std::filesystem::create_directories(parent);
  }
}
} // namespace


int
main(int argc, char const* const* argv)
{
  try
  {
    auto const options = parse_options({argv, static_cast<std::size_t>(argc)});
    if(options.help)
    {
      std::cout
          << "Polaris MPPC numerical baseline (C++20)\n"
             "  --scenario avoidance|empty|blocked  (default: avoidance)\n"
             "  --headless       Do not spawn the Rerun viewer\n"
             "  --output FILE    Save a Rerun .rrd recording (no viewer)\n"
             "  --csv FILE       Save per-step telemetry\n"
             "  --steps N        Maximum simulation steps (default: 300)\n"
             "  --realtime       Pace simulation to its 0.2 s time step\n"
             "Exit: 0 goal reached, 2 blocked/infeasible/step limit, 3 "
             "collision, 1 error\n";
      return 0;
    }
    auto const scenario = polaris::make_scenario(options.scenario);
    polaris::MppcPlanner planner;
#ifdef POLARIS_WITH_RERUN
    std::unique_ptr<polaris::Visualizer> visualizer;
    if(!options.output.empty() || !options.headless)
    {
      create_parent(options.output);
      visualizer = std::make_unique<polaris::Visualizer>(
          scenario,
          planner.vehicle(),
          planner.config().safety_distance,
          options.output,
          options.output.empty());
    }
#else
    if(!options.output.empty() || !options.headless)
    {
      throw std::runtime_error("Built without Rerun; use --headless or rebuild "
                               "with POLARIS_WITH_RERUN=ON");
    }
#endif
    std::ofstream csv;
    if(!options.csv.empty())
    {
      create_parent(options.csv);
      csv.open(options.csv);
      if(!csv)
      {
        throw std::runtime_error("Cannot open CSV output");
      }
      csv << "step,time,x,y,yaw,speed,yaw_rate,scan_points,predicted_clearance,"
             "objective,solve_ms,feasible,braking,status\n";
      csv << std::setprecision(10);
    }
    auto const wall_start = std::chrono::steady_clock::now();
    auto const result = polaris::simulate(
        scenario,
        planner,
        options.steps,
        [&](polaris::Frame const& frame)
        {
#ifdef POLARIS_WITH_RERUN
          if(visualizer)
          {
            visualizer->log(frame);
          }
#endif
          if(csv.is_open())
          {
            csv << frame.step << ',' << frame.time << ',' << frame.state.x
                << ',' << frame.state.y << ',' << frame.state.yaw << ','
                << frame.plan.get().command.speed << ','
                << frame.plan.get().command.yaw_rate << ',' << frame.scan.size()
                << ',' << frame.plan.get().min_clearance << ','
                << frame.plan.get().objective << ','
                << frame.plan.get().solve_ms << ',' << frame.plan.get().feasible
                << ',' << frame.plan.get().braking << ','
                << frame.plan.get().status << '\n';
          }
          if(frame.step % 20 == 0)
          {
            std::cout << "step=" << frame.step << " position=(" << frame.state.x
                      << ',' << frame.state.y
                      << ") v=" << frame.plan.get().command.speed
                      << " solve_ms=" << frame.plan.get().solve_ms << '\n';
          }
          if(options.realtime)
          {
            std::this_thread::sleep_until(
                wall_start + std::chrono::duration<double>(frame.time));
          }
        });
#ifdef POLARIS_WITH_RERUN
    if(visualizer)
    {
      visualizer->finish(result);
    }
#endif
    std::cout << std::fixed << std::setprecision(4)
              << "status=" << result.status << " steps=" << result.steps
              << " goal_error=" << result.final_error
              << " min_clearance=" << result.min_clearance
              << " max_tracking_error=" << result.max_tracking_error
              << " mean_solve_ms=" << result.mean_solve_ms
              << " max_solve_ms=" << result.max_solve_ms
              << " braking_steps=" << result.braking_steps << '\n';
    if(result.collision)
    {
      return 3;
    }
    return result.reached_goal ? 0 : 2;
  }
  catch(std::exception const& error)
  {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
