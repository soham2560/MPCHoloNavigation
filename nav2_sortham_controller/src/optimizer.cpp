// Copyright (c) 2022 Samsung Research America, @artofnothingness Alexey Budyakov
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "nav2_sortham_controller/optimizer.hpp"

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <xtensor/xmath.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xnoalias.hpp>

#include "nav2_costmap_2d/costmap_filters/filter_values.hpp"
#include <casadi/casadi.hpp>
#include "tf2/utils.h"
#include <tf2/LinearMath/Quaternion.h>
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace sortham
{

using namespace xt::placeholders;  // NOLINT
using xt::evaluation_strategy::immediate;
using namespace casadi; // NOLINT

void Optimizer::initialize(
  rclcpp_lifecycle::LifecycleNode::WeakPtr parent, const std::string & name,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros,
  ParametersHandler * param_handler)
{
  parent_ = parent;
  name_ = name;
  costmap_ros_ = costmap_ros;
  costmap_ = costmap_ros_->getCostmap();
  parameters_handler_ = param_handler;

  auto node = parent_.lock();
  logger_ = node->get_logger();

  getParams();

  critic_manager_.on_configure(parent_, name_, costmap_ros_, parameters_handler_);
  noise_generator_.initialize(settings_, isHolonomic(), name_, parameters_handler_);

  reset();

  mpc_problem_defined_ = false;
  last_solve_successful_ = false;

  last_v_ = DM(0);
  last_w_ = DM(0);
  last_vy_ = DM(0);
}

void Optimizer::shutdown()
{
  noise_generator_.shutdown();
}

void Optimizer::getParams()
{
  std::string motion_model_name;

  auto & s = settings_;
  auto getParam = parameters_handler_->getParamGetter(name_);
  auto getParentParam = parameters_handler_->getParamGetter("");
  getParam(s.model_dt, "model_dt", 0.05f);
  getParam(s.time_steps, "time_steps", 56);
  getParam(s.batch_size, "batch_size", 1000);
  getParam(s.iteration_count, "iteration_count", 1);
  getParam(s.temperature, "temperature", 0.3f);
  getParam(s.gamma, "gamma", 0.015f);
  getParam(s.base_constraints.vx_max, "vx_max", 0.5);
  getParam(s.base_constraints.vx_min, "vx_min", -0.35);
  getParam(s.base_constraints.vy, "vy_max", 0.5);
  getParam(s.base_constraints.wz, "wz_max", 1.9);
  getParam(s.sampling_std.vx, "vx_std", 0.2);
  getParam(s.sampling_std.vy, "vy_std", 0.2);
  getParam(s.sampling_std.wz, "wz_std", 0.4);
  getParam(s.retry_attempt_limit, "retry_attempt_limit", 1);

  getParam(s.weight_x, "weight_x", 20.0);
  getParam(s.weight_y, "weight_y", 20.0);
  getParam(s.weight_theta, "weight_theta", 5.0);
  getParam(s.weight_v, "weight_v", 0.1);
  getParam(s.weight_w, "weight_w", 0.1);
  getParam(s.weight_vy, "weight_vy", 0.1);
  getParam(s.weight_v_accel, "weight_v_accel", 0.01);
  getParam(s.weight_w_accel, "weight_w_accel", 0.01);
  getParam(s.weight_vy_accel, "weight_vy_accel", 0.01);
  getParam(s.weight_terminal_x, "weight_terminal_x", s.weight_x);
  getParam(s.weight_terminal_y, "weight_terminal_y", s.weight_y);
  getParam(s.weight_terminal_theta, "weight_terminal_theta", s.weight_theta);
  getParam(s.max_v_change, "max_v_change", 0.1);
  getParam(s.max_w_change, "max_w_change", 0.5);
  getParam(s.max_vy_change, "max_vy_change", 0.1);


  getParam(motion_model_name, "motion_model", std::string("DiffDrive"));

  s.constraints = s.base_constraints;
  setMotionModel(motion_model_name);
  if (!isHolonomic()) {
    RCLCPP_WARN(logger_, "MPC currently setup assuming Omni/Holonomic model for controls [vx, vy, wz]");
    // Need modification in setupCasADiProblem and solveMPC if using DiffDrive/Ackermann
  }
  parameters_handler_->addPostCallback([this]() {
      RCLCPP_INFO(logger_, "Parameters reloaded, resetting optimizer and MPC problem.");
      reset();
      mpc_problem_defined_ = false; // Force re-setup on next eval
      last_solve_successful_ = false;
      last_v_ = DM(0);
      last_vy_ = DM(0);
      last_w_ = DM(0);
  });


  double controller_frequency;
  getParentParam(controller_frequency, "controller_frequency", 20.0); // Default 20Hz
  // setOffset(controller_frequency); // MPPI specific, not needed for MPC
}

// MPPI Specific - Unused by core MPC logic
void Optimizer::setOffset(double /*controller_frequency*/)
{
  // This logic is for MPPI's sequence shifting.
  // MPC applies the first control command directly.
  settings_.shift_control_sequence = false; // Ensure MPPI shifting is off
}

void Optimizer::reset()
{
  state_.reset(settings_.batch_size, settings_.time_steps);
  control_sequence_.reset(settings_.time_steps);
  control_history_[0] = {0.0, 0.0, 0.0};
  control_history_[1] = {0.0, 0.0, 0.0};
  control_history_[2] = {0.0, 0.0, 0.0};
  control_history_[3] = {0.0, 0.0, 0.0};

  settings_.constraints = settings_.base_constraints;

  costs_ = xt::zeros<float>({settings_.batch_size});
  generated_trajectories_.reset(settings_.batch_size, settings_.time_steps);

  noise_generator_.reset(settings_, isHolonomic());

  // Reset MPC state
  last_optimal_X_flat_ = casadi::DM();
  last_optimal_U_flat_ = casadi::DM();
  last_v_ = DM(0);
  last_w_ = DM(0);
  last_vy_ = DM(0);
  last_solve_successful_ = false;

  RCLCPP_INFO(logger_, "Optimizer reset (MPPI & MPC state)");
}

geometry_msgs::msg::TwistStamped Optimizer::evalControl(
  const geometry_msgs::msg::PoseStamped & robot_pose,
  const geometry_msgs::msg::Twist & robot_speed,
  const nav_msgs::msg::Path & plan,
  const geometry_msgs::msg::Pose & goal,
  nav2_core::GoalChecker * goal_checker)
{
  prepare(robot_pose, robot_speed, plan, goal, goal_checker);

  if (!mpc_problem_defined_) {
      RCLCPP_INFO(logger_, "Setting up CasADi NLP problem...");
      setupCasADiProblem();
      RCLCPP_INFO(logger_, "CasADi NLP problem setup complete.");
  }

  RCLCPP_DEBUG(logger_, "Attempting to solve MPC problem...");
  bool success = solveMPC();

  if (success) {
      RCLCPP_DEBUG(logger_, "MPC solve successful.");
      // Extract the first control command from the optimal sequence U*
      // Assuming U is ordered [vx0, vy0, wz0, vx1, vy1, wz1, ...]
      double vx_cmd = last_optimal_U_flat_(0).scalar();
      double vy_cmd = last_optimal_U_flat_(1).scalar();
      double wz_cmd = last_optimal_U_flat_(2).scalar();

      // Update last commands for rate limiting in next step
      last_v_ = vx_cmd;
      last_vy_ = vy_cmd;
      last_w_ = wz_cmd;

      // Create and return Twist message
      return utils::toTwistStamped(vx_cmd, vy_cmd, wz_cmd, plan.header.stamp, costmap_ros_->getBaseFrameID());
  } else {
      RCLCPP_WARN(logger_, "MPC optimization failed. Applying zero velocity command.");
      // Fallback: Apply zero velocity command
      last_v_ = DM(0);
      last_vy_ = DM(0);
      last_w_ = DM(0);
      last_solve_successful_ = false; // Ensure no warm start next time
      // Optionally, throw an exception or try MPPI fallback if implemented
      // throw nav2_core::ControllerException("MPC Optimizer failed to find a solution.");
      return utils::toTwistStamped(0.0, 0.0, 0.0, plan.header.stamp, costmap_ros_->getBaseFrameID());
  }
}

// MPPI Optimization Loop - Unused by core MPC logic
void Optimizer::optimize()
{
 /*
  for (size_t i = 0; i < settings_.iteration_count; ++i) {
    generateNoisedTrajectories();
    critic_manager_.evalTrajectoriesScores(critics_data_);
    updateControlSequence();
  }
  */
}

// MPPI Fallback Logic - Potentially adaptable for MPC failure
bool Optimizer::fallback(bool /*fail*/)
{
  /*
  static size_t counter = 0;

  if (!fail) {
    counter = 0;
    return false;
  }

  reset(); // Reset MPPI state, maybe MPC state too?
if (++counter > settings_.retry_attempt_limit) {
    counter = 0;
    // throw std::runtime_error("Optimizer fail to compute path");
    return false; // Indicate permanent failure after retries
  }
  RCLCPP_WARN(logger_, "Optimizer failed, retrying attempt %zu of %zu", counter, settings_.retry_attempt_limit);
  return true; // Indicate retry should happen
  */
 return false; // Basic MPC doesn't use this loop structure currently
}

void Optimizer::prepare(
  const geometry_msgs::msg::PoseStamped & robot_pose,
  const geometry_msgs::msg::Twist & robot_speed,
  const nav_msgs::msg::Path & plan,
  const geometry_msgs::msg::Pose & goal,
  nav2_core::GoalChecker * goal_checker)
{
  // Prepare state for both MPPI and MPC
  state_.pose = robot_pose;
  state_.speed = robot_speed;
  path_ = utils::toTensor(plan); // Keep tensor path for getReferencePoseHorizon
  goal_ = goal;

  // Reset MPPI cost buffer
  costs_.fill(0);

  // Reset MPPI critic data structure (partially reused?)
  critics_data_.fail_flag = false;
  critics_data_.goal_checker = goal_checker; // Needed? MPC doesn't use GoalChecker directly
  critics_data_.motion_model = motion_model_; // Needed? MPC uses its own CasADi model
  critics_data_.furthest_reached_path_point.reset();
  critics_data_.path_pts_valid.reset();
}

// MPPI Specific - Unused by core MPC logic
void Optimizer::shiftControlSequence()
{
 /*
  using namespace xt::placeholders;  // NOLINT
  control_sequence_.vx = xt::roll(control_sequence_.vx, -1);
  control_sequence_.wz = xt::roll(control_sequence_.wz, -1);


  xt::view(control_sequence_.vx, -1) =
    xt::view(control_sequence_.vx, -2);

  xt::view(control_sequence_.wz, -1) =
    xt::view(control_sequence_.wz, -2);


  if (isHolonomic()) {
    control_sequence_.vy = xt::roll(control_sequence_.vy, -1);
    xt::view(control_sequence_.vy, -1) =
      xt::view(control_sequence_.vy, -2);
  }
  */
}

// MPPI Specific - Unused by core MPC logic
void Optimizer::generateNoisedTrajectories()
{
 /*
  noise_generator_.setNoisedControls(state_, control_sequence_);
  noise_generator_.generateNextNoises();
  updateStateVelocities(state_);
  integrateStateVelocities(generated_trajectories_, state_);
  */
}

bool Optimizer::isHolonomic() const { return motion_model_->isHolonomic(); }

// MPPI Specific - Unused by core MPC logic
void Optimizer::applyControlSequenceConstraints()
{
 /*
  auto & s = settings_;

  if (isHolonomic()) {
    control_sequence_.vy = xt::clip(control_sequence_.vy, -s.constraints.vy, s.constraints.vy);
  }

  control_sequence_.vx = xt::clip(control_sequence_.vx, s.constraints.vx_min, s.constraints.vx_max);
  control_sequence_.wz = xt::clip(control_sequence_.wz, -s.constraints.wz, s.constraints.wz);

  motion_model_->applyConstraints(control_sequence_);
  */
}

// MPPI Specific - Unused by core MPC logic
void Optimizer::updateStateVelocities(
  models::State & /*state*/) const
{
 /*
  updateInitialStateVelocities(state);
  propagateStateVelocitiesFromInitials(state);
  */
}

// MPPI Specific - Unused by core MPC logic
void Optimizer::updateInitialStateVelocities(
  models::State & /*state*/) const
{
 /*
  xt::noalias(xt::view(state.vx, xt::all(), 0)) = state.speed.linear.x;
  xt::noalias(xt::view(state.wz, xt::all(), 0)) = state.speed.angular.z;

  if (isHolonomic()) {
    xt::noalias(xt::view(state.vy, xt::all(), 0)) = state.speed.linear.y;
  }
  */
}

// MPPI Specific - Unused by core MPC logic
void Optimizer::propagateStateVelocitiesFromInitials(
  models::State & /*state*/) const
{
 // motion_model_->predict(state);
}

// MPPI Specific - Unused by core MPC logic
// Can be adapted to visualize MPC predicted trajectory if needed
void Optimizer::integrateStateVelocities(
  xt::xtensor<float, 2> & /*trajectory*/,
  const xt::xtensor<float, 2> & /*sequence*/) const
{
 /*
  float initial_yaw = tf2::getYaw(state_.pose.pose.orientation);

  const auto vx = xt::view(sequence, xt::all(), 0);
  const auto vy = xt::view(sequence, xt::all(), 2); // Assumes holonomic if used
  const auto wz = xt::view(sequence, xt::all(), 1);
auto traj_x = xt::view(trajectory, xt::all(), 0);
  auto traj_y = xt::view(trajectory, xt::all(), 1);
  auto traj_yaws = xt::view(trajectory, xt::all(), 2);

  xt::noalias(traj_yaws) = xt::cumsum(wz * settings_.model_dt, 0) + initial_yaw;

  auto && yaw_cos = xt::xtensor<float, 1>::from_shape(traj_yaws.shape());
  auto && yaw_sin = xt::xtensor<float, 1>::from_shape(traj_yaws.shape());

  const auto yaw_offseted = xt::view(traj_yaws, xt::range(1, _));

  xt::noalias(xt::view(yaw_cos, 0)) = cosf(initial_yaw);
  xt::noalias(xt::view(yaw_sin, 0)) = sinf(initial_yaw);
  xt::noalias(xt::view(yaw_cos, xt::range(1, _))) = xt::cos(yaw_offseted);
  xt::noalias(xt::view(yaw_sin, xt::range(1, _))) = xt::sin(yaw_offseted);

  auto && dx = xt::eval(vx * yaw_cos);
  auto && dy = xt::eval(vx * yaw_sin);

  if (isHolonomic()) { // Assuming sequence has vy at index 2 if holonomic
    dx = dx - vy * yaw_sin;
    dy = dy + vy * yaw_cos;
  }

  xt::noalias(traj_x) = state_.pose.pose.position.x + xt::cumsum(dx * settings_.model_dt, 0);
  xt::noalias(traj_y) = state_.pose.pose.position.y + xt::cumsum(dy * settings_.model_dt, 0);
  */
}

// MPPI Specific - Unused by core MPC logic
void Optimizer::integrateStateVelocities(
  models::Trajectories & /*trajectories*/,
  const models::State & /*state*/) const
{
/*
  const float initial_yaw = tf2::getYaw(state.pose.pose.orientation);

  xt::noalias(trajectories.yaws) =
    xt::cumsum(state.wz * settings_.model_dt, 1) + initial_yaw;

  const auto yaws_cutted = xt::view(trajectories.yaws, xt::all(), xt::range(0, -1));

  auto && yaw_cos = xt::xtensor<float, 2>::from_shape(trajectories.yaws.shape());
  auto && yaw_sin = xt::xtensor<float, 2>::from_shape(trajectories.yaws.shape());
  xt::noalias(xt::view(yaw_cos, xt::all(), 0)) = cosf(initial_yaw);
  xt::noalias(xt::view(yaw_sin, xt::all(), 0)) = sinf(initial_yaw);
  xt::noalias(xt::view(yaw_cos, xt::all(), xt::range(1, _))) = xt::cos(yaws_cutted);
  xt::noalias(xt::view(yaw_sin, xt::all(), xt::range(1, _))) = xt::sin(yaws_cutted);

  auto && dx = xt::eval(state.vx * yaw_cos);
  auto && dy = xt::eval(state.vx * yaw_sin);

  if (isHolonomic()) {
    dx = dx - state.vy * yaw_sin;
    dy = dy + state.vy * yaw_cos;
  }

  xt::noalias(trajectories.x) = state.pose.pose.position.x +
    xt::cumsum(dx * settings_.model_dt, 1);
  xt::noalias(trajectories.y) = state.pose.pose.position.y +
    xt::cumsum(dy * settings_.model_dt, 1);
    */
}

// This should now return the MPC predicted optimal trajectory
xt::xtensor<float, 2> Optimizer::getOptimizedTrajectory()
{
  if (!last_solve_successful_ || last_optimal_X_flat_.is_empty()) {
      // Return empty or single point trajectory if no solution available
      return xt::xtensor<float, 2>::from_shape({0, 3});
  }

  size_t N = settings_.time_steps;
  std::vector<std::size_t> shape = {static_cast<std::size_t>(N + 1), 3ul};
  xt::xtensor<float, 2> trajectories = xt::zeros<float>(shape);

  // last_optimal_X_flat_ is (n_states_ * (N+1)) x 1
  // We need to reshape or access elements carefully
  // Assuming n_states_ = 3 (x, y, theta)
   if (n_states_ != 3) {
       RCLCPP_ERROR(logger_, "getOptimizedTrajectory assumes n_states_ = 3, but it is %d", n_states_);
       return xt::xtensor<float, 2>::from_shape({0, 3});
   }

  std::vector<double> x_vec = last_optimal_X_flat_.get_elements();

  for (size_t i = 0; i <= N; ++i) {
      trajectories(i, 0) = static_cast<float>(x_vec[i * n_states_ + 0]); // x
      trajectories(i, 1) = static_cast<float>(x_vec[i * n_states_ + 1]); // y
      trajectories(i, 2) = static_cast<float>(x_vec[i * n_states_ + 2]); // theta
  }

  return trajectories;
}
// MPPI Specific - Unused by core MPC logic
void Optimizer::updateControlSequence()
{
 /*
  auto & s = settings_;
  auto bounded_noises_vx = state_.cvx - control_sequence_.vx;
  auto bounded_noises_wz = state_.cwz - control_sequence_.wz;
  xt::noalias(costs_) +=
    s.gamma / powf(s.sampling_std.vx, 2) * xt::sum(
    xt::view(control_sequence_.vx, xt::newaxis(), xt::all()) * bounded_noises_vx, 1, immediate);
  xt::noalias(costs_) +=
    s.gamma / powf(s.sampling_std.wz, 2) * xt::sum(
    xt::view(control_sequence_.wz, xt::newaxis(), xt::all()) * bounded_noises_wz, 1, immediate);

  if (isHolonomic()) {
    auto bounded_noises_vy = state_.cvy - control_sequence_.vy;
    xt::noalias(costs_) +=
      s.gamma / powf(s.sampling_std.vy, 2) * xt::sum(
      xt::view(control_sequence_.vy, xt::newaxis(), xt::all()) * bounded_noises_vy,
      1, immediate);
  }

  auto && costs_normalized = costs_ - xt::amin(costs_, immediate);
  auto && exponents = xt::eval(xt::exp(-1 / settings_.temperature * costs_normalized));
  auto && softmaxes = xt::eval(exponents / xt::sum(exponents, immediate));
  auto && softmaxes_extened = xt::eval(xt::view(softmaxes, xt::all(), xt::newaxis()));

  xt::noalias(control_sequence_.vx) = xt::sum(state_.cvx * softmaxes_extened, 0, immediate);
  xt::noalias(control_sequence_.wz) = xt::sum(state_.cwz * softmaxes_extened, 0, immediate);
  if (isHolonomic()) {
    xt::noalias(control_sequence_.vy) = xt::sum(state_.cvy * softmaxes_extened, 0, immediate);
  }

  applyControlSequenceConstraints();
  */
}

// MPPI Specific - Unused by core MPC logic
geometry_msgs::msg::TwistStamped Optimizer::getControlFromSequenceAsTwist(
  const builtin_interfaces::msg::Time & /*stamp*/)
{
 /*
  unsigned int offset = settings_.shift_control_sequence ? 1 : 0;

  auto vx = control_sequence_.vx(offset);
  auto wz = control_sequence_.wz(offset);

  if (isHolonomic()) {
    auto vy = control_sequence_.vy(offset);
    return utils::toTwistStamped(vx, vy, wz, stamp, costmap_ros_->getBaseFrameID());
  }

  return utils::toTwistStamped(vx, wz, stamp, costmap_ros_->getBaseFrameID());
  */
 return geometry_msgs::msg::TwistStamped(); // Should not be called in MPC mode
}

void Optimizer::setMotionModel(const std::string & model)
{
  if (model == "DiffDrive") {
    motion_model_ = std::make_shared<DiffDriveMotionModel>();
  } else if (model == "Omni") {
    motion_model_ = std::make_shared<OmniMotionModel>();
  } else if (model == "Ackermann") {
    motion_model_ = std::make_shared<AckermannMotionModel>(parameters_handler_, name_);
  } else {
    throw std::runtime_error(
            std::string(
              "Model " + model + " is not valid! Valid options are DiffDrive, Omni, "
              "or Ackermann"));
  }
  RCLCPP_INFO(logger_, "Using motion model: %s", model.c_str());
}
void Optimizer::setSpeedLimit(double speed_limit, bool percentage)
{
  // This function needs to update settings_.constraints which are used
  // by the MPC setup for bounds.
  auto & s = settings_;
  const double epsilon = 1e-5; // For floating point comparison
  if (speed_limit < epsilon) // Treat 0 or negative as no speed limit
  {
    s.constraints.vx_max = s.base_constraints.vx_max;
    s.constraints.vx_min = s.base_constraints.vx_min;
    s.constraints.vy = s.base_constraints.vy;
    s.constraints.wz = s.base_constraints.wz;
     RCLCPP_INFO(logger_, "Speed limit reset to base constraints.");
  } else {
    double ratio = 1.0;
    if (percentage) {
      ratio = speed_limit / 100.0;
      RCLCPP_INFO(logger_, "Setting speed limit to %.2f%% of base limits.", speed_limit);
    } else {
      // Absolute limit - calculate ratio based on max linear speed
      if (s.base_constraints.vx_max > epsilon) {
         ratio = speed_limit / s.base_constraints.vx_max;
         RCLCPP_INFO(logger_, "Setting speed limit to absolute %.2f m/s (ratio %.2f).", speed_limit, ratio);
      } else {
          RCLCPP_WARN(logger_, "Cannot set absolute speed limit: base vx_max is zero.");
          return; // Keep existing constraints
      }
    }
    // Apply ratio to all limits (consider if wz should be scaled differently)
    s.constraints.vx_max = s.base_constraints.vx_max * ratio;
    // Ensure min_vx doesn't become positive if base_vx_min is negative
    s.constraints.vx_min = (s.base_constraints.vx_min < 0) ? (s.base_constraints.vx_min * ratio) : 0.0;
    s.constraints.vy = s.base_constraints.vy * ratio; // Assumes vy_max is positive
    s.constraints.wz = s.base_constraints.wz * ratio; // Assumes wz_max is positive
  }

  // Important: If the MPC problem is already defined, we need to update its bounds.
  // This simple implementation requires recreating the problem or updating bounds directly.
  // For now, force recreation on parameter change via the post-callback.
  RCLCPP_INFO(logger_, "New velocity limits: vx[%.2f, %.2f], vy +/-%.2f, wz +/-%.2f",
              s.constraints.vx_min, s.constraints.vx_max, s.constraints.vy, s.constraints.wz);
}

// This returns MPPI trajectories, becomes less useful.
// Could be adapted to return the single MPC trajectory in the required format if needed.
models::Trajectories & Optimizer::getGeneratedTrajectories()
{
  // Return empty or potentially the optimized trajectory if visualization expects it
  // For now, return the (likely empty) MPPI structure
  return generated_trajectories_;
}

//------------------------------------------------------------------------------
// MPC Specific Functions
//------------------------------------------------------------------------------

void Optimizer::setupCasADiProblem()
{
    // Define Problem Dimensions
    n_states_ = 3; // x, y, theta
    n_controls_ = 3; // vx, vy, wz (assuming Omni/Mecanum)
    int N = settings_.time_steps; // Prediction horizon
    double T = settings_.model_dt; // Time step

    // Symbolic Variables
    X_sym_ = MX::sym("X", n_states_, N + 1); // States over N+1 time steps
    U_sym_ = MX::sym("U", n_controls_, N);   // Controls over N time steps

    // Parameters (initial state, reference trajectory, last control)
    int n_ref_params = n_states_ * (N + 1);
    int n_init_state_params = n_states_;
    int n_last_control_params = n_controls_;
    n_params_ = n_init_state_params + n_ref_params + n_last_control_params;
    P_sym_ = MX::sym("P", n_params_);

    // Decision Variables Vector (flattened states and controls)
    MX V = vertcat(reshape(X_sym_, n_states_ * (N + 1), 1),
                   reshape(U_sym_, n_controls_ * N, 1));
    n_opt_vars_ = V.rows();

    // Objective Function (Cost Function J)
    MX J = 0;
    MX x_init = P_sym_(Slice(0, n_init_state_params)); // x0, y0, th0
    MX U_last = P_sym_(Slice(n_params_ - n_last_control_params, n_params_)); // vx_last, vy_last, wz_last
for (int k = 0; k < N; ++k) {
        MX X_k = X_sym_(Slice(), k);
        MX U_k = U_sym_(Slice(), k);
        MX X_ref_k = P_sym_(Slice(n_init_state_params + k * n_states_,
                                n_init_state_params + (k + 1) * n_states_));

        // State Cost (Tracking Error)
        MX state_error = X_k - X_ref_k;
        // Normalize angle error
        state_error(2) = atan2(sin(state_error(2)), cos(state_error(2)));
        J += settings_.weight_x * pow(state_error(0), 2);
        J += settings_.weight_y * pow(state_error(1), 2);
        J += settings_.weight_theta * pow(state_error(2), 2);

        // Control Cost (Effort)
        J += settings_.weight_v * pow(U_k(0), 2);
        J += settings_.weight_vy * pow(U_k(1), 2); // Omni
        J += settings_.weight_w * pow(U_k(2), 2);

        // Control Rate Cost (Smoothness)
        MX U_prev = (k == 0) ? U_last : U_sym_(Slice(), k - 1);
        MX delta_U = U_k - U_prev;
        J += settings_.weight_v_accel * pow(delta_U(0), 2);
        J += settings_.weight_vy_accel * pow(delta_U(1), 2); // Omni
        J += settings_.weight_w_accel * pow(delta_U(2), 2);
    }

    // Terminal Cost
    MX X_N = X_sym_(Slice(), N);
    MX X_ref_N = P_sym_(Slice(n_init_state_params + N * n_states_,
                             n_init_state_params + (N + 1) * n_states_));
    MX terminal_error = X_N - X_ref_N;
    terminal_error(2) = atan2(sin(terminal_error(2)), cos(terminal_error(2)));
    J += settings_.weight_terminal_x * pow(terminal_error(0), 2);
    J += settings_.weight_terminal_y * pow(terminal_error(1), 2);
    J += settings_.weight_terminal_theta * pow(terminal_error(2), 2);

    // Constraints (Dynamics, Initial State)
    MXVector g;

    // Initial State Constraint
    g.push_back(X_sym_(Slice(), 0) - x_init);

    // Dynamics Constraints (using Euler integration - Omni Model)
    for (int k = 0; k < N; ++k) {
        MX X_k = X_sym_(Slice(), k);
        MX U_k = U_sym_(Slice(), k);
        MX X_k_plus_1 = X_sym_(Slice(), k + 1);

        MX theta_k = X_k(2);
        MX vx_k = U_k(0);
        MX vy_k = U_k(1);
        MX wz_k = U_k(2);

        // State prediction using the model
        MX x_next = X_k(0) + (vx_k * cos(theta_k) - vy_k * sin(theta_k)) * T;
        MX y_next = X_k(1) + (vx_k * sin(theta_k) + vy_k * cos(theta_k)) * T;
        MX theta_next = X_k(2) + wz_k * T;
        // Normalize angle ? Not strictly necessary for dynamics constraint itself

        MX X_k_plus_1_predicted = vertcat(x_next, y_next, theta_next);

        // Add dynamics defect constraint (X_k+1 - predicted = 0)
        g.push_back(X_k_plus_1 - X_k_plus_1_predicted);
    }

    // Concatenate constraints
    MX g_flat = vertcat(g);
    n_constraints_ = g_flat.rows();

    // NLP Definition
    MXDict nlp = {{"x", V}, {"f", J}, {"g", g_flat}, {"p", P_sym_}};

    // Create Solver
    Dict solver_opts;
    solver_opts["ipopt.print_level"] = 0; // Suppress IPOPT output
    solver_opts["ipopt.sb"] = "yes";      // Suppress IPOPT banner
    solver_opts["print_time"] = 0;
    // Add other IPOPT options if needed (e.g., tol, max_iter)
    // solver_opts["ipopt.tol"] = 1e-4;
    // solver_opts["ipopt.max_iter"] = 50;
    solver_func_ = nlpsol("solver", "ipopt", nlp, solver_opts);

    // --- Define Fixed Bounds ---
    x_flat_lower_bounds_.resize(n_opt_vars_);
    x_flat_upper_bounds_.resize(n_opt_vars_);
    g_flat_lower_bounds_.resize(n_constraints_);
    g_flat_upper_bounds_.resize(n_constraints_);

    // State bounds (X): Set loosely initially, refine if needed
    for (int i = 0; i < n_states_ * (N + 1); ++i) {
        x_flat_lower_bounds_[i] = -inf;
        x_flat_upper_bounds_[i] = inf;
    }
    // Override angle bounds if desired [-pi, pi] or similar
    for (int k = 0; k <= N; ++k) {
         x_flat_lower_bounds_[k * n_states_ + 2] = -10*M_PI; // Looser than -pi for solver freedom
         x_flat_upper_bounds_[k * n_states_ + 2] = 10*M_PI; // Looser than pi
    }
    // Control bounds (U)
    int u_offset = n_states_ * (N + 1);
    for (int k = 0; k < N; ++k) {
        // vx
        x_flat_lower_bounds_[u_offset + k * n_controls_ + 0] = settings_.constraints.vx_min;
        x_flat_upper_bounds_[u_offset + k * n_controls_ + 0] = settings_.constraints.vx_max;
        // vy
        x_flat_lower_bounds_[u_offset + k * n_controls_ + 1] = -settings_.constraints.vy;
        x_flat_upper_bounds_[u_offset + k * n_controls_ + 1] = settings_.constraints.vy;
        // wz
        x_flat_lower_bounds_[u_offset + k * n_controls_ + 2] = -settings_.constraints.wz;
        x_flat_upper_bounds_[u_offset + k * n_controls_ + 2] = settings_.constraints.wz;
    }

    // Constraint bounds (g)
    // Initial state constraint: lb = ub = 0 (set dynamically later)
    // Dynamics constraints: lb = ub = 0
    for (int i = 0; i < n_constraints_; ++i) {
        g_flat_lower_bounds_[i] = 0;
        g_flat_upper_bounds_[i] = 0;
    }

    // Store fixed bounds as DM (CasADi matrix type)
    lbx_fixed_ = DM(x_flat_lower_bounds_);
    ubx_fixed_ = DM(x_flat_upper_bounds_);
    lbg_fixed_ = DM(g_flat_lower_bounds_);
    ubg_fixed_ = DM(g_flat_upper_bounds_);

    mpc_problem_defined_ = true;
}

bool Optimizer::solveMPC()
{
    int N = settings_.time_steps;

    // 1. Prepare Parameters (P)
    std::vector<double> p_vec;
    p_vec.reserve(n_params_);

    // Initial State (x0, y0, theta0)
    double current_x = state_.pose.pose.position.x;
    double current_y = state_.pose.pose.position.y;
    double current_theta = tf2::getYaw(state_.pose.pose.orientation);
    p_vec.push_back(current_x);
    p_vec.push_back(current_y);
    p_vec.push_back(current_theta);

    // Reference Trajectory (x_ref, y_ref, theta_ref for k=0 to N)
    std::vector<geometry_msgs::msg::Pose> ref_poses = getReferencePoseHorizon(state_.pose.pose, N, settings_.model_dt);
    if (ref_poses.size() != static_cast<size_t>(N + 1)) {
         RCLCPP_ERROR(logger_, "Reference trajectory generation failed. Expected %d poses, got %zu", N + 1, ref_poses.size());
         return false;
    }
    for (const auto &pose : ref_poses) {
        p_vec.push_back(pose.position.x);
        p_vec.push_back(pose.position.y);
        p_vec.push_back(tf2::getYaw(pose.orientation));
    }

    // Last Control Command (for rate limits)
    p_vec.push_back(last_v_.scalar());
    p_vec.push_back(last_vy_.scalar());
    p_vec.push_back(last_w_.scalar());

    DM p = DM(p_vec);

    // 2. Prepare Initial Guess (x0_guess)
    DM x0_guess;
    if (last_solve_successful_ && !last_optimal_X_flat_.is_empty() && !last_optimal_U_flat_.is_empty()) {
        // Warm start: Shift previous solution
        DM X_prev = reshape(last_optimal_X_flat_, n_states_, N + 1);
        DM U_prev = reshape(last_optimal_U_flat_, n_controls_, N);

        DM X_guess = horzcat(X_prev(Slice(), Slice(1, N + 1)), X_prev(Slice(), N)); // Shift state, repeat last
        DM U_guess = horzcat(U_prev(Slice(), Slice(1, N)), U_prev(Slice(), N-1)); // Shift control, repeat last
        x0_guess = vertcat(reshape(X_guess, n_states_ * (N + 1), 1),
                           reshape(U_guess, n_controls_ * N, 1));

    } else {
        // Cold start: Repeat initial state, zero controls
        std::vector<double> x0_vec(n_opt_vars_, 0.0);
        // Fill states with initial state
        for (int k = 0; k <= N; ++k) {
            x0_vec[k * n_states_ + 0] = current_x;
            x0_vec[k * n_states_ + 1] = current_y;
            x0_vec[k * n_states_ + 2] = current_theta;
        }
        // Controls are already zero initialized
        x0_guess = DM(x0_vec);
    }

    // 3. Prepare Bounds (lbx, ubx, lbg, ubg)
    DM lbx = lbx_fixed_;
    DM ubx = ubx_fixed_;
    DM lbg = lbg_fixed_;
    DM ubg = ubg_fixed_;
    // Update bounds for the initial state constraint X(0) == P(0:2)
    for (int i = 0; i < n_states_; ++i) {
        lbx(i) = p_vec[i]; // Fix initial state variable to parameter value
        ubx(i) = p_vec[i];
        // Corresponding constraint g(0:2) is X(0) - P(0:2), bounds are 0
        lbg(i) = 0;
        ubg(i) = 0;
    }
    // Note: Dynamics constraints g(3:) bounds remain 0

    // 4. Call Solver
    DMDict arg = {{"p", p}, {"x0", x0_guess}, {"lbx", lbx}, {"ubx", ubx}, {"lbg", lbg}, {"ubg", ubg}};
    DMDict res;
    try {
       res = solver_func_(arg);
    } catch (const std::exception &e) {
        RCLCPP_ERROR(logger_, "CasADi solver failed with exception: %s", e.what());
        last_solve_successful_ = false;
        return false;
    }


    // 5. Process Results
    if (solver_func_.stats().at("success")) {
        DM sol = res.at("x");
        // Split solution back into states and controls
        last_optimal_X_flat_ = sol(Slice(0, n_states_ * (N + 1)));
        last_optimal_U_flat_ = sol(Slice(n_states_ * (N + 1), n_opt_vars_));
        last_solve_successful_ = true;
        return true;
    } else {
        RCLCPP_WARN(logger_, "MPC solver did not find an optimal solution. Status: %s",
                   static_cast<std::string>(solver_func_.stats().at("return_status")).c_str());
        last_solve_successful_ = false;
        // Clear previous solution to avoid bad warm start
        last_optimal_X_flat_ = casadi::DM();
        last_optimal_U_flat_ = casadi::DM();
        return false;
    }
}


std::vector<geometry_msgs::msg::Pose> Optimizer::getReferencePoseHorizon(
        const geometry_msgs::msg::Pose& current_robot_pose, int N, double T)
{
    std::vector<geometry_msgs::msg::Pose> ref_poses;
    ref_poses.reserve(N + 1);

    const auto& global_plan = path_; // Using the xtensor path_ member
    if (global_plan.x.shape(0) < 2) {
        RCLCPP_WARN(logger_, "Global plan is too short (%zu points) to generate reference horizon.", global_plan.x.shape(0));
        // Return horizon repeating current pose maybe? Or empty? Let's return current pose
        for (int i=0; i <= N; ++i) {
            ref_poses.push_back(current_robot_pose);
        }
        return ref_poses;
    }

    // Find the closest point on the path to the current robot pose
    double min_dist_sq = std::numeric_limits<double>::max();
    size_t closest_idx = 0;
    double current_x = current_robot_pose.position.x;
    double current_y = current_robot_pose.position.y;

    for (size_t i = 0; i < global_plan.x.shape(0); ++i) {
        double dx = global_plan.x(i) - current_x;
        double dy = global_plan.y(i) - current_y;
        double dist_sq = dx * dx + dy * dy;
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            closest_idx = i;
        }
    }

    // Estimate a desired speed along the path (simple constant speed for now)
    // Could be dynamic based on curvature, proximity to goal, etc.
    double desired_speed = settings_.constraints.vx_max * 0.8; // Target 80% of max speed
    if (desired_speed < 0.1) desired_speed = 0.1; // Minimum speed to ensure progress

    // Calculate cumulative distances along the path starting from the closest point
    std::vector<double> cumulative_dist;
    cumulative_dist.push_back(0.0);
    double total_path_dist = 0;
    for (size_t i = closest_idx; i < global_plan.x.shape(0) - 1; ++i) {
         double dx = global_plan.x(i + 1) - global_plan.x(i);
         double dy = global_plan.y(i + 1) - global_plan.y(i);
         total_path_dist += std::sqrt(dx*dx + dy*dy);
         cumulative_dist.push_back(total_path_dist);
    }

    // Generate reference poses for each time step k=0 to N
    size_t current_segment_idx = 0; // Index relative to cumulative_dist / path segment start
    for (int k = 0; k <= N; ++k) {
        double target_dist_k = k * desired_speed * T;
// Find the path segment where the target distance falls
        while (current_segment_idx < cumulative_dist.size() - 1 &&
               cumulative_dist[current_segment_idx + 1] < target_dist_k)
        {
            current_segment_idx++;
        }

        geometry_msgs::msg::Pose ref_pose_k;
        size_t global_idx0 = closest_idx + current_segment_idx;
        size_t global_idx1 = closest_idx + current_segment_idx + 1;

        if (global_idx1 >= global_plan.x.shape(0)) {
            // Target distance is beyond the end of the path, clamp to the last pose
             ref_pose_k.position.x = global_plan.x(global_plan.x.shape(0)-1);
             ref_pose_k.position.y = global_plan.y(global_plan.x.shape(0)-1);
             tf2::Quaternion q;
             q.setRPY(0, 0, global_plan.yaws(global_plan.x.shape(0)-1)); // Access the last yaw value
             ref_pose_k.orientation = tf2::toMsg(q);
        } else {
             // Interpolate within the segment
             double segment_len = cumulative_dist[current_segment_idx + 1] - cumulative_dist[current_segment_idx];
             double dist_into_segment = target_dist_k - cumulative_dist[current_segment_idx];
             double ratio = (segment_len > 1e-6) ? (dist_into_segment / segment_len) : 0.0;
             ratio = std::max(0.0, std::min(1.0, ratio)); // Clamp ratio

             double x0 = global_plan.x(global_idx0);
             double y0 = global_plan.y(global_idx0);
             double yaw0 = global_plan.yaws(global_idx0); // Use yaws(index)
             double x1 = global_plan.x(global_idx1);
             double y1 = global_plan.y(global_idx1);
             double yaw1 = global_plan.yaws(global_idx1);

             ref_pose_k.position.x = x0 + ratio * (x1 - x0);
             ref_pose_k.position.y = y0 + ratio * (y1 - y0);
             // Interpolate yaw (handle wraparound - simple linear interp for now)
             double dyaw = yaw1 - yaw0;
             dyaw = atan2(sin(dyaw), cos(dyaw)); // Normalize angle difference
             double interpolated_yaw = yaw0 + ratio * dyaw;
             interpolated_yaw = atan2(sin(interpolated_yaw), cos(interpolated_yaw)); // Normalize result
             tf2::Quaternion q_interp;
             q_interp.setRPY(0, 0, interpolated_yaw);
             ref_pose_k.orientation = tf2::toMsg(q_interp);
        }
        ref_poses.push_back(ref_pose_k);
    }

    return ref_poses;
}


}  // namespace sortham