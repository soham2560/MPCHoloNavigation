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
#include <numeric>
#include <algorithm>
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

  getParam(robot_radius_, "robot_radius", 0.25);
  getParam(lidar_safety_margin_, "lidar_safety_margin", 0.05);
  getParam(lidar_cluster_tolerance_, "lidar_cluster_tolerance", 0.20);
  getParam(lidar_min_cluster_size_, "lidar_min_cluster_size", 3);
  getParam(max_obstacles_, "max_obstacles", 10);
  effective_radius_padding_sq_ = std::pow(robot_radius_ + lidar_safety_margin_, 2);
  RCLCPP_INFO(logger_, "Lidar constraints: radius=%.2f, margin=%.2f, cluster_tol=%.2f, min_cluster=%d, max_obs=%d",
        robot_radius_, lidar_safety_margin_, lidar_cluster_tolerance_, lidar_min_cluster_size_, max_obstacles_);


  getParam(motion_model_name, "motion_model", std::string("DiffDrive"));

  s.constraints = s.base_constraints;
  setMotionModel(motion_model_name);
  if (!isHolonomic()) {
    RCLCPP_WARN(logger_, "MPC currently setup assuming Omni/Holonomic model for controls [vx, vy, wz]");
    // Need modification in setupCasADiProblem and solveMPC if using DiffDrive/Ackermann
  }
  parameters_handler_->addPostCallback([this]() {
      RCLCPP_INFO(logger_, "Parameters reloaded, resetting optimizer and MPC problem.");
      getParams();
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
  const std::vector<geometry_msgs::msg::Point> & obstacle_points, // <<< Use this name >>>
  nav2_core::GoalChecker * /*goal_checker*/) // Goal checker not used directly by MPC core
{
  // Store current state and plan info for use in solveMPC
  state_.pose = robot_pose;
  state_.speed = robot_speed;
  path_ = utils::toTensor(plan); // Convert plan to internal format
  goal_ = goal;

  // Ensure MPC problem structure is defined
  if (!mpc_problem_defined_) {
      RCLCPP_INFO(logger_, "Setting up CasADi NLP problem for the first time...");
      setupCasADiProblem();
      RCLCPP_INFO(logger_, "CasADi NLP problem setup complete.");
  }

  RCLCPP_DEBUG(logger_, "Attempting to solve MPC problem...");
  bool success = solveMPC(obstacle_points); // <<< Pass obstacle points >>>

  if (success) {
      RCLCPP_DEBUG(logger_, "MPC solve successful.");
      // Extract the first control command from the optimal sequence U*
      // U* is stored flat in last_optimal_U_flat_
      double vx_cmd = last_optimal_U_flat_(0).scalar();
      double vy_cmd = last_optimal_U_flat_(1).scalar();
      double wz_cmd = last_optimal_U_flat_(2).scalar();

      // Update last commands for rate limiting in next step's cost function
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
    RCLCPP_INFO(logger_, "Setting up CasADi MPC problem structure...");
    // Define Problem Dimensions
    n_states_ = 3; // x, y, theta
    n_controls_ = 3; // vx, vy, wz (Omni)
    int N = settings_.time_steps; // Prediction horizon
    double T = settings_.model_dt; // Time step

    // Symbolic Variables
    MX X_sym = MX::sym("X", n_states_, N + 1); // States over N+1 time steps
    MX U_sym = MX::sym("U", n_controls_, N);   // Controls over N time steps

    // Parameters (initial state, reference trajectory, last control, obstacle params)
    // <<< Store sizes in MEMBER variables >>>
    n_init_state_params_ = n_states_;
    n_ref_params_ = n_states_ * (N + 1);
    n_last_control_params_ = n_controls_;
    n_obstacle_params_ = max_obstacles_ * n_obstacle_params_per_obs_; // cx, cy, r_sq

    n_params_ = n_init_state_params_ + n_ref_params_ + n_last_control_params_ + n_obstacle_params_;
    // <<< END Store sizes >>>
    P_sym_ = MX::sym("P", n_params_);
    RCLCPP_INFO(logger_, "MPC Params: N=%d, n_states=%d, n_controls=%d, n_params=%d (init=%d, ref=%d, last_ctrl=%d, obs=%d)",
        N, n_states_, n_controls_, n_params_, n_init_state_params_, n_ref_params_, n_last_control_params_, n_obstacle_params_); // Use member vars here too


    // Decision Variables Vector (flattened states and controls)
    V_sym_ = vertcat(reshape(X_sym, n_states_ * (N + 1), 1),
                     reshape(U_sym, n_controls_ * N, 1));
    n_opt_vars_ = V_sym_.rows();
    RCLCPP_INFO(logger_, "MPC: n_opt_vars=%d", n_opt_vars_);

    // Objective Function (Cost Function J)
    MX J = 0;
    int param_idx = 0;
    MX x_init = P_sym_(Slice(param_idx, param_idx + n_init_state_params_)); param_idx += n_init_state_params_;

    // Reference trajectory params start after initial state
    int ref_traj_start_idx = param_idx;

    // Last control params start after reference trajectory
    int last_ctrl_start_idx = ref_traj_start_idx + n_ref_params_;
    MX U_last = P_sym_(Slice(last_ctrl_start_idx, last_ctrl_start_idx + n_last_control_params_));

    // Obstacle params start after last control
    int obs_params_start_idx = last_ctrl_start_idx + n_last_control_params_;

    // ... (rest of cost function calculation remains the same) ...
    // ... Using LOCAL ref_traj_start_idx and last_ctrl_start_idx is fine here ...
     for (int k = 0; k < N; ++k) {
        MX X_k = X_sym(Slice(), k);
        MX U_k = U_sym(Slice(), k);
        MX X_ref_k = P_sym_(Slice(ref_traj_start_idx + k * n_states_,
                                ref_traj_start_idx + (k + 1) * n_states_));
        // ... cost terms ...
        MX U_prev = (k == 0) ? U_last : U_sym(Slice(), k - 1);
        // ... cost terms ...
    }
     // Terminal Cost
    MX X_N = X_sym(Slice(), N);
    MX X_ref_N = P_sym_(Slice(ref_traj_start_idx + N * n_states_,
                              ref_traj_start_idx + (N + 1) * n_states_));
    // ... terminal cost terms ...


    // Constraints Vector (g)
    MXVector g_vec;

    // Initial State Constraint (g0: X(0) - P(0:n_states-1) = 0)
    g_vec.push_back(X_sym(Slice(), 0) - x_init);

    for (int k = 0; k < N; ++k) {
        MX X_k = X_sym(Slice(), k);
        MX U_k = U_sym(Slice(), k);
        MX X_k_plus_1 = X_sym(Slice(), k + 1); // <<< DEFINE X_k_plus_1 HERE

        MX theta_k = X_k(2);
        MX vx_k = U_k(0);
        MX vy_k = U_k(1);
        MX wz_k = U_k(2);

        // State prediction using Omni model
        // <<< USE T instead of settings_.model_dt >>>
        MX x_next = X_k(0) + (vx_k * cos(theta_k) - vy_k * sin(theta_k)) * T;
        MX y_next = X_k(1) + (vx_k * sin(theta_k) + vy_k * cos(theta_k)) * T;
        MX theta_next = X_k(2) + wz_k * T;

        // <<< DEFINE X_k_plus_1_predicted HERE >>>
        MX X_k_plus_1_predicted = vertcat(x_next, y_next, theta_next);

        // Add dynamics defect constraint (X_k+1 - predicted = 0)
        // <<< Now uses the defined variables >>>
        g_vec.push_back(X_k_plus_1 - X_k_plus_1_predicted);
    }
    int dynamics_constraints_end_idx = g_vec.size();


    // Obstacle Constraints (gN+1 onwards)
    // <<< Use obs_params_start_idx calculated above >>>
    for (int j = 0; j < max_obstacles_; ++j) {
        // Get symbolic parameters for this obstacle slot
        int current_obs_param_idx = obs_params_start_idx + j * n_obstacle_params_per_obs_;
        MX cx_j = P_sym_(current_obs_param_idx + 0);
        MX cy_j = P_sym_(current_obs_param_idx + 1);
        MX r_sq_j = P_sym_(current_obs_param_idx + 2); // Using radius squared directly

        for (int k = 1; k <= N; ++k) { // Start from k=1 (predicted states)
            MX X_k = X_sym(Slice(), k);
            MX dx = X_k(0) - cx_j;
            MX dy = X_k(1) - cy_j;
            MX dist_sq = pow(dx, 2) + pow(dy, 2);
            g_vec.push_back(dist_sq);
        }
    }

    // Concatenate constraints
    MX g_flat = vertcat(g_vec);
    n_constraints_ = g_flat.rows();
    RCLCPP_INFO(logger_, "MPC: n_constraints=%d (init=%d, dyn=%d, obs=%d)",
                n_constraints_, n_states_, N * n_states_, max_obstacles_ * N);

    // NLP Definition
    MXDict nlp = {{"x", V_sym_}, {"f", J}, {"g", g_flat}, {"p", P_sym_}};

    // Create Solver
    Dict solver_opts;
    // solver_opts["ipopt.print_level"] = 3; // Verbose output for debugging
    solver_opts["ipopt.print_level"] = 0; // Suppress IPOPT output
    solver_opts["ipopt.sb"] = "yes";      // Suppress IPOPT banner
    solver_opts["print_time"] = 0;
    solver_opts["ipopt.warm_start_init_point"] = "yes"; // Enable warm starts
    solver_opts["ipopt.tol"] = 1e-3; // Adjust tolerance if needed
    solver_opts["ipopt.acceptable_tol"] = 1e-2; // Adjust acceptable tolerance
    solver_opts["ipopt.max_iter"] = 100; // Adjust max iterations

    solver_func_ = nlpsol("solver", "ipopt", nlp, solver_opts);

    // --- Define and Store Fixed Parts of Bounds ---
    // Vectors to hold the bounds that don't change between iterations
    x_flat_lower_bounds_.assign(n_opt_vars_, -inf); // Initialize with -inf
    x_flat_upper_bounds_.assign(n_opt_vars_, inf);  // Initialize with inf
    g_flat_lower_bounds_.assign(n_constraints_, 0.0); // Initialize with 0
    g_flat_upper_bounds_.assign(n_constraints_, 0.0); // Initialize with 0

    // State bounds (X)
    // Refine angle bounds
    for (int k = 0; k <= N; ++k) {
         x_flat_lower_bounds_[k * n_states_ + 2] = -10*M_PI; // Looser bounds help solver
         x_flat_upper_bounds_[k * n_states_ + 2] = 10*M_PI;
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
    // Initial state (g[0] to g[n_states-1]): lb=ub=0 (fixed later)
    // Dynamics (g[n_states] to g[dynamics_constraints_end_idx-1]): lb=ub=0 (already set)

    // Obstacle constraint bounds (will be set dynamically in solveMPC)
    // Initialize them to be non-binding (-inf, inf)
    int obs_constraint_start_idx = dynamics_constraints_end_idx;
    for (int i = obs_constraint_start_idx; i < n_constraints_; ++i) {
        g_flat_lower_bounds_[i] = -inf;
        g_flat_upper_bounds_[i] = inf;
    }

    mpc_problem_defined_ = true;
    RCLCPP_INFO(logger_, "CasADi problem structure setup complete.");
}

bool Optimizer::solveMPC(const std::vector<geometry_msgs::msg::Point> & obstacle_points)
{
    if (!mpc_problem_defined_) {
        RCLCPP_ERROR(logger_, "solveMPC called before setupCasADiProblem!");
        return false;
    }

    int N = settings_.time_steps;

    // 1. Process Obstacles
    std::vector<ClusterInfo> obstacles = processObstacles(obstacle_points);
    size_t num_detected_obstacles = obstacles.size();
    RCLCPP_DEBUG(logger_, "Detected %zu obstacle clusters.", num_detected_obstacles);

    // 2. Prepare Parameters Vector (P)
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

    // Last Control Command (vx_last, vy_last, wz_last)
    p_vec.push_back(last_v_.scalar());
    p_vec.push_back(last_vy_.scalar());
    p_vec.push_back(last_w_.scalar());

    // Obstacle Parameters (cx, cy, r_sq for j=0 to max_obstacles-1)
    double dummy_cx = 1e6; // Put dummy obstacles far away
    double dummy_cy = 1e6;
    double dummy_r_sq = 0.0;
    for (int j = 0; j < max_obstacles_; ++j) {
        if (j < static_cast<int>(num_detected_obstacles)) {
            p_vec.push_back(obstacles[j].cx);
            p_vec.push_back(obstacles[j].cy);
            p_vec.push_back(obstacles[j].radius_sq);
            RCLCPP_DEBUG(logger_, "  Obstacle %d: cx=%.2f, cy=%.2f, r_sq=%.2f", j, obstacles[j].cx, obstacles[j].cy, obstacles[j].radius_sq);
        } else {
            p_vec.push_back(dummy_cx);
            p_vec.push_back(dummy_cy);
            p_vec.push_back(dummy_r_sq);
        }
    }

    if (p_vec.size() != static_cast<size_t>(n_params_)) {
         RCLCPP_ERROR(logger_, "Parameter vector size mismatch! Expected %d, got %zu", n_params_, p_vec.size());
        return false;
    }
    DM p = DM(p_vec);

    // 3. Prepare Initial Guess (x0_guess) for V (states and controls)
    DM x0_guess;
    if (last_solve_successful_ && !last_optimal_X_flat_.is_empty() && !last_optimal_U_flat_.is_empty()) {
        // Warm start: Use the shifted previous optimal solution
        RCLCPP_DEBUG(logger_, "Using warm start initial guess.");
        DM X_prev = reshape(last_optimal_X_flat_, n_states_, N + 1);
        DM U_prev = reshape(last_optimal_U_flat_, n_controls_, N);

        // Shift state trajectory left, repeat last state
        DM X_guess = horzcat(X_prev(Slice(), Slice(1, N + 1)), X_prev(Slice(), N));
        // Shift control sequence left, repeat last control
        DM U_guess = horzcat(U_prev(Slice(), Slice(1, N)), U_prev(Slice(), N-1));

        x0_guess = vertcat(reshape(X_guess, n_states_ * (N + 1), 1),
                           reshape(U_guess, n_controls_ * N, 1));
    } else {
        // Cold start: Repeat initial state, zero controls
        RCLCPP_DEBUG(logger_, "Using cold start initial guess.");
        std::vector<double> x0_vec(n_opt_vars_, 0.0);
        // Fill states part with current state
        for (int k = 0; k <= N; ++k) {
            x0_vec[k * n_states_ + 0] = current_x;
            x0_vec[k * n_states_ + 1] = current_y;
            x0_vec[k * n_states_ + 2] = current_theta;
        }
        // Controls part remains zero initialized
        x0_guess = DM(x0_vec);
    }

    // 4. Prepare Bounds (lbx, ubx, lbg, ubg)
    // Start with the fixed bounds calculated in setup
    DM lbx = DM(x_flat_lower_bounds_);
    DM ubx = DM(x_flat_upper_bounds_);
    DM lbg = DM(g_flat_lower_bounds_);
    DM ubg = DM(g_flat_upper_bounds_);

    // Update bounds for the initial state constraint X(0) == P(0:2)
    // Variables X(0), X(1), X(2) are fixed to initial state
    for (int i = 0; i < n_states_; ++i) {
        lbx(i) = p_vec[i];
        ubx(i) = p_vec[i];
        // Corresponding initial state constraints g(0) to g(n_states-1) have bounds [0, 0]
        lbg(i) = 0.0;
        ubg(i) = 0.0;
    }
    // Bounds for dynamics constraints g(n_states) to g(dynamics_end-1) remain [0, 0]

    // Update bounds for obstacle constraints
    int dynamics_constraints_end_idx = n_states_ + N * n_states_;
    int obs_constraint_start_idx = dynamics_constraints_end_idx;
    int current_g_idx = obs_constraint_start_idx;

    for (int j = 0; j < max_obstacles_; ++j) {
        // Get the radius^2 for obstacle slot j from the parameter vector P
        int p_idx_r_sq = n_init_state_params_ + n_ref_params_ + n_last_control_params_ + j * n_obstacle_params_per_obs_ + 2;
        double r_sq_j = p_vec[p_idx_r_sq];

        bool is_real_obstacle = (j < static_cast<int>(num_detected_obstacles) && r_sq_j >= 0); // Check if it's not a dummy

        double lower_bound = -inf; // Default for dummy obstacles
        if (is_real_obstacle) {
             // Constraint is dist_sq >= r_sq + 2*r*pad + pad^2
             // pad^2 = effective_radius_padding_sq_
             // r = sqrt(r_sq)
             // pad = robot_radius_ + lidar_safety_margin_
             // We need lower bound for dist_sq which is g[idx]
             // lower_bound = pow(sqrt(r_sq_j) + robot_radius_ + lidar_safety_margin_, 2);
             // Avoid sqrt if possible: lower_bound = r_sq_j + 2*sqrt(r_sq_j)*(robot_radius_+margin) + (robot_radius_+margin)^2
             // Let's use the simpler (R_obs + R_padding)^2 where R_padding = robot_radius + margin
             // lower_bound = pow(sqrt(r_sq_j) + (robot_radius_ + lidar_safety_margin_), 2);
             // simpler: use effective_radius_padding_sq_ = (robot_radius + margin)^2
             // bound should be r_sq + 2*sqrt(r_sq)*sqrt(eff_pad_sq) + eff_pad_sq ? No.
             // Constraint g = dist_sq. We need g >= (R_cluster + R_padding)^2
             // R_padding = robot_radius + margin
             double R_padding = robot_radius_ + lidar_safety_margin_;
             lower_bound = pow(sqrt(r_sq_j) + R_padding, 2);

             // Alternative idea: If R_cluster is 0 (single point), bound is R_padding^2
             // if (r_sq_j < 1e-6) { // Treat as point obstacle
             //     lower_bound = effective_radius_padding_sq_;
             // } else {
             //     lower_bound = pow(sqrt(r_sq_j) + R_padding, 2);
             // }
        }

        for (int k = 1; k <= N; ++k) { // For each timestep k
            if (current_g_idx >= n_constraints_) {
                RCLCPP_ERROR(logger_, "Constraint index %d out of bounds %d", current_g_idx, n_constraints_);
                return false;
            }
            lbg(current_g_idx) = lower_bound; // Apply lower bound (or -inf for dummy)
            ubg(current_g_idx) = inf;         // Upper bound is always infinity
            current_g_idx++;
        }
    }

    // 5. Call Solver
    DMDict arg = {{"p", p}, {"x0", x0_guess}, {"lbx", lbx}, {"ubx", ubx}, {"lbg", lbg}, {"ubg", ubg}};
    DMDict res;
    auto solve_start_time = std::chrono::high_resolution_clock::now();
    try {
       res = solver_func_(arg);
    } catch (const std::exception &e) {
        RCLCPP_ERROR(logger_, "CasADi solver threw exception: %s", e.what());
        last_solve_successful_ = false;
        return false;
    }
    auto solve_end_time = std::chrono::high_resolution_clock::now();
    auto solve_duration = std::chrono::duration_cast<std::chrono::microseconds>(solve_end_time - solve_start_time);
    RCLCPP_DEBUG(logger_, "CasADi solve time: %.3f ms", solve_duration.count() / 1000.0);

    // 6. Process Results
    bool success = solver_func_.stats().at("success");
    if (success) {
        DM sol = res.at("x");
        // Store solution for warm starting and command extraction
        last_optimal_X_flat_ = sol(Slice(0, n_states_ * (N + 1)));
        last_optimal_U_flat_ = sol(Slice(n_states_ * (N + 1), n_opt_vars_));
        last_solve_successful_ = true;
        RCLCPP_DEBUG(logger_, "Solve successful. Final cost: %f", static_cast<double>(res.at("f")));
        return true;
    } else {
        RCLCPP_WARN(logger_, "MPC solver did not find an optimal solution. Status: %s",
                   static_cast<std::string>(solver_func_.stats().at("return_status")).c_str());
        last_solve_successful_ = false;
        // Clear previous solution to force cold start next time
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

// --- Obstacle Processing Implementation ---

// Basic Euclidean Clustering (can be improved with KD-Trees for performance)
std::vector<std::vector<size_t>> Optimizer::clusterPoints(
    const std::vector<geometry_msgs::msg::Point>& points)
{
    std::vector<std::vector<size_t>> clusters;
    if (points.empty()) {
        return clusters;
    }

    size_t n_points = points.size();
    std::vector<bool> visited(n_points, false);
    double tolerance_sq = lidar_cluster_tolerance_ * lidar_cluster_tolerance_;

    for (size_t i = 0; i < n_points; ++i) {
        if (!visited[i]) {
            std::vector<size_t> current_cluster_indices;
            std::vector<size_t> q; // Queue for BFS-like search
            q.push_back(i);
            visited[i] = true;

            size_t head = 0;
            while(head < q.size()) {
                size_t current_idx = q[head++];
                current_cluster_indices.push_back(current_idx);
                const auto& p1 = points[current_idx];

                // Find neighbors
                for (size_t j = 0; j < n_points; ++j) {
                    if (!visited[j]) {
                        const auto& p2 = points[j];
                        double dx = p1.x - p2.x;
                        double dy = p1.y - p2.y;
                        if ((dx * dx + dy * dy) < tolerance_sq) {
                            visited[j] = true;
                            q.push_back(j);
                        }
                    }
                }
            }
            // Store the cluster if it meets the minimum size
            if (current_cluster_indices.size() >= static_cast<size_t>(lidar_min_cluster_size_)) {
                clusters.push_back(current_cluster_indices);
            }
        }
    }
    return clusters;
}

ClusterInfo Optimizer::computeBoundingCircle(
    const std::vector<geometry_msgs::msg::Point>& cluster_points)
{
    ClusterInfo info;
    if (cluster_points.empty()) {
        return info;
    }

    info.num_points = cluster_points.size();

    // Calculate centroid
    double sum_x = 0.0, sum_y = 0.0;
    for (const auto& pt : cluster_points) {
        sum_x += pt.x;
        sum_y += pt.y;
    }
    info.cx = sum_x / info.num_points;
    info.cy = sum_y / info.num_points;

    // Find max squared distance from centroid to any point
    double max_dist_sq = 0.0;
    for (const auto& pt : cluster_points) {
        double dx = pt.x - info.cx;
        double dy = pt.y - info.cy;
        max_dist_sq = std::max(max_dist_sq, dx * dx + dy * dy);
    }
    info.radius_sq = max_dist_sq;

    return info;
}

std::vector<ClusterInfo> Optimizer::processObstacles(
    const std::vector<geometry_msgs::msg::Point>& points)
{
    std::vector<ClusterInfo> obstacle_info_list;

    if (points.empty()) {
        return obstacle_info_list;
    }

    // 1. Cluster points
    std::vector<std::vector<size_t>> cluster_indices = clusterPoints(points);

    // 2. Compute bounding circle for each valid cluster
    obstacle_info_list.reserve(cluster_indices.size());
    for (const auto& indices : cluster_indices) {
        // Create a temporary vector of points for the current cluster
        std::vector<geometry_msgs::msg::Point> current_cluster_points;
        current_cluster_points.reserve(indices.size());
        for (size_t idx : indices) {
            current_cluster_points.push_back(points[idx]);
        }

        // Compute bounding circle
        ClusterInfo info = computeBoundingCircle(current_cluster_points);
        if (info.num_points > 0) { // Should always be true if cluster was valid
            obstacle_info_list.push_back(info);
        }
    }

    // Optional: Sort obstacles (e.g., by distance to robot) if needed,
    //           although the parameter vector uses fixed slots.

    // Limit to max_obstacles_
    if (obstacle_info_list.size() > static_cast<size_t>(max_obstacles_)) {
        // Optional: Add logic here to select the 'closest' or 'most important' obstacles
        // For now, just take the first ones found.
        obstacle_info_list.resize(max_obstacles_);
    }

    return obstacle_info_list;
}


}  // namespace sortham