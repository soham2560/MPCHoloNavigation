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

// C++ Standard Library
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// External Libraries
#include <casadi/casadi.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <xtensor/xmath.hpp>

// ROS / Nav2 Headers
#include "nav2_costmap_2d/costmap_filters/filter_values.hpp"

// Project Headers
#include "nav2_sortham_controller/optimizer.hpp"

namespace sortham
{

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
  if (!node) {
    throw std::runtime_error("Optimizer::initialize: Unable to lock node.");
  }
  logger_ = node->get_logger();
  clock_ = node->get_clock();

  getParams();

  current_milestone_idx_ = 0;
  last_path_metadata_ = "";
  current_milestones_goal_ = std::nullopt;

  reset();

  mpc_problem_defined_ = false;
  last_solve_successful_ = false;

  last_v_ = DM(0);
  last_w_ = DM(0);
  last_vy_ = DM(0);

  RCLCPP_INFO(logger_, "MPC Optimizer Initialized for %s", name_.c_str());
}

void Optimizer::getParams()
{
  std::string motion_model_name;
  auto & s = settings_;
  auto getParam = parameters_handler_->getParamGetter(name_);

  // --- Basic MPC Settings ---
  getParam(s.model_dt, "model_dt", 0.05f);
  getParam(s.time_steps, "time_steps", 56);

  // --- Motion Model ---
  getParam(motion_model_name, "motion_model", std::string("Omni"));

  // --- Base Constraints ---
  getParam(s.base_constraints.vx_max, "vx_max", 0.5);
  getParam(s.base_constraints.vx_min, "vx_min", -0.35);
  getParam(s.base_constraints.vy, "vy_max", 0.5);
  getParam(s.base_constraints.wz, "wz_max", 1.9);

  // --- MPC Weights ---
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

  // --- Obstacle Parameters ---
  getParam(robot_radius_, "robot_radius", 0.25);
  getParam(lidar_safety_margin_, "lidar_safety_margin", 0.05);
  getParam(lidar_cluster_tolerance_, "lidar_cluster_tolerance", 0.05);
  getParam(lidar_min_cluster_size_, "lidar_min_cluster_size", 3);
  getParam(max_obstacles_, "max_obstacles", 10);
  getParam(max_cluster_radius_, "max_cluster_radius", 1.0);
  RCLCPP_INFO(
    logger_,
    "Lidar constraints: radius=%.2f, margin=%.2f, cluster_tol=%.2f, min_cluster=%d, max_obs=%d, max_cluster_radius=%.2f",
    robot_radius_, lidar_safety_margin_, lidar_cluster_tolerance_, lidar_min_cluster_size_,
    max_obstacles_, max_cluster_radius_);

  // --- Milestone Parameters ---
  getParam(milestone_reached_threshold_, "milestone_reached_threshold", 0.05);
  getParam(milestone_min_spacing_, "milestone_min_spacing", 0.3);
  getParam(milestone_max_spacing_, "milestone_max_spacing", 3.0);
  getParam(milestone_curvature_factor_, "milestone_curvature_factor", 5.0);
  RCLCPP_INFO(
    logger_,
    "Milestone Params: reach_thresh=%.2f, min_space=%.2f, max_space=%.2f, curv_factor=%.2f",
    milestone_reached_threshold_, milestone_min_spacing_, milestone_max_spacing_,
    milestone_curvature_factor_);


  // --- Setup based on parameters ---
  s.constraints = s.base_constraints;
  setMotionModel(motion_model_name);
  if (!isHolonomic()) {
    RCLCPP_ERROR(
      logger_,
      "MPC currently setup assuming Omni model. Non-holonomic model '%s' requires code changes.",
      motion_model_name.c_str());
    throw std::runtime_error("Non-holonomic model requires MPC code adaptation.");
  }

  parameters_handler_->addPostCallback(
    [this]() {
      RCLCPP_INFO(logger_, "Parameters reloaded, resetting optimizer and MPC problem.");
      reset();
      mpc_problem_defined_ = false;
      last_solve_successful_ = false;
      last_v_ = DM(0);
      last_vy_ = DM(0);
      last_w_ = DM(0);
    });
}


void Optimizer::reset()
{
  last_optimal_X_flat_ = casadi::DM();
  last_optimal_U_flat_ = casadi::DM();
  last_v_ = DM(0);
  last_w_ = DM(0);
  last_vy_ = DM(0);
  last_solve_successful_ = false;
  current_milestones_.clear();
  current_milestone_idx_ = 0;
  last_path_metadata_ = "";
  current_milestones_goal_ = std::nullopt;

  last_processed_obstacles_.clear();

  RCLCPP_INFO(logger_, "Optimizer MPC state reset");
}

geometry_msgs::msg::TwistStamped Optimizer::evalControl(
  const geometry_msgs::msg::PoseStamped & robot_pose,
  const geometry_msgs::msg::Twist & robot_speed,
  const nav_msgs::msg::Path & plan,
  const geometry_msgs::msg::Pose & goal,
  const std::vector<geometry_msgs::msg::Point> & obstacle_points,
  nav2_core::GoalChecker * /*goal_checker*/)
{
  // --- 1. Store current state ---
  current_state_.pose = robot_pose;
  current_state_.speed = robot_speed;

  // --- 2. Check if Path is New or Cleared ---
  bool need_new_milestones = false;

  if (!current_milestones_goal_.has_value()) {
      // Case 1: No milestones generated yet for any goal
      if (!plan.poses.empty()) {
          RCLCPP_INFO(logger_, "First valid plan received. Generating initial milestones.");
          need_new_milestones = true;
      } else {
          RCLCPP_WARN_ONCE(logger_, "Waiting for initial valid plan to generate milestones...");
          // Cannot proceed without milestones, return zero command or handle appropriately
          return utils::toTwistStamped(0.0, 0.0, 0.0, plan.header.stamp, costmap_ros_->getBaseFrameID());
      }
  } else {
      // Case 2: Milestones exist. Check if the *goal* changed significantly.
      constexpr double kGoalPositionToleranceSq = 0.1 * 0.1; // e.g., 10cm position tolerance squared
      constexpr double kGoalYawTolerance = 0.1; // e.g., ~6 degrees yaw tolerance

      double dx = goal.position.x - current_milestones_goal_.value().position.x;
      double dy = goal.position.y - current_milestones_goal_.value().position.y;
      double dist_sq = dx * dx + dy * dy;

      double current_goal_yaw = tf2::getYaw(current_milestones_goal_.value().orientation);
      double new_goal_yaw = tf2::getYaw(goal.orientation);
      double dyaw = new_goal_yaw - current_goal_yaw;
      dyaw = std::abs(atan2(sin(dyaw), cos(dyaw))); // Absolute angular difference

      if (dist_sq > kGoalPositionToleranceSq || dyaw > kGoalYawTolerance) {
          RCLCPP_INFO(logger_, "New goal detected (Dist sq: %.3f > %.3f or Yaw diff: %.3f > %.3f). Regenerating milestones.",
                      dist_sq, kGoalPositionToleranceSq, dyaw, kGoalYawTolerance);
          if (!plan.poses.empty()) {
              need_new_milestones = true;
          } else {
              RCLCPP_WARN(logger_, "New goal detected, but received empty plan. Clearing milestones and waiting.");
              current_milestones_.clear();
              current_milestone_idx_ = 0;
              current_milestones_goal_ = std::nullopt; // Clear stored goal
              // Return zero command?
              return utils::toTwistStamped(0.0, 0.0, 0.0, plan.header.stamp, costmap_ros_->getBaseFrameID());
          }
      } else {
          RCLCPP_DEBUG(logger_, "Goal unchanged. Continuing with existing milestones.");
      }
  }

  // --- Generate Milestones if Needed ---
  if (need_new_milestones) {
      generateMilestones(plan); // Use the provided plan (assumed to be for the new goal)
      current_milestones_goal_ = goal; // Store the goal these milestones are for
      // Reset index? Yes, start from the beginning for the new goal.
      current_milestone_idx_ = 0;
  }

  // Check if milestones are valid after potential generation/clearing
  if (current_milestones_.empty()) {
      RCLCPP_ERROR(logger_, "No milestones available to select target. Cannot proceed.");
      return utils::toTwistStamped(0.0, 0.0, 0.0, plan.header.stamp, costmap_ros_->getBaseFrameID());
  }

  // --- 3. Ensure MPC Problem is Defined ---
  if (!mpc_problem_defined_) {
    RCLCPP_INFO(logger_, "Setting up CasADi NLP problem for the first time or after reset...");
    try {
      setupCasADiProblem();
      if (!mpc_problem_defined_) {
        RCLCPP_ERROR(logger_, "Failed to setup CasADi problem on demand within evalControl.");
        return utils::toTwistStamped(
          0.0, 0.0, 0.0, plan.header.stamp,
          costmap_ros_->getBaseFrameID());
      }
      RCLCPP_INFO(logger_, "CasADi NLP problem setup complete.");
    } catch (const std::exception & e) {
      RCLCPP_ERROR(
        logger_, "Exception during on-demand setupCasADiProblem in evalControl: %s",
        e.what());
      return utils::toTwistStamped(
        0.0, 0.0, 0.0, plan.header.stamp,
        costmap_ros_->getBaseFrameID());
    }
  }

  // --- 4. Solve MPC using Milestone Target ---
  RCLCPP_DEBUG(
    logger_, "Attempting to solve MPC problem targeting milestone %zu...",
    current_milestone_idx_);
  bool success = solveMPC(obstacle_points);

  // --- 5. Process Results and Generate Command ---
  geometry_msgs::msg::TwistStamped cmd_vel;
  cmd_vel.header.stamp = clock_->now();
  cmd_vel.header.frame_id = costmap_ros_->getBaseFrameID();

  if (success && !last_optimal_U_flat_.is_empty()) {
    RCLCPP_DEBUG(logger_, "MPC solve successful.");
    double vx_cmd = last_optimal_U_flat_(0).scalar();
    double vy_cmd = last_optimal_U_flat_(1).scalar();
    double wz_cmd = last_optimal_U_flat_(2).scalar();

    last_v_ = vx_cmd;
    last_vy_ = vy_cmd;
    last_w_ = wz_cmd;

    cmd_vel.twist.linear.x = vx_cmd;
    cmd_vel.twist.linear.y = vy_cmd;
    cmd_vel.twist.angular.z = wz_cmd;

    RCLCPP_DEBUG(logger_, "MPC Command: [vx=%.3f, vy=%.3f, wz=%.3f]", vx_cmd, vy_cmd, wz_cmd);

  } else {
    RCLCPP_WARN(
      logger_,
      "MPC optimization failed or returned empty solution. Applying zero velocity command.");
    last_v_ = DM(0);
    last_vy_ = DM(0);
    last_w_ = DM(0);
    last_solve_successful_ = false;

    cmd_vel.twist.linear.x = 0.0;
    cmd_vel.twist.linear.y = 0.0;
    cmd_vel.twist.angular.z = 0.0;
  }

  return cmd_vel;
}

bool Optimizer::isHolonomic() const
{
  if (!motion_model_) {
    RCLCPP_ERROR(logger_, "Motion model not initialized in isHolonomic check!");
    return true;     // Or throw
  }
  return motion_model_->isHolonomic();
}

void Optimizer::setMotionModel(const std::string & model)
{
  if (model == "DiffDrive") {
    motion_model_ = std::make_shared<DiffDriveMotionModel>();
  } else if (model == "Omni") {
    motion_model_ = std::make_shared<OmniMotionModel>();
  } else if (model == "Ackermann") {
    if (!parameters_handler_) {
      throw std::runtime_error("Parameter handler not available for Ackermann model initialization!");
    }
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
  auto & s = settings_;
  const double epsilon = 1e-5;

  RCLCPP_DEBUG(
    logger_, "setSpeedLimit called with limit: %.2f, percentage: %d", speed_limit,
    percentage);

  auto new_limits = s.base_constraints; // Start with base limits

  if (speed_limit >= epsilon) { // Apply limit if positive
    double ratio = 1.0;
    if (percentage) {
      ratio = speed_limit / 100.0;
      RCLCPP_INFO(
        logger_, "Setting speed limit to %.2f%% of base limits (ratio %.2f).",
        speed_limit, ratio);
    } else {
      if (s.base_constraints.vx_max > epsilon) {
        ratio = speed_limit / s.base_constraints.vx_max;
        ratio = std::min(1.0, ratio);  // Clamp ratio
        RCLCPP_INFO(
          logger_, "Setting speed limit to absolute %.2f m/s (ratio %.2f).", speed_limit,
          ratio);
      } else {
        RCLCPP_WARN(logger_, "Cannot set absolute speed limit: base vx_max is zero.");
        return;
      }
    }
    new_limits.vx_max = s.base_constraints.vx_max * ratio;
    new_limits.vx_min =
      (s.base_constraints.vx_min < 0) ? (s.base_constraints.vx_min * ratio) : std::min(
      0.0f,
      new_limits.vx_max);
    new_limits.vy = s.base_constraints.vy * ratio;
    new_limits.wz = s.base_constraints.wz * ratio;
  } else {
    RCLCPP_INFO(logger_, "Speed limit reset to base constraints.");
    // new_limits is already base_constraints
  }

  // Check if limits actually changed before forcing recreation
  if (std::fabs(s.constraints.vx_max - new_limits.vx_max) > epsilon ||
    std::fabs(s.constraints.vx_min - new_limits.vx_min) > epsilon ||
    std::fabs(s.constraints.vy - new_limits.vy) > epsilon ||
    std::fabs(s.constraints.wz - new_limits.wz) > epsilon)
  {
    s.constraints = new_limits;
    RCLCPP_INFO(
      logger_, "New velocity limits: vx[%.2f, %.2f], vy +/-%.2f, wz +/-%.2f",
      s.constraints.vx_min, s.constraints.vx_max, s.constraints.vy, s.constraints.wz);
    mpc_problem_defined_ = false;   // Force recreation due to bound change
  } else {
    RCLCPP_DEBUG(logger_, "Calculated speed limits are same as current ones.");
  }
}

xt::xtensor<float, 2> Optimizer::getOptimizedTrajectory()
{
  if (!last_solve_successful_ || last_optimal_X_flat_.is_empty()) {
    RCLCPP_DEBUG(logger_, "getOptimizedTrajectory: No successful solution available.");
    return xt::xtensor<float, 2>::from_shape({0, 3});
  }

  size_t N = settings_.time_steps;
  std::vector<std::size_t> shape = {static_cast<std::size_t>(N + 1), 3ul};
  xt::xtensor<float, 2> trajectory = xt::zeros<float>(shape);

  if (n_states_ != 3) {
    RCLCPP_ERROR(logger_, "getOptimizedTrajectory assumes n_states_ = 3, but it is %d", n_states_);
    return xt::xtensor<float, 2>::from_shape({0, 3});
  }

  std::vector<double> x_vec = last_optimal_X_flat_.get_elements();

  for (size_t i = 0; i <= N; ++i) {
    trajectory(i, 0) = static_cast<float>(x_vec[i * n_states_ + 0]);
    trajectory(i, 1) = static_cast<float>(x_vec[i * n_states_ + 1]);
    trajectory(i, 2) = static_cast<float>(x_vec[i * n_states_ + 2]);
  }
  RCLCPP_DEBUG(
    logger_, "getOptimizedTrajectory: Returning trajectory with %zu points.",
    trajectory.shape()[0]);
  return trajectory;
}

void Optimizer::setupCasADiProblem()
{
  RCLCPP_INFO(logger_, "Setting up CasADi NLP symbolic structure (Nonlinear)...");
  const double inf = std::numeric_limits<double>::infinity();
  const double kEpsilonSqrt = 1e-9;
  const double kAngleBoundMultiplier = 10.0;

  // ==============================================================
  // 1. Dimensions & Symbolic Variables (Decision Vars V = [X; U])
  // ==============================================================
  n_states_ = 3;
  n_controls_ = 3;
  int N = settings_.time_steps;

  MX X_sym = MX::sym("X", n_states_, N + 1);
  MX U_sym = MX::sym("U", n_controls_, N);
  V_sym_ = vertcat(
    reshape(X_sym, n_states_ * (N + 1), 1),
    reshape(U_sym, n_controls_ * N, 1));
  n_opt_vars_ = V_sym_.rows();

  // ==============================================================
  // 2. Symbolic Parameters P
  // ==============================================================
  n_init_state_params_ = n_states_;
  n_last_control_params_ = n_controls_;
  n_local_target_params_ = n_states_;
  n_obstacle_params_per_obs_ = 3;
  n_obstacle_params_ = max_obstacles_ * n_obstacle_params_per_obs_;

  n_params_ = n_init_state_params_ + n_last_control_params_ + n_local_target_params_ +
    n_obstacle_params_;
  P_sym_ = MX::sym("P", n_params_);
  RCLCPP_INFO(
    logger_,
    "NLP Params: N=%d, n_states=%d, n_controls=%d, n_params=%d (init=%d, last_ctrl=%d, target=%d, obs=%d)",
    N, n_states_, n_controls_, n_params_, n_init_state_params_, n_last_control_params_, n_local_target_params_,
    n_obstacle_params_);

  int current_p_idx = 0;
  Slice init_state_slice(current_p_idx, current_p_idx + n_init_state_params_);
  current_p_idx += n_init_state_params_;
  Slice last_ctrl_slice(current_p_idx, current_p_idx + n_last_control_params_);
  current_p_idx += n_last_control_params_;
  Slice local_target_slice(current_p_idx, current_p_idx + n_local_target_params_);
  current_p_idx += n_local_target_params_;
  Slice obs_slice(current_p_idx, current_p_idx + n_obstacle_params_);

  MX P_init_state = P_sym_(init_state_slice);
  MX P_last_ctrl = P_sym_(last_ctrl_slice);
  MX P_local_target = P_sym_(local_target_slice);
  MX P_obstacles_flat = P_sym_(obs_slice);
  MX P_obstacles = reshape(P_obstacles_flat, n_obstacle_params_per_obs_, max_obstacles_).T();

  // ==============================================================
  // 3. Create Dynamics Function (Symbolic)
  // ==============================================================
  MX x_sym_dyn = MX::sym("x_dyn", n_states_, 1);
  MX u_sym_dyn = MX::sym("u_dyn", n_controls_, 1);
  MX theta_sym_dyn = x_sym_dyn(2);
  MX vx_sym_dyn = u_sym_dyn(0);
  MX vy_sym_dyn = u_sym_dyn(1);
  MX wz_sym_dyn = u_sym_dyn(2);

  MX f_dyn_expr = vertcat(
    x_sym_dyn(0) + (vx_sym_dyn * cos(theta_sym_dyn) - vy_sym_dyn * sin(theta_sym_dyn)) *
    settings_.model_dt,
    x_sym_dyn(1) + (vx_sym_dyn * sin(theta_sym_dyn) + vy_sym_dyn * cos(theta_sym_dyn)) *
    settings_.model_dt,
    x_sym_dyn(2) + wz_sym_dyn * settings_.model_dt
  );

  dyn_func_ = Function("f", {x_sym_dyn, u_sym_dyn}, {f_dyn_expr}, {"x_k", "u_k"}, {"x_k_plus_1"});
  RCLCPP_INFO(logger_, "Created nonlinear dynamics function.");

  // ==============================================================
  // 4. Define Symbolic Objective J
  // ==============================================================
  MX J = 0;
  DM Q = DM::diag(DM({settings_.weight_x, settings_.weight_y, settings_.weight_theta}));
  DM QN = DM::diag(
    DM({settings_.weight_terminal_x, settings_.weight_terminal_y, settings_.weight_terminal_theta})
  );
  DM R = DM::diag(DM({settings_.weight_v, settings_.weight_vy, settings_.weight_w}));
  DM Rd = DM::diag(
    DM({settings_.weight_v_accel, settings_.weight_vy_accel, settings_.weight_w_accel})
  );

  for (int k = 0; k < N; ++k) {
    MX x_k = X_sym(Slice(), k);
    MX u_k = U_sym(Slice(), k);
    MX state_error = x_k - P_local_target;
    state_error(2) = atan2(sin(state_error(2)), cos(state_error(2)));
    J += mtimes(state_error.T(), mtimes(Q, state_error));
    J += mtimes(u_k.T(), mtimes(R, u_k));
    MX u_prev = (k == 0) ? P_last_ctrl : U_sym(Slice(), k - 1);
    MX delta_u = u_k - u_prev;
    J += mtimes(delta_u.T(), mtimes(Rd, delta_u));
  }

  MX x_N = X_sym(Slice(), N);
  MX terminal_error = x_N - P_local_target;
  terminal_error(2) = atan2(sin(terminal_error(2)), cos(terminal_error(2)));
  J += mtimes(terminal_error.T(), mtimes(QN, terminal_error));

  // ==============================================================
  // 5. Define Symbolic Constraints Vector (g)
  // ==============================================================
  MXVector g_vec;

  // --- Initial State Constraint ---
  g_vec.push_back(X_sym(Slice(), 0) - P_init_state);

  // --- Dynamics Constraints ---
  if (!dyn_func_.get()) {
    RCLCPP_ERROR(logger_,
      "Nonlinear dynamics function (dyn_func_) not created before constraint setup!");
    throw std::runtime_error("Dynamics function invalid");
  }
  for (int k = 0; k < N; ++k) {
    MXVector current_inputs = {X_sym(Slice(), k), U_sym(Slice(), k)};
    MXVector predicted_output = dyn_func_(current_inputs);
    MX X_k_plus_1_predicted = predicted_output[0];
    MX dyn_defect_k = X_sym(Slice(), k + 1) - X_k_plus_1_predicted;
    g_vec.push_back(dyn_defect_k);
  }

  // --- Obstacle Constraints ---
  double R_padding = robot_radius_ + lidar_safety_margin_;

  for (int j = 0; j < max_obstacles_; ++j) {
    MX cx_j = P_obstacles(j, 0);
    MX cy_j = P_obstacles(j, 1);
    MX r_sq_j = P_obstacles(j, 2);

    MX R_obs_j_robust = sqrt(fmax(0.0, r_sq_j) + kEpsilonSqrt);
    MX R_safe_j_sq = pow(R_obs_j_robust + R_padding, 2);

    for (int k = 1; k <= N; ++k) {
      MX X_k = X_sym(Slice(), k);
      MX dx = X_k(0) - cx_j;
      MX dy = X_k(1) - cy_j;
      MX dist_sq = pow(dx, 2) + pow(dy, 2);
      g_vec.push_back(dist_sq - R_safe_j_sq);
    }
  }

  // --- Concatenate constraints ---
  MX g_sym = vertcat(g_vec);
  n_constraints_ = g_sym.rows();
  obs_constraint_start_idx_ = n_states_ + N * n_states_;
  RCLCPP_INFO(
    logger_, "NLP Structure: n_constraints=%d (init=%d, dyn=%d, obs=%d)",
    n_constraints_, n_states_, N * n_states_, max_obstacles_ * N);

  // ==============================================================
  // 6. NLP Definition & Solver Creation
  // ==============================================================
  MXDict nlp_dict = {{"x", V_sym_}, {"f", J}, {"g", g_sym}, {"p", P_sym_}};

  Dict solver_opts;
  std::string solver_name = "ipopt";
  RCLCPP_INFO(logger_, "Using NLP solver: %s", solver_name.c_str());

  solver_opts["ipopt.print_level"] = 0;
  solver_opts["ipopt.sb"] = "yes";
  solver_opts["print_time"] = 0;
  solver_opts["ipopt.warm_start_init_point"] = "yes";
  solver_opts["ipopt.tol"] = 1e-3;
  solver_opts["ipopt.acceptable_tol"] = 1e-2;
  solver_opts["ipopt.max_iter"] = 100;

  try {
    solver_func_ = nlpsol("solver", solver_name, nlp_dict, solver_opts);
    RCLCPP_INFO(logger_, "Successfully created IPOPT solver interface via nlpsol.");
  } catch (const std::exception & e) {
    RCLCPP_FATAL(logger_, "Failed to create CasADi NLP solver interface for '%s': %s",
      solver_name.c_str(), e.what());
    throw std::runtime_error("Failed to create CasADi solver.");
  }

  // ==============================================================
  // 7. Define Fixed Variable Bounds (lbx, ubx)
  // ==============================================================
  x_flat_lower_bounds_.assign(n_opt_vars_, -inf);
  x_flat_upper_bounds_.assign(n_opt_vars_, inf);

  for (int k = 0; k <= N; ++k) {
    int idx = k * n_states_ + 2;
    x_flat_lower_bounds_[idx] = -kAngleBoundMultiplier * M_PI;
    x_flat_upper_bounds_[idx] = kAngleBoundMultiplier * M_PI;
  }

  int u_offset = n_states_ * (N + 1);
  for (int k = 0; k < N; ++k) {
    int base_idx = u_offset + k * n_controls_;
    x_flat_lower_bounds_[base_idx + 0] = settings_.constraints.vx_min;
    x_flat_upper_bounds_[base_idx + 0] = settings_.constraints.vx_max;
    x_flat_lower_bounds_[base_idx + 1] = -settings_.constraints.vy;
    x_flat_upper_bounds_[base_idx + 1] = settings_.constraints.vy;
    x_flat_lower_bounds_[base_idx + 2] = -settings_.constraints.wz;
    x_flat_upper_bounds_[base_idx + 2] = settings_.constraints.wz;
  }

  // ==============================================================
  // 8. Define Fixed Constraint Bounds (lbg, ubg)
  // ==============================================================
  g_flat_lower_bounds_.assign(n_constraints_, 0.0);
  g_flat_upper_bounds_.assign(n_constraints_, 0.0);

  int obs_constraint_start_idx = n_states_ + N * n_states_;
  for (int i = obs_constraint_start_idx; i < n_constraints_; ++i) {
    g_flat_lower_bounds_[i] = 0.0;
    g_flat_upper_bounds_[i] = inf;
  }

  // ==============================================================
  // 9. Finalize Setup
  // ==============================================================
  mpc_problem_defined_ = true;
  RCLCPP_INFO(logger_, "CasADi NLP problem structure setup complete.");
}

bool Optimizer::solveMPC(const std::vector<geometry_msgs::msg::Point> & obstacle_points)
{
  // --- Initial Checks ---
  if (!mpc_problem_defined_) {
    RCLCPP_ERROR(logger_, "solveMPC called before setupCasADiProblem was successfully completed!");
    try {
      setupCasADiProblem();
      if (!mpc_problem_defined_) {
        RCLCPP_ERROR(logger_, "Failed to setup CasADi problem on demand within solveMPC.");
        return false;
      }
      RCLCPP_INFO(logger_, "Successfully setup CasADi problem on demand within solveMPC.");
    } catch (const std::exception & e) {
      RCLCPP_ERROR(logger_, "Exception during on-demand setupCasADiProblem in solveMPC: %s",
        e.what());
      return false;
    }
  }
  if (!solver_func_.get()) {
    RCLCPP_ERROR(logger_,
      "solveMPC called but solver_func_ is not valid (null)! Ensure setup was successful.");
    return false;
  }

  auto function_start_time = std::chrono::high_resolution_clock::now();
  int N = settings_.time_steps;
  const double inf = std::numeric_limits<double>::infinity();
  constexpr double kDummyObstacleCoordinate = 1e6;
  constexpr double kDummyObstacleRadiusSq = -1.0;

  double current_x = current_state_.pose.pose.position.x;
  double current_y = current_state_.pose.pose.position.y;
  double current_theta = tf2::getYaw(current_state_.pose.pose.orientation);

  // --- STEP 1: Process Obstacles & Select Milestone Target ---
  auto obs_proc_start_time = std::chrono::high_resolution_clock::now();

  std::vector<ClusterInfo> current_obstacles = processObstacles(obstacle_points);
  last_processed_obstacles_ = current_obstacles;
  size_t num_detected_obstacles = last_processed_obstacles_.size();
  RCLCPP_DEBUG(logger_, "Processed %zu points into %zu obstacle clusters.",
    obstacle_points.size(), num_detected_obstacles);

  geometry_msgs::msg::Pose current_target_pose = selectCurrentMilestoneTarget();

  auto obs_proc_end_time = std::chrono::high_resolution_clock::now();
  auto obs_proc_duration = std::chrono::duration_cast<std::chrono::microseconds>(
    obs_proc_end_time - obs_proc_start_time);

  // --- STEP 2: Prepare Parameter Vector P ---
  auto param_prep_start_time = std::chrono::high_resolution_clock::now();
  std::vector<double> p_vec;
  p_vec.resize(n_params_);
  int p_idx = 0;

  // -- Fill Initial State (X0) --
  if (static_cast<size_t>(p_idx + n_init_state_params_) > p_vec.size()) {
    RCLCPP_ERROR(logger_, "Parameter vector p_vec too small for initial state!");
    return false;
  }
  p_vec[p_idx++] = current_x;
  p_vec[p_idx++] = current_y;
  p_vec[p_idx++] = current_theta;

  // -- Fill Last Control Command (U_{-1}) --
  if (static_cast<size_t>(p_idx + n_last_control_params_) > p_vec.size()) {
    RCLCPP_ERROR(logger_, "Parameter vector p_vec too small for last control!");
    return false;
  }
  p_vec[p_idx++] = last_v_.scalar();
  p_vec[p_idx++] = last_vy_.scalar();
  p_vec[p_idx++] = last_w_.scalar();

  // -- Fill Local Target Pose (X_ref) --
  if (static_cast<size_t>(p_idx + n_local_target_params_) > p_vec.size()) {
    RCLCPP_ERROR(logger_, "Parameter vector p_vec too small for local target!");
    return false;
  }
  p_vec[p_idx++] = current_target_pose.position.x;
  p_vec[p_idx++] = current_target_pose.position.y;
  p_vec[p_idx++] = tf2::getYaw(current_target_pose.orientation);

  // -- Fill Obstacle Parameters (cx, cy, r_sq) --
  int obs_param_start_idx = p_idx;
  double cos_th = cos(current_theta);
  double sin_th = sin(current_theta);

  for (int j = 0; j < max_obstacles_; ++j) {
    bool is_real = (static_cast<size_t>(j) < num_detected_obstacles);
    double cx_param = kDummyObstacleCoordinate;
    double cy_param = kDummyObstacleCoordinate;
    double r_sq_param = kDummyObstacleRadiusSq;

    if (is_real) {
      double cx_local = last_processed_obstacles_[j].cx;
      double cy_local = last_processed_obstacles_[j].cy;
      r_sq_param = last_processed_obstacles_[j].radius_sq;

      cx_param = current_x + (cx_local * cos_th - cy_local * sin_th);
      cy_param = current_y + (cx_local * sin_th + cy_local * cos_th);

      RCLCPP_DEBUG(
        logger_, "Param obstacle %d (Real): Local(%.2f, %.2f) -> Global(%.2f, %.2f), r_sq=%.2f",
        j, cx_local, cy_local, cx_param, cy_param, r_sq_param);
    } else {
      RCLCPP_DEBUG(logger_, "Param obstacle %d (Dummy): Using dummy values", j);
    }

    if (static_cast<size_t>(p_idx + n_obstacle_params_per_obs_) > p_vec.size()) {
      RCLCPP_ERROR(logger_, "Param index OOB filling NLP obs params j=%d. p_idx=%d, n_params=%d", j,
        p_idx, n_params_);
      return false;
    }
    p_vec[p_idx++] = cx_param;
    p_vec[p_idx++] = cy_param;
    p_vec[p_idx++] = r_sq_param;
  }

  if (p_idx != obs_param_start_idx + n_obstacle_params_) {
    RCLCPP_ERROR(logger_, "NLP Obstacle parameter fill mismatch! Expected end index %d, got %d",
      obs_param_start_idx + n_obstacle_params_, p_idx);
    return false;
  }

  if (p_idx != n_params_) {
    RCLCPP_ERROR(logger_,
      "NLP Parameter vector final size mismatch! Expected total size %d, filled %d",
      n_params_, p_idx);
    return false;
  }

  DM p = DM(p_vec);
  auto param_prep_end_time = std::chrono::high_resolution_clock::now();
  auto param_prep_duration = std::chrono::duration_cast<std::chrono::microseconds>(
    param_prep_end_time - param_prep_start_time);

  // --- STEP 3: Prepare Initial Guess (x0_guess) ---
  auto guess_prep_start_time = std::chrono::high_resolution_clock::now();
  DM x0_guess;
  bool use_prev_solution = last_solve_successful_ && !last_optimal_X_flat_.is_empty() &&
    !last_optimal_U_flat_.is_empty();

  if (use_prev_solution) {
    RCLCPP_DEBUG(logger_, "Using warm start initial guess (shifted previous solution).");
    try {
      DM X_prev =
        DM::reshape(last_optimal_X_flat_, static_cast<casadi_int>(n_states_),
          static_cast<casadi_int>(N + 1));
      DM U_prev = DM::reshape(last_optimal_U_flat_, static_cast<casadi_int>(n_controls_),
          static_cast<casadi_int>(N));

      DM X_guess_mat = horzcat(X_prev(Slice(), Slice(1, N + 1)), X_prev(Slice(), N));
      DM U_guess_mat = horzcat(U_prev(Slice(), Slice(1, N)), U_prev(Slice(), N - 1));

      x0_guess = vertcat(
        DM::reshape(X_guess_mat, n_states_ * (N + 1), 1),
        DM::reshape(U_guess_mat, n_controls_ * N, 1));
    } catch (const std::exception & e) {
      RCLCPP_ERROR(logger_,
        "Error reshaping previous solution for warm start: %s. Falling back to cold start.",
        e.what());
      use_prev_solution = false;
      last_solve_successful_ = false;
    }
  }

  if (!use_prev_solution) {
    RCLCPP_WARN(logger_, "Using cold start initial guess (Reference Horizon + Zero Controls).");
    std::vector<double> x0_vec(n_opt_vars_, 0.0);

    std::vector<geometry_msgs::msg::Pose> ref_poses_for_guess = generateReferenceHorizon(
      current_target_pose);

    if (ref_poses_for_guess.size() == static_cast<size_t>(N + 1)) {
      for (int k = 0; k <= N; ++k) {
        int base_idx = k * n_states_;
        if (static_cast<size_t>(base_idx + n_states_) > x0_vec.size()) {
          RCLCPP_ERROR(logger_, "Index out of bounds during cold start state guess fill (k=%d)", k);
          break;
        }
        x0_vec[base_idx + 0] = ref_poses_for_guess[k].position.x;
        x0_vec[base_idx + 1] = ref_poses_for_guess[k].position.y;
        x0_vec[base_idx + 2] = tf2::getYaw(ref_poses_for_guess[k].orientation);
      }
    } else {
      RCLCPP_ERROR(
        logger_,
        "Reference trajectory size mismatch (%zu vs %d) in cold start guess prep! Falling back to current state.",
        ref_poses_for_guess.size(), N + 1);
      for (int k = 0; k <= N; ++k) {
        int base_idx = k * n_states_;
        if (static_cast<size_t>(base_idx + n_states_) > x0_vec.size()) {
          RCLCPP_ERROR(logger_, "Index out of bounds during cold start fallback guess fill (k=%d)",
            k);
          break;
        }
        x0_vec[base_idx + 0] = current_x;
        x0_vec[base_idx + 1] = current_y;
        x0_vec[base_idx + 2] = current_theta;
      }
    }
    x0_guess = DM(x0_vec);
  }
  auto guess_prep_end_time = std::chrono::high_resolution_clock::now();
  auto guess_prep_duration = std::chrono::duration_cast<std::chrono::microseconds>(
    guess_prep_end_time - guess_prep_start_time);


  // --- STEP 4: Prepare Bounds (lbx, ubx, lbg, ubg) ---
  auto bounds_prep_start_time = std::chrono::high_resolution_clock::now();

  DM lbx = DM(x_flat_lower_bounds_);
  DM ubx = DM(x_flat_upper_bounds_);
  DM lbg = DM(g_flat_lower_bounds_);
  DM ubg = DM(g_flat_upper_bounds_);

  // -- Update initial state variable bounds --
  for (int i = 0; i < n_states_; ++i) {
    if (i < lbx.rows() && i < ubx.rows()) {
      lbx(i) = p_vec[i];
      ubx(i) = p_vec[i];
    } else {
      RCLCPP_ERROR(logger_, "Index OOB when setting initial state bounds (i=%d)", i);
    }
  }

  // -- Dynamically update dummy obstacle constraint bounds --
  for (int j = 0; j < max_obstacles_; ++j) {
    int r_sq_p_idx = obs_param_start_idx + j * n_obstacle_params_per_obs_ + 2;
    bool is_dummy = false;
    if (r_sq_p_idx >= 0 && static_cast<size_t>(r_sq_p_idx) < p_vec.size()) {
      is_dummy = (p_vec[r_sq_p_idx] < 0.0);
    } else {
      RCLCPP_ERROR_THROTTLE(logger_, *clock_, 1000,
        "Index OOB checking r_sq in p_vec for dummy obstacle check (j=%d, r_sq_p_idx=%d)", j,
        r_sq_p_idx);
      continue;
    }

    if (is_dummy) {
      RCLCPP_DEBUG(logger_, "Relaxing bounds for dummy obstacle slot %d", j);
      for (int k = 1; k <= N; ++k) {
        int const_idx = obs_constraint_start_idx_ + j * N + (k - 1);
        if (const_idx >= 0 && const_idx < lbg.rows()) {
          lbg(const_idx) = -inf;
        } else {
          RCLCPP_ERROR_THROTTLE(logger_, *clock_, 1000,
            "Constraint index OOB when relaxing dummy obs j=%d, k=%d -> const_idx=%d (lbg rows=%d)",
            j, k, const_idx,
            (int)lbg.rows());
        }
      }
    }
  }

  auto bounds_prep_end_time = std::chrono::high_resolution_clock::now();
  auto bounds_prep_duration = std::chrono::duration_cast<std::chrono::microseconds>(
    bounds_prep_end_time - bounds_prep_start_time);

  // --- STEP 5: Call Solver ---
  DMDict arg = {{"p", p},
    {"x0", x0_guess},
    {"lbx", lbx},
    {"ubx", ubx},
    {"lbg", lbg},
    {"ubg", ubg}};

  DMDict res;
  auto solve_start_time = std::chrono::high_resolution_clock::now();
  try {
    RCLCPP_DEBUG(logger_, "Calling CasADi nlpsol function...");
    res = solver_func_(arg);
    RCLCPP_DEBUG(logger_, "CasADi nlpsol function returned.");
  } catch (const std::exception & e) {
    RCLCPP_ERROR(logger_, "CasADi solver threw exception during solve: %s", e.what());
    last_solve_successful_ = false;
    last_optimal_X_flat_ = casadi::DM();
    last_optimal_U_flat_ = casadi::DM();
    return false;
  }
  auto solve_end_time = std::chrono::high_resolution_clock::now();
  auto solve_duration = std::chrono::duration_cast<std::chrono::microseconds>(
    solve_end_time - solve_start_time);
  RCLCPP_DEBUG(logger_, "CasADi solve time: %.3f ms", solve_duration.count() / 1000.0);

  // --- STEP 6: Process Results ---
  try {
    std::string solve_status = static_cast<std::string>(solver_func_.stats().at("return_status"));
    bool success = solver_func_.stats().at("success");

    if (success) {
      DM sol = res.at("x");
      last_optimal_X_flat_ = sol(Slice(0, n_states_ * (N + 1)));
      last_optimal_U_flat_ = sol(Slice(n_states_ * (N + 1), n_opt_vars_));
      last_solve_successful_ = true;
      RCLCPP_DEBUG(logger_, "Solve successful. Final cost: %f, Status: %s",
        static_cast<double>(res.at("f")), solve_status.c_str());
    } else {
      RCLCPP_WARN(logger_, "NLP solver did not find an optimal solution. Status: %s",
        solve_status.c_str());
      last_solve_successful_ = false;
      last_optimal_X_flat_ = casadi::DM();
      last_optimal_U_flat_ = casadi::DM();
    }

    auto function_end_time = std::chrono::high_resolution_clock::now();
    auto function_duration = std::chrono::duration_cast<std::chrono::microseconds>(
      function_end_time - function_start_time);
    RCLCPP_DEBUG(
      logger_,
      "solveMPC total time: %.3f ms (Obs+Tgt: %.3f, Param: %.3f, Guess: %.3f, Bounds: %.3f, Solve: %.3f)",
      function_duration.count() / 1000.0,
      obs_proc_duration.count() / 1000.0,
      param_prep_duration.count() / 1000.0,
      guess_prep_duration.count() / 1000.0,
      bounds_prep_duration.count() / 1000.0,
      solve_duration.count() / 1000.0);

    return success;

  } catch (const std::exception & e) {
    RCLCPP_ERROR(logger_, "Exception processing solver results: %s", e.what());
    last_solve_successful_ = false;
    last_optimal_X_flat_ = casadi::DM();
    last_optimal_U_flat_ = casadi::DM();
    return false;
  }
}

const std::vector<ClusterInfo> & Optimizer::getLastProcessedObstacles() const
{
  return last_processed_obstacles_;
}

std::optional<geometry_msgs::msg::Pose> Optimizer::getLastLocalTarget() const
{
  return last_local_target_;
}

// --- Helper Functions (Clustering, Bounding Circle) ---

ClusterInfo Optimizer::computeBoundingCircle(
  const std::vector<geometry_msgs::msg::Point> & cluster_points)
{
  ClusterInfo info;
  info.num_points = cluster_points.size();
  if (info.num_points == 0) {
    return info;
  }

  double sum_x = 0.0, sum_y = 0.0;
  for (const auto & pt : cluster_points) {
    sum_x += pt.x;
    sum_y += pt.y;
  }
  info.cx = sum_x / info.num_points;
  info.cy = sum_y / info.num_points;

  double max_dist_sq = 0.0;
  for (const auto & pt : cluster_points) {
    double dx = pt.x - info.cx;
    double dy = pt.y - info.cy;
    max_dist_sq = std::max(max_dist_sq, dx * dx + dy * dy);
  }
  info.radius_sq = max_dist_sq/2;

  return info;
}

std::vector<std::vector<geometry_msgs::msg::Point>> Optimizer::clusterPointsInternal(
  const std::vector<geometry_msgs::msg::Point> & points_to_cluster, double tolerance, int min_size)
{
  std::vector<std::vector<geometry_msgs::msg::Point>> clusters;
  if (points_to_cluster.empty() || min_size <= 0) {
    return clusters;
  }

  size_t n_points = points_to_cluster.size();
  std::vector<bool> visited(n_points, false);
  double tolerance_sq = tolerance * tolerance;

  for (size_t i = 0; i < n_points; ++i) {
    if (!visited[i]) {
      std::vector<size_t> current_indices_local;
      std::vector<size_t> q;
      q.push_back(i);
      visited[i] = true;

      size_t head = 0;
      while (head < q.size()) {
        size_t current_idx_local = q[head++];
        current_indices_local.push_back(current_idx_local);
        const auto & p1 = points_to_cluster[current_idx_local];

        for (size_t j = 0; j < n_points; ++j) {
          if (!visited[j]) {
            const auto & p2 = points_to_cluster[j];
            double dx = p1.x - p2.x;
            double dy = p1.y - p2.y;
            if ((dx * dx + dy * dy) < tolerance_sq) {
              visited[j] = true;
              q.push_back(j);
            }
          }
        }
      }

      if (current_indices_local.size() >= static_cast<size_t>(min_size)) {
        std::vector<geometry_msgs::msg::Point> current_cluster_points;
        current_cluster_points.reserve(current_indices_local.size());
        for (size_t local_idx : current_indices_local) {
          current_cluster_points.push_back(points_to_cluster[local_idx]);
        }
        clusters.push_back(current_cluster_points);
      }
    }
  }
  return clusters;
}

// --- Obstacle Processing ---

std::vector<ClusterInfo> Optimizer::processObstacles(
  const std::vector<geometry_msgs::msg::Point> & points)
{
  std::vector<ClusterInfo> final_obstacle_info_list;
  if (points.empty()) {
    return final_obstacle_info_list;
  }

  RCLCPP_DEBUG(logger_, "Running initial clustering with tolerance %.2f", lidar_cluster_tolerance_);
  std::vector<std::vector<geometry_msgs::msg::Point>> initial_clusters =
    clusterPointsInternal(points, lidar_cluster_tolerance_, lidar_min_cluster_size_);
  RCLCPP_DEBUG(logger_, "Initial clustering found %zu potential clusters.", initial_clusters.size());

  double max_radius_sq = max_cluster_radius_ * max_cluster_radius_;
  double split_tolerance = lidar_cluster_tolerance_ / 2.0;

  for (const auto & current_cluster_points : initial_clusters) {
    if (current_cluster_points.empty()) {
      continue;
    }

    ClusterInfo initial_info = computeBoundingCircle(current_cluster_points);

    if (initial_info.num_points > 0 && initial_info.radius_sq > max_radius_sq) {
      RCLCPP_DEBUG(
        logger_, "Cluster too large (radius_sq=%.2f > %.2f), attempting split with tolerance %.2f.",
        initial_info.radius_sq, max_radius_sq, split_tolerance);

      std::vector<std::vector<geometry_msgs::msg::Point>> sub_clusters_points =
        clusterPointsInternal(current_cluster_points, split_tolerance, lidar_min_cluster_size_);
      RCLCPP_DEBUG(logger_, "Splitting resulted in %zu sub-clusters.", sub_clusters_points.size());

      for (const auto & sub_cluster : sub_clusters_points) {
        if (!sub_cluster.empty()) {
          ClusterInfo sub_info = computeBoundingCircle(sub_cluster);
          if (sub_info.num_points > 0) {
             if (sub_info.radius_sq <= max_radius_sq) {
                 final_obstacle_info_list.push_back(sub_info);
             } else {
                 RCLCPP_DEBUG(logger_, "Sub-cluster still too large (radius_sq=%.2f), discarding.", sub_info.radius_sq);
             }
          }
        }
      }
    } else if (initial_info.num_points > 0) {
      RCLCPP_DEBUG(logger_, "Cluster accepted (radius_sq=%.2f <= %.2f).", initial_info.radius_sq,
        max_radius_sq);
      final_obstacle_info_list.push_back(initial_info);
    }
  }

  if (final_obstacle_info_list.size() > static_cast<size_t>(max_obstacles_)) {
    std::sort(
      final_obstacle_info_list.begin(), final_obstacle_info_list.end(),
      [this](const ClusterInfo & a, const ClusterInfo & b) {
        double current_x = current_state_.pose.pose.position.x;
        double current_y = current_state_.pose.pose.position.y;
        double dist_sq_a = std::pow(a.cx - current_x, 2) + std::pow(a.cy - current_y, 2);
        double dist_sq_b = std::pow(b.cx - current_x, 2) + std::pow(b.cy - current_y, 2);
        return dist_sq_a < dist_sq_b;
      });

    RCLCPP_WARN_THROTTLE(
      logger_, *clock_, 5000,
      "Detected %zu final clusters after splitting/filtering, limiting to %d closest for constraints.",
      final_obstacle_info_list.size(), max_obstacles_);
    final_obstacle_info_list.resize(max_obstacles_);
  }

  RCLCPP_DEBUG(logger_, "Returning %zu final obstacles for MPC.", final_obstacle_info_list.size());
  return final_obstacle_info_list;
}

// --- Milestone and Reference Generation ---
std::vector<geometry_msgs::msg::Pose> Optimizer::generateReferenceHorizon(
  const geometry_msgs::msg::Pose & target_pose)
{
  int N = settings_.time_steps;
  std::vector<geometry_msgs::msg::Pose> ref_poses;
  ref_poses.resize(N + 1);

  ref_poses[0] = current_state_.pose.pose;

  double start_x = current_state_.pose.pose.position.x;
  double start_y = current_state_.pose.pose.position.y;
  double start_yaw = tf2::getYaw(current_state_.pose.pose.orientation);

  double target_x = target_pose.position.x;
  double target_y = target_pose.position.y;
  double target_yaw = tf2::getYaw(target_pose.orientation);

  double delta_yaw = target_yaw - start_yaw;
  delta_yaw = atan2(sin(delta_yaw), cos(delta_yaw));

  for (int k = 1; k <= N; ++k) {
    double ratio = static_cast<double>(k) / static_cast<double>(N);
    ref_poses[k].position.x = start_x + ratio * (target_x - start_x);
    ref_poses[k].position.y = start_y + ratio * (target_y - start_y);
    ref_poses[k].position.z = 0.0;

    double interp_yaw = start_yaw + ratio * delta_yaw;
    tf2::Quaternion q;
    q.setRPY(0, 0, interp_yaw);
    ref_poses[k].orientation = tf2::toMsg(q);
  }

  RCLCPP_DEBUG(logger_, "Generated reference horizon for cold start guess.");
  return ref_poses;
}

double Optimizer::calculatePathCurvature(const nav_msgs::msg::Path & path, size_t index)
{
   constexpr double kMinSegmentLenSq = 1e-6; // Avoid sqrt
   constexpr double kSmallSegmentLen = 1e-3;

   if (index == 0 || index >= path.poses.size() - 1) {
     return 0.0;
   }

   const auto & p_prev = path.poses[index - 1].pose;
   const auto & p_curr = path.poses[index].pose;
   const auto & p_next = path.poses[index + 1].pose;

   double dx1 = p_curr.position.x - p_prev.position.x;
   double dy1 = p_curr.position.y - p_prev.position.y;
   double len1_sq = dx1 * dx1 + dy1 * dy1;

   double dx2 = p_next.position.x - p_curr.position.x;
   double dy2 = p_next.position.y - p_curr.position.y;
   double len2_sq = dx2 * dx2 + dy2 * dy2;

   if (len1_sq < kMinSegmentLenSq || len2_sq < kMinSegmentLenSq) {
     return 0.0;
   }

   double len1 = std::sqrt(len1_sq);
   double len2 = std::sqrt(len2_sq);

   if (len1 < kSmallSegmentLen || len2 < kSmallSegmentLen) {
       return 0.0;
   }

   double yaw1 = std::atan2(dy1, dx1);
   double yaw2 = std::atan2(dy2, dx2);
   double dyaw = yaw2 - yaw1;
   dyaw = atan2(sin(dyaw), cos(dyaw));

   double avg_len = (len1 + len2) / 2.0;
   return std::abs(dyaw) / avg_len;
}

void Optimizer::generateMilestones(const nav_msgs::msg::Path & path)
{
  current_milestones_.clear();
  current_milestone_idx_ = 0;

  constexpr double kMinSegmentLen = 1e-4;
  constexpr double kMilestoneCheckStep = 0.1;
  constexpr double kFinalGoalTol = 1e-3;

  if (path.poses.size() < 2) {
    if (!path.poses.empty()) {
      current_milestones_.push_back(path.poses.back().pose);
    }
    RCLCPP_WARN(logger_, "Path too short (%zu poses) to generate milestones.", path.poses.size());
    return;
  }

  current_milestones_.push_back(path.poses[0].pose);
  if (current_milestones_.back().orientation.w == 0 && current_milestones_.back().orientation.x == 0 &&
      current_milestones_.back().orientation.y == 0 && current_milestones_.back().orientation.z == 0)
  {
      double dx = path.poses[1].pose.position.x - path.poses[0].pose.position.x;
      double dy = path.poses[1].pose.position.y - path.poses[0].pose.position.y;
      tf2::Quaternion q;
      q.setRPY(0, 0, std::atan2(dy, dx));
      current_milestones_.back().orientation = tf2::toMsg(q);
  } else if (current_milestones_.back().orientation.w == 0) {
      current_milestones_.back().orientation.w = 1.0; 
  }


  double accumulated_dist = 0.0;

  for (size_t i = 0; i < path.poses.size() - 1; ++i) {
    const auto & p1 = path.poses[i].pose;
    const auto & p2 = path.poses[i + 1].pose;

    double seg_dx = p2.position.x - p1.position.x;
    double seg_dy = p2.position.y - p1.position.y;
    double seg_len = std::hypot(seg_dx, seg_dy);

    if (seg_len < kMinSegmentLen) {
      continue;
    }

    double current_pos_on_segment = 0.0;
    while (current_pos_on_segment < seg_len) {
      double curvature = calculatePathCurvature(path, i);
      double desired_spacing = milestone_max_spacing_ / (1.0 + milestone_curvature_factor_ * curvature);
      desired_spacing = std::max(milestone_min_spacing_, desired_spacing);

      double dist_since_last_milestone = accumulated_dist + current_pos_on_segment;

      if (dist_since_last_milestone >= desired_spacing) {
        double ratio = (seg_len > kMinSegmentLen) ? (current_pos_on_segment / seg_len) : 0.0;
        geometry_msgs::msg::Pose new_milestone;
        new_milestone.position.x = p1.position.x + ratio * seg_dx;
        new_milestone.position.y = p1.position.y + ratio * seg_dy;
        new_milestone.position.z = p1.position.z;

        double yaw1 = tf2::getYaw(p1.orientation);
        double yaw2 = tf2::getYaw(p2.orientation);
        bool p1_quat_invalid = (p1.orientation.w == 0 && p1.orientation.x == 0 && p1.orientation.y == 0 && p1.orientation.z == 0);
        bool p2_quat_invalid = (p2.orientation.w == 0 && p2.orientation.x == 0 && p2.orientation.y == 0 && p2.orientation.z == 0);

        if (p1_quat_invalid) { yaw1 = std::atan2(seg_dy, seg_dx); }
        if (p2_quat_invalid) { yaw2 = std::atan2(seg_dy, seg_dx); }

        double dyaw = yaw2 - yaw1;
        dyaw = atan2(sin(dyaw), cos(dyaw));
        double interp_yaw = yaw1 + ratio * dyaw;

        tf2::Quaternion q;
        q.setRPY(0, 0, interp_yaw);
        new_milestone.orientation = tf2::toMsg(q);

        current_milestones_.push_back(new_milestone);
        accumulated_dist = 0.0;
      }

      double remaining_seg_len = seg_len - current_pos_on_segment;
      double check_step = std::min(remaining_seg_len, kMilestoneCheckStep);

      if (check_step < kMinSegmentLen) {
        break;
      }

      if (accumulated_dist + check_step < desired_spacing) {
        accumulated_dist += check_step;
        current_pos_on_segment += check_step;
      } else {
        double dist_needed = desired_spacing - accumulated_dist;
        dist_needed = std::max(0.0, dist_needed);
        dist_needed = std::min(dist_needed, remaining_seg_len);

        current_pos_on_segment += dist_needed;
        accumulated_dist += dist_needed;
      }
       if (std::abs(seg_len - current_pos_on_segment) < kMinSegmentLen / 10.0) {
             current_pos_on_segment = seg_len;
        }
    }
    accumulated_dist += (seg_len - current_pos_on_segment);
    accumulated_dist = std::max(0.0, accumulated_dist);
  }

  const auto & final_goal = path.poses.back().pose;
  if (current_milestones_.empty() ||
    (std::hypot(
      final_goal.position.x - current_milestones_.back().position.x,
      final_goal.position.y - current_milestones_.back().position.y) > kFinalGoalTol))
  {
    geometry_msgs::msg::Pose final_milestone = final_goal;
    if (final_milestone.orientation.w == 0 && final_milestone.orientation.x == 0 &&
        final_milestone.orientation.y == 0 && final_milestone.orientation.z == 0)
    {
      if (path.poses.size() > 1) {
        const auto & prev_pose = path.poses[path.poses.size() - 2].pose;
        double dx = final_milestone.position.x - prev_pose.position.x;
        double dy = final_milestone.position.y - prev_pose.position.y;
        tf2::Quaternion q;
        q.setRPY(0, 0, std::atan2(dy, dx));
        final_milestone.orientation = tf2::toMsg(q);
      } else {
        final_milestone.orientation.w = 1.0;
      }
    } else if (final_milestone.orientation.w == 0) {
         final_milestone.orientation.w = 1.0;
    }
    current_milestones_.push_back(final_milestone);
  }

  RCLCPP_INFO(logger_, "Generated %zu milestones from global path.", current_milestones_.size());
}

geometry_msgs::msg::Pose Optimizer::selectCurrentMilestoneTarget()
{
  if (current_milestones_.empty()) {
    RCLCPP_WARN_ONCE(logger_, "No milestones available, targeting current robot pose.");
    last_local_target_ = std::nullopt;
    return current_state_.pose.pose;
  }

  if (current_milestone_idx_ >= current_milestones_.size()) {
    current_milestone_idx_ = current_milestones_.size() - 1;
  }

  geometry_msgs::msg::Pose current_target = current_milestones_[current_milestone_idx_];

  if (current_milestone_idx_ < current_milestones_.size() - 1) {
    double dx = current_target.position.x - current_state_.pose.pose.position.x;
    double dy = current_target.position.y - current_state_.pose.pose.position.y;
    double dist_sq_to_target = dx * dx + dy * dy;

    double threshold_sq = milestone_reached_threshold_ * milestone_reached_threshold_;

    if (dist_sq_to_target < threshold_sq) {
      current_milestone_idx_++;
      current_target = current_milestones_[current_milestone_idx_];
      RCLCPP_INFO(
        logger_, "Milestone %zu reached. Switching target to milestone %zu (%.2f, %.2f)",
        current_milestone_idx_ - 1, current_milestone_idx_,
        current_target.position.x, current_target.position.y);
    }
  }

  last_local_target_ = current_target;

  RCLCPP_DEBUG(
    logger_, "Selected milestone %zu (%.2f, %.2f) as local target.",
    current_milestone_idx_, current_target.position.x, current_target.position.y);

  return current_target;
}

} // namespace sortham
