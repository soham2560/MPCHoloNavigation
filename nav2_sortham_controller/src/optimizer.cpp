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

#include "nav2_costmap_2d/costmap_filters/filter_values.hpp"
#include <casadi/casadi.hpp>
#include "tf2/utils.h"
#include <tf2/LinearMath/Quaternion.h>
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

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
  auto getParentParam = parameters_handler_->getParamGetter("");
  getParam(s.model_dt, "model_dt", 0.05f);
  getParam(s.time_steps, "time_steps", 56);
  getParam(s.base_constraints.vx_max, "vx_max", 0.5);
  getParam(s.base_constraints.vx_min, "vx_min", -0.35);
  getParam(s.base_constraints.vy, "vy_max", 0.5);
  getParam(s.base_constraints.wz, "wz_max", 1.9);

  // MPC Weights
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

  // Obstacle parameters
  getParam(robot_radius_, "robot_radius", 0.25);
  getParam(lidar_safety_margin_, "lidar_safety_margin", 0.05);
  getParam(lidar_cluster_tolerance_, "lidar_cluster_tolerance", 0.05);
  getParam(lidar_min_cluster_size_, "lidar_min_cluster_size", 3);
  getParam(max_obstacles_, "max_obstacles", 10);
  effective_radius_padding_sq_ = std::pow(robot_radius_ + lidar_safety_margin_, 2);
  RCLCPP_INFO(logger_, "Lidar constraints: radius=%.2f, margin=%.2f, cluster_tol=%.2f, min_cluster=%d, max_obs=%d",
        robot_radius_, lidar_safety_margin_, lidar_cluster_tolerance_, lidar_min_cluster_size_, max_obstacles_);

  // Motion model
  getParam(motion_model_name, "motion_model", std::string("Omni"));

  s.constraints = s.base_constraints;
  setMotionModel(motion_model_name);
  if (!isHolonomic()) {
      RCLCPP_ERROR(logger_, "MPC currently setup assuming Omni model. Non-holonomic model '%s' requires code changes.", motion_model_name.c_str());
      throw std::runtime_error("Non-holonomic model requires MPC code adaptation.");
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

}


void Optimizer::reset()
{
  last_optimal_X_flat_ = casadi::DM();
  last_optimal_U_flat_ = casadi::DM();
  last_v_ = DM(0);
  last_w_ = DM(0);
  last_vy_ = DM(0);
  last_solve_successful_ = false;

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
  // Store current state and plan/goal info
  current_state_.pose = robot_pose;
  current_state_.speed = robot_speed;
  path_ = utils::toTensor(plan);
  goal_ = goal;

  if (!mpc_problem_defined_) {
      RCLCPP_INFO(logger_, "Setting up CasADi NLP problem...");
      setupCasADiProblem();
      RCLCPP_INFO(logger_, "CasADi NLP problem setup complete.");
  }

  RCLCPP_DEBUG(logger_, "Attempting to solve MPC problem...");
  bool success = solveMPC(obstacle_points);

  if (success) {
      RCLCPP_DEBUG(logger_, "MPC solve successful.");
      double vx_cmd = last_optimal_U_flat_(0).scalar();
      double vy_cmd = last_optimal_U_flat_(1).scalar();
      double wz_cmd = last_optimal_U_flat_(2).scalar();

      last_v_ = vx_cmd;
      last_vy_ = vy_cmd;
      last_w_ = wz_cmd;

      RCLCPP_DEBUG(logger_, "MPC Command: [vx=%.3f, vy=%.3f, wz=%.3f]", vx_cmd, vy_cmd, wz_cmd);

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

// --- isHolonomic, setMotionModel, setSpeedLimit - Implementations remain the same ---
bool Optimizer::isHolonomic() const {
    if (!motion_model_) {
        RCLCPP_ERROR(logger_, "Motion model not initialized in isHolonomic check!");
        return true; // Or throw
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

  RCLCPP_DEBUG(logger_, "setSpeedLimit called with limit: %.2f, percentage: %d", speed_limit, percentage);

  auto new_limits = s.base_constraints; // Start with base limits

  if (speed_limit >= epsilon) { // Apply limit if positive
    double ratio = 1.0;
    if (percentage) {
      ratio = speed_limit / 100.0;
      RCLCPP_INFO(logger_, "Setting speed limit to %.2f%% of base limits (ratio %.2f).", speed_limit, ratio);
    } else {
      if (s.base_constraints.vx_max > epsilon) {
         ratio = speed_limit / s.base_constraints.vx_max;
         ratio = std::min(1.0, ratio); // Clamp ratio
         RCLCPP_INFO(logger_, "Setting speed limit to absolute %.2f m/s (ratio %.2f).", speed_limit, ratio);
      } else {
          RCLCPP_WARN(logger_, "Cannot set absolute speed limit: base vx_max is zero.");
          return;
      }
    }
    new_limits.vx_max = s.base_constraints.vx_max * ratio;
    new_limits.vx_min = (s.base_constraints.vx_min < 0) ? (s.base_constraints.vx_min * ratio) : std::min(0.0f, new_limits.vx_max);
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
      RCLCPP_INFO(logger_, "New velocity limits: vx[%.2f, %.2f], vy +/-%.2f, wz +/-%.2f",
            s.constraints.vx_min, s.constraints.vx_max, s.constraints.vy, s.constraints.wz);
      mpc_problem_defined_ = false; // Force recreation due to bound change
  } else {
      RCLCPP_DEBUG(logger_, "Calculated speed limits are same as current ones.");
  }
}


// getOptimizedTrajectory - Implementation remains the same
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
      trajectory(i, 0) = static_cast<float>(x_vec[i * n_states_ + 0]); // x
      trajectory(i, 1) = static_cast<float>(x_vec[i * n_states_ + 1]); // y
      trajectory(i, 2) = static_cast<float>(x_vec[i * n_states_ + 2]); // theta
  }
  RCLCPP_DEBUG(logger_, "getOptimizedTrajectory: Returning trajectory with %zu points.", trajectory.shape()[0]);
  return trajectory;
}

void Optimizer::setupCasADiProblem()
{
    RCLCPP_INFO(logger_, "Setting up CasADi QP symbolic structure...");
    // ==============================================================
    // 1. Dimensions & Symbolic Variables (Decision Vars V = [X; U])
    // ==============================================================
    n_states_ = 3; n_controls_ = 3;
    int N = settings_.time_steps;
    // double T = settings_.model_dt; // T is used implicitly in linearized params

    MX X_sym = MX::sym("X", n_states_, N + 1); // States matrix [x; y; th] for k=0...N
    MX U_sym = MX::sym("U", n_controls_, N);   // Controls matrix [vx; vy; wz] for k=0...N-1
    V_sym_ = vertcat(reshape(X_sym, n_states_ * (N + 1), 1),
                     reshape(U_sym, n_controls_ * N, 1));
    n_opt_vars_ = V_sym_.rows();

    // ==============================================================
    // 2. Symbolic Parameters P (Init State, Ref, Last Ctrl, Lin Params)
    // ==============================================================
    n_init_state_params_ = n_states_;
    n_ref_params_ = n_states_ * (N + 1); // Reference states for objective
    n_last_control_params_ = n_controls_;
    // Lin Dynamics Params: A_k(nxn), B_k(nxm), c_k(nx1) for k=0..N-1
    int n_A_params_per_step = n_states_ * n_states_;
    int n_B_params_per_step = n_states_ * n_controls_;
    int n_c_params_per_step = n_states_;
    n_dyn_lin_params_ = N * (n_A_params_per_step + n_B_params_per_step + n_c_params_per_step);
    // Lin Obstacle Params: n_jk(2x1 normal vector), rhs_jk(scalar) for j=0..max_obs-1, k=1..N
    int n_obs_norm_params_per_step = 2; // nx, ny
    int n_obs_rhs_params_per_step = 1;  // scalar rhs
    n_obs_lin_params_ = max_obstacles_ * N * (n_obs_norm_params_per_step + n_obs_rhs_params_per_step);

    n_params_ = n_init_state_params_ + n_ref_params_ + n_last_control_params_ +
                 n_dyn_lin_params_ + n_obs_lin_params_;
    P_sym_ = MX::sym("P", n_params_);
    RCLCPP_INFO(logger_, "QP Params: N=%d, n_states=%d, n_controls=%d, n_params=%d (init=%d, ref=%d, last_ctrl=%d, dyn_lin=%d, obs_lin=%d)",
        N, n_states_, n_controls_, n_params_, n_init_state_params_, n_ref_params_, n_last_control_params_, n_dyn_lin_params_, n_obs_lin_params_);

    // Symbolic parameter slices
    int current_p_idx = 0;
    Slice init_state_slice(current_p_idx, current_p_idx + n_init_state_params_); current_p_idx += n_init_state_params_;
    Slice ref_traj_slice(current_p_idx, current_p_idx + n_ref_params_); current_p_idx += n_ref_params_;
    Slice last_ctrl_slice(current_p_idx, current_p_idx + n_last_control_params_); current_p_idx += n_last_control_params_;
    Slice dyn_lin_slice(current_p_idx, current_p_idx + n_dyn_lin_params_); current_p_idx += n_dyn_lin_params_;
    Slice obs_lin_slice(current_p_idx, current_p_idx + n_obs_lin_params_);

    MX P_init_state = P_sym_(init_state_slice);
    MX P_ref_traj_flat = P_sym_(ref_traj_slice);
    MX P_ref_traj = reshape(P_ref_traj_flat, n_states_, N + 1);
    MX P_last_ctrl = P_sym_(last_ctrl_slice);
    MX P_dyn_lin_flat = P_sym_(dyn_lin_slice);
    MX P_obs_lin_flat = P_sym_(obs_lin_slice);

    // ==============================================================
    // 3. Define Symbolic Quadratic Objective J
    // ==============================================================
    MX J = 0;
    // --- Weight Matrices ---
    DM Q = DM::diag(DM({settings_.weight_x, settings_.weight_y, settings_.weight_theta}));
    DM QN = DM::diag(DM({settings_.weight_terminal_x, settings_.weight_terminal_y, settings_.weight_terminal_theta}));
    DM R = DM::diag(DM({settings_.weight_v, settings_.weight_vy, settings_.weight_w}));
    DM Rd = DM::diag(DM({settings_.weight_v_accel, settings_.weight_vy_accel, settings_.weight_w_accel}));

    // --- Sum stage costs (k=0 to N-1) ---
    for (int k = 0; k < N; ++k) {
        MX x_k = X_sym(Slice(), k); MX u_k = U_sym(Slice(), k); MX x_ref_k = P_ref_traj(Slice(), k);
        MX state_error = x_k - x_ref_k;
        J += mtimes(state_error.T(), mtimes(Q, state_error));
        J += mtimes(u_k.T(), mtimes(R, u_k));
        MX u_prev = (k == 0) ? P_last_ctrl : U_sym(Slice(), k - 1);
        MX delta_u = u_k - u_prev;
        J += mtimes(delta_u.T(), mtimes(Rd, delta_u));
    }
    // --- Terminal cost (k=N) ---
    MX x_N = X_sym(Slice(), N); MX x_ref_N = P_ref_traj(Slice(), N);
    MX terminal_error = x_N - x_ref_N;
    J += mtimes(terminal_error.T(), mtimes(QN, terminal_error));

    MX x_sym_dyn = MX::sym("x_dyn", n_states_, 1);
    MX u_sym_dyn = MX::sym("u_dyn", n_controls_, 1);
    MX theta_sym_dyn = x_sym_dyn(2);
    MX vx_sym_dyn = u_sym_dyn(0); MX vy_sym_dyn = u_sym_dyn(1); MX wz_sym_dyn = u_sym_dyn(2);
    MX f_dyn_expr = vertcat(
        x_sym_dyn(0) + (vx_sym_dyn * cos(theta_sym_dyn) - vy_sym_dyn * sin(theta_sym_dyn)) * settings_.model_dt,
        x_sym_dyn(1) + (vx_sym_dyn * sin(theta_sym_dyn) + vy_sym_dyn * cos(theta_sym_dyn)) * settings_.model_dt,
        x_sym_dyn(2) + wz_sym_dyn * settings_.model_dt
    );
    MX Jx_sym_dyn = jacobian(f_dyn_expr, x_sym_dyn);
    MX Ju_sym_dyn = jacobian(f_dyn_expr, u_sym_dyn);

    dyn_func_ = Function("f", {x_sym_dyn, u_sym_dyn}, {f_dyn_expr}, {"x_k", "u_k"}, {"x_k_plus_1"});
    dyn_jac_x_func_ = Function("Jx", {x_sym_dyn, u_sym_dyn}, {Jx_sym_dyn}, {"x_k", "u_k"}, {"Jx"});
    dyn_jac_u_func_ = Function("Ju", {x_sym_dyn, u_sym_dyn}, {Ju_sym_dyn}, {"x_k", "u_k"}, {"Ju"});
    RCLCPP_INFO(logger_, "Created dynamics and Jacobian functions.");

    // ==============================================================
    // 4. Define Symbolic Linear Constraints Vector (g)
    // ==============================================================
    MXVector g_vec;
    // --- Initial State Constraint ---
    g_vec.push_back(X_sym(Slice(), 0) - P_init_state);
    // --- Linearized Dynamics Constraints ---
    int dyn_param_idx = 0;
    for (int k = 0; k < N; ++k) {
        MX A_k = reshape(P_dyn_lin_flat(Slice(dyn_param_idx, dyn_param_idx + n_A_params_per_step)), n_states_, n_states_);
        dyn_param_idx += n_A_params_per_step;
        MX B_k = reshape(P_dyn_lin_flat(Slice(dyn_param_idx, dyn_param_idx + n_B_params_per_step)), n_states_, n_controls_);
        dyn_param_idx += n_B_params_per_step;
        MX c_k = P_dyn_lin_flat(Slice(dyn_param_idx, dyn_param_idx + n_c_params_per_step));
        dyn_param_idx += n_c_params_per_step;
        MX dyn_defect_k = X_sym(Slice(), k+1) - (mtimes(A_k, X_sym(Slice(), k)) + mtimes(B_k, U_sym(Slice(), k)) + c_k);
        g_vec.push_back(dyn_defect_k);
    }
    int dynamics_constraints_end_idx = g_vec.size();
    // --- Linearized Obstacle Constraints ---
    int obs_lin_param_idx = 0;
    for (int j = 0; j < max_obstacles_; ++j) {
        for (int k = 1; k <= N; ++k) {
            MX n_jk = P_obs_lin_flat(Slice(obs_lin_param_idx, obs_lin_param_idx + n_obs_norm_params_per_step));
            obs_lin_param_idx += n_obs_norm_params_per_step;
            obs_lin_param_idx += n_obs_rhs_params_per_step; // Skip RHS param symbolically
            MX linear_obs_expr = mtimes(n_jk.T(), X_sym(Slice(0,2), k));
            g_vec.push_back(linear_obs_expr);
        }
    }
    // --- Concatenate constraints ---
    MX g_sym = vertcat(g_vec);
    n_constraints_ = g_sym.rows();
    RCLCPP_INFO(logger_, "QP Structure: n_constraints=%d (init=%d, dyn_lin=%d, obs_lin=%d)",
                n_constraints_, n_states_, N * n_states_, max_obstacles_ * N);

    // ==============================================================
    // 5. QP Definition & Solver Creation
    // ==============================================================
    // <<< DEFINE qp_dict HERE >>>
    MXDict qp_dict = {{"x", V_sym_}, {"f", J}, {"g", g_sym}, {"p", P_sym_}};
    // <<< END DEFINE qp_dict >>>

    Dict solver_opts;
    std::string solver_name = "ipopt";
    RCLCPP_INFO(logger_, "Using solver: %s (via nlpsol interface)", solver_name.c_str());

    solver_opts["ipopt.print_level"] = 0;
    solver_opts["ipopt.sb"] = "yes";
    solver_opts["print_time"] = 0;
    solver_opts["ipopt.warm_start_init_point"] = "yes";
    solver_opts["ipopt.tol"] = 1e-3;
    solver_opts["ipopt.acceptable_tol"] = 1e-2;
    solver_opts["ipopt.max_iter"] = 100;

    try {
        solver_func_ = nlpsol("solver", solver_name, qp_dict, solver_opts); // Use defined qp_dict
        RCLCPP_INFO(logger_, "Successfully created IPOPT solver interface via nlpsol.");
    } catch (const std::exception& e) {
         RCLCPP_FATAL(logger_, "Failed to create CasADi solver interface for '%s': %s", solver_name.c_str(), e.what());
         throw std::runtime_error("Failed to create CasADi solver.");
    }

    // ==============================================================
    // 6. Define Fixed Variable Bounds (lbx, ubx)
    // ==============================================================
    x_flat_lower_bounds_.assign(n_opt_vars_, -inf);
    x_flat_upper_bounds_.assign(n_opt_vars_, inf);
    // State angle bounds
    for (int k = 0; k <= N; ++k) {
         int idx = k * n_states_ + 2;
         x_flat_lower_bounds_[idx] = -10*M_PI; x_flat_upper_bounds_[idx] = 10*M_PI;
    }
    // Control bounds
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
    // 7. Define Fixed Constraint Bounds (lba, uba / lbg, ubg)
    // ==============================================================
    g_flat_lower_bounds_.assign(n_constraints_, 0.0); // Default for equalities
    g_flat_upper_bounds_.assign(n_constraints_, 0.0);
    // Obstacle constraints (g = n.x >= rhs) -> lbg=rhs, ubg=inf
    int obs_constraint_start_idx = dynamics_constraints_end_idx;
    for (int i = obs_constraint_start_idx; i < n_constraints_; ++i) {
        g_flat_lower_bounds_[i] = -inf; // Placeholder, set numerically in solveMPC
        g_flat_upper_bounds_[i] = inf;
    }

    // ==============================================================
    // 8. Finalize Setup
    // ==============================================================
    RCLCPP_INFO(logger_, "Setting mpc_problem_defined_ to true...");
    mpc_problem_defined_ = true;
    RCLCPP_INFO(logger_, "CasADi QP problem structure setup complete.");
}

const std::vector<ClusterInfo>& Optimizer::getLastProcessedObstacles() const
{
    return last_processed_obstacles_;
}

bool Optimizer::solveMPC(const std::vector<geometry_msgs::msg::Point> & obstacle_points)
{
    if (!mpc_problem_defined_) {
        RCLCPP_ERROR(logger_, "solveMPC called before setupCasADiProblem!");
        return false;
    }
    // <<< Check if function objects hold valid internal pointers >>>
    if (!solver_func_.get()) { // Check internal pointer
         RCLCPP_ERROR(logger_, "solveMPC called but solver_func_ is not valid!");
         return false;
    }
    if (!dyn_func_.get() || !dyn_jac_x_func_.get() || !dyn_jac_u_func_.get()) { // Check internal pointers
        RCLCPP_ERROR(logger_, "solveMPC called but dynamics/Jacobian functions are not valid!");
        return false;
    }

    auto function_start_time = std::chrono::high_resolution_clock::now();
    int N = settings_.time_steps;

    double current_x = current_state_.pose.pose.position.x;
    double current_y = current_state_.pose.pose.position.y;
    double current_theta = tf2::getYaw(current_state_.pose.pose.orientation);

    // --------------------------------------------------------------
    // <<< STEP 0: Determine Linearization Trajectory >>>
    // --------------------------------------------------------------
    DM X_linearize, U_linearize;
    bool use_prev_solution = last_solve_successful_ && !last_optimal_X_flat_.is_empty() && !last_optimal_U_flat_.is_empty();

    if (use_prev_solution) {
        RCLCPP_DEBUG(logger_, "Using warm start linearization trajectory.");
        // Use static reshape for existing DMs
        X_linearize = DM::reshape(last_optimal_X_flat_, static_cast<casadi_int>(n_states_), static_cast<casadi_int>(N + 1));
        U_linearize = DM::reshape(last_optimal_U_flat_, static_cast<casadi_int>(n_controls_), static_cast<casadi_int>(N));
    } else {
        RCLCPP_WARN(logger_, "No previous solution for linearization, using current state and zero controls.");
        std::vector<double> x_lin_vec(n_states_ * (N+1));
        std::vector<double> u_lin_vec(n_controls_ * N, 0.0);
        double cur_x = current_x;
        double cur_y = current_y;
        double cur_th = current_theta;
        for(int k=0; k<=N; ++k){
             x_lin_vec[k*n_states_ + 0] = cur_x; x_lin_vec[k*n_states_ + 1] = cur_y; x_lin_vec[k*n_states_ + 2] = cur_th;
        }
        DM dm_x_flat = DM(x_lin_vec); DM dm_u_flat = DM(u_lin_vec);
        X_linearize = DM::reshape(dm_x_flat, n_states_, N+1);
        U_linearize = DM::reshape(dm_u_flat, n_controls_, N);
    }

    // --------------------------------------------------------------
    // <<< STEP 1: Compute Numerical Linearization Parameters & Prepare RHS Vec >>>
    // --------------------------------------------------------------
    auto lin_param_start_time = std::chrono::high_resolution_clock::now();
    // ** Intermediate vectors removed **
    std::vector<double> obs_rhs_computed_vec; // Still need this for lbg
    obs_rhs_computed_vec.reserve(max_obstacles_ * N);

    // --- Prepare Parameter Vector Directly ---
    std::vector<double> p_vec;
    p_vec.resize(n_params_); // <<< RESIZE UPFRONT >>>
    int p_idx = 0; // Index for filling p_vec

    // -- Fill Initial State --
    p_vec[p_idx++] = current_x;
    p_vec[p_idx++] = current_y;
    p_vec[p_idx++] = current_theta;

    // -- Fill Reference Trajectory --
    std::vector<geometry_msgs::msg::Pose> ref_poses = getReferencePoseHorizon(current_state_.pose.pose, N, settings_.model_dt);
    if (ref_poses.size() != static_cast<size_t>(N + 1)) {
        RCLCPP_ERROR(logger_, "Reference trajectory generation failed.");
        return false;
    }
    for (const auto &pose : ref_poses) {
        if (p_idx + 3 > n_params_) { RCLCPP_ERROR(logger_, "Parameter index out of bounds filling ref traj"); return false; }
        p_vec[p_idx++] = pose.position.x;
        p_vec[p_idx++] = pose.position.y;
        p_vec[p_idx++] = tf2::getYaw(pose.orientation);
    }

    // -- Fill Last Control Command --
    if (p_idx + 3 > n_params_) { RCLCPP_ERROR(logger_, "Parameter index out of bounds filling last ctrl"); return false; }
    p_vec[p_idx++] = last_v_.scalar();
    p_vec[p_idx++] = last_vy_.scalar();
    p_vec[p_idx++] = last_w_.scalar();

    // -- Compute and Fill Linearized Dynamics Parameters --
    int dyn_lin_start_idx = p_idx; // Store start index for verification if needed
    for (int k=0; k<N; ++k) {
        DM x_lin_k = X_linearize(Slice(), k);
        DM u_lin_k = U_linearize(Slice(), k);
        DM A_k_num = dyn_jac_x_func_(DMVector{x_lin_k, u_lin_k})[0];
        DM B_k_num = dyn_jac_u_func_(DMVector{x_lin_k, u_lin_k})[0];
        DM f_eval_num = dyn_func_(DMVector{x_lin_k, u_lin_k})[0];
        DM c_k_num = f_eval_num - mtimes(A_k_num, x_lin_k) - mtimes(B_k_num, u_lin_k);

        std::vector<double> A_k_elem = A_k_num.get_elements();
        std::vector<double> B_k_elem = B_k_num.get_elements();
        std::vector<double> c_k_elem = c_k_num.get_elements();

        if (p_idx + A_k_elem.size() > static_cast<size_t>(n_params_)) { RCLCPP_ERROR(logger_, "Parameter index OOB filling A_k %d", k); return false; }
        std::copy(A_k_elem.begin(), A_k_elem.end(), p_vec.begin() + p_idx); p_idx += A_k_elem.size();

        if (p_idx + B_k_elem.size() > static_cast<size_t>(n_params_)) { RCLCPP_ERROR(logger_, "Parameter index OOB filling B_k %d", k); return false; }
        std::copy(B_k_elem.begin(), B_k_elem.end(), p_vec.begin() + p_idx); p_idx += B_k_elem.size();

        if (p_idx + c_k_elem.size() > static_cast<size_t>(n_params_)) { RCLCPP_ERROR(logger_, "Parameter index OOB filling c_k %d", k); return false; }
        std::copy(c_k_elem.begin(), c_k_elem.end(), p_vec.begin() + p_idx); p_idx += c_k_elem.size();
    }
    // Verification log
    if (p_idx != dyn_lin_start_idx + n_dyn_lin_params_) {
        RCLCPP_ERROR(logger_, "Dynamics parameter fill mismatch! Expected end index %d, got %d", dyn_lin_start_idx + n_dyn_lin_params_, p_idx); return false;
    }

    // -- Compute and Fill Linearized Obstacle Parameters AND RHS vector --
    int obs_lin_start_idx = p_idx; // Store start index
    std::vector<ClusterInfo> current_obstacles = processObstacles(obstacle_points);
    last_processed_obstacles_ = current_obstacles;
    size_t num_detected_obstacles = last_processed_obstacles_.size();
    RCLCPP_DEBUG(logger_, "Detected %zu obstacle clusters (QP linearization).", num_detected_obstacles);

    double dummy_nx = 0.0; double dummy_ny = 0.0; double dummy_rhs = -inf;
    double R_safe_base = robot_radius_ + lidar_safety_margin_;
    obs_rhs_computed_vec.clear(); // Clear before filling
    obs_rhs_computed_vec.reserve(max_obstacles_ * N);

    for (int j=0; j < max_obstacles_; ++j) {
        bool is_real = (static_cast<size_t>(j) < num_detected_obstacles);
        double cx_j = 1e6, cy_j = 1e6, r_j = 0.0;
        if (is_real) {
             cx_j = last_processed_obstacles_[j].cx; cy_j = last_processed_obstacles_[j].cy;
             if(last_processed_obstacles_[j].radius_sq >= 0) { r_j = sqrt(last_processed_obstacles_[j].radius_sq); }
        }
        double R_safe_j = R_safe_base + r_j;

        for (int k=1; k<=N; ++k) {
            double nx_jk = dummy_nx; double ny_jk = dummy_ny; double rhs_jk = dummy_rhs;
            if (is_real) {
                DM x_lin_k = X_linearize(Slice(0,2), k);
                double dx = x_lin_k(0).scalar() - cx_j; double dy = x_lin_k(1).scalar() - cy_j;
                double dist = sqrt(dx*dx + dy*dy);
                nx_jk = (dist > 1e-6) ? dx/dist : 1.0; ny_jk = (dist > 1e-6) ? dy/dist : 0.0;
                rhs_jk = nx_jk * cx_j + ny_jk * cy_j + R_safe_j;
            }
            // Fill p_vec directly
            if (p_idx + 3 > n_params_) { RCLCPP_ERROR(logger_, "Parameter index OOB filling obs params j=%d, k=%d", j, k); return false; }
            p_vec[p_idx++] = nx_jk;
            p_vec[p_idx++] = ny_jk;
            p_vec[p_idx++] = rhs_jk; // Parameter part
            // Store computed RHS separately for lbg
            obs_rhs_computed_vec.push_back(rhs_jk);
        }
    }
    // Verification log
    if (p_idx != obs_lin_start_idx + n_obs_lin_params_) {
        RCLCPP_ERROR(logger_, "Obstacle parameter fill mismatch! Expected end index %d, got %d", obs_lin_start_idx + n_obs_lin_params_, p_idx); return false;
    }
    auto lin_param_end_time = std::chrono::high_resolution_clock::now();
    auto lin_param_duration = std::chrono::duration_cast<std::chrono::microseconds>(lin_param_end_time - lin_param_start_time);

    // --------------------------------------------------------------
    // 2. Prepare Parameters DM (p) - This section is now simpler
    // --------------------------------------------------------------
    auto param_prep_start_time = std::chrono::high_resolution_clock::now();
    // Final size check (redundant if above checks pass, but safe)
    if (p_vec.size() != static_cast<size_t>(n_params_)) {
         RCLCPP_ERROR(logger_, "QP Parameter vector final size mismatch! Expected %d, got %zu", n_params_, p_vec.size());
        return false;
    }
    DM p = DM(p_vec);
    auto param_prep_end_time = std::chrono::high_resolution_clock::now();
    auto param_prep_duration = std::chrono::duration_cast<std::chrono::microseconds>(param_prep_end_time - param_prep_start_time);


    // --------------------------------------------------------------
    // 3. Prepare Initial Guess (x0_guess) - NO CHANGE NEEDED
    // --------------------------------------------------------------
    auto guess_prep_start_time = std::chrono::high_resolution_clock::now();
    DM x0_guess;
    if (use_prev_solution) {
        RCLCPP_DEBUG(logger_, "Using warm start initial guess.");
        DM X_prev = DM::reshape(last_optimal_X_flat_, static_cast<casadi_int>(n_states_), static_cast<casadi_int>(N + 1));
        DM U_prev = DM::reshape(last_optimal_U_flat_, static_cast<casadi_int>(n_controls_), static_cast<casadi_int>(N));
        DM X_guess = horzcat(X_prev(Slice(), Slice(1, N + 1)), X_prev(Slice(), N));
        DM U_guess = horzcat(U_prev(Slice(), Slice(1, N)), U_prev(Slice(), N-1));
        x0_guess = vertcat(DM::reshape(X_guess, n_states_ * (N + 1), 1),
                           DM::reshape(U_guess, n_controls_ * N, 1));
    } else {
        RCLCPP_DEBUG(logger_, "Using cold start initial guess.");
        std::vector<double> x0_vec(n_opt_vars_, 0.0);
        for (int k = 0; k <= N; ++k) {
            x0_vec[k * n_states_ + 0] = current_x; x0_vec[k * n_states_ + 1] = current_y; x0_vec[k * n_states_ + 2] = current_theta;
        }
        x0_guess = DM(x0_vec);
    }
    auto guess_prep_end_time = std::chrono::high_resolution_clock::now();
    auto guess_prep_duration = std::chrono::duration_cast<std::chrono::microseconds>(guess_prep_end_time - guess_prep_start_time);


    // --------------------------------------------------------------
    // 4. Prepare Bounds (lbx, ubx, lbg, ubg) - NO CHANGE NEEDED
    //    (Uses obs_rhs_computed_vec filled above)
    // --------------------------------------------------------------
    auto bounds_prep_start_time = std::chrono::high_resolution_clock::now();
    DM lbx = DM(x_flat_lower_bounds_);
    DM ubx = DM(x_flat_upper_bounds_);
    DM lbg = DM(g_flat_lower_bounds_);
    DM ubg = DM(g_flat_upper_bounds_);

    // Update initial state bounds
    for (int i = 0; i < n_states_; ++i) {
        lbx(i) = p_vec[i]; ubx(i) = p_vec[i];
        lbg(i) = 0.0; ubg(i) = 0.0;
    }

    // Update obstacle constraint bounds
    int dynamics_constraints_end_idx = n_states_ + N * n_states_;
    int current_g_idx = dynamics_constraints_end_idx;
    // Size check for RHS vector
    if (obs_rhs_computed_vec.size() != static_cast<size_t>(max_obstacles_ * N)) {
        RCLCPP_ERROR(logger_, "Computed RHS vector size mismatch! Expected %d, got %zu",
                     max_obstacles_ * N, obs_rhs_computed_vec.size());
        return false;
    }
    int current_rhs_idx = 0;
    for (int j = 0; j < max_obstacles_; ++j) {
        for (int k = 1; k <= N; ++k) {
            if (current_g_idx >= n_constraints_) {
                 RCLCPP_ERROR(logger_, "QP Constraint index g[%d] out of bounds (max %d) for obstacle %d, step %d",
                              current_g_idx, n_constraints_-1, j, k);
                 return false;
            }
            lbg(current_g_idx) = obs_rhs_computed_vec[current_rhs_idx];
            current_g_idx++;
            current_rhs_idx++;
        }
    }
    auto bounds_prep_end_time = std::chrono::high_resolution_clock::now();
    auto bounds_prep_duration = std::chrono::duration_cast<std::chrono::microseconds>(bounds_prep_end_time - bounds_prep_start_time);


    // --------------------------------------------------------------
    // 5. Call Solver - NO CHANGE NEEDED
    // --------------------------------------------------------------
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

    // --------------------------------------------------------------
    // 6. Process Results - NO CHANGE NEEDED
    // --------------------------------------------------------------
    bool success = solver_func_.stats().at("success");
    if (success) {
        DM sol = res.at("x");
        last_optimal_X_flat_ = sol(Slice(0, n_states_ * (N + 1)));
        last_optimal_U_flat_ = sol(Slice(n_states_ * (N + 1), n_opt_vars_));
        last_solve_successful_ = true;
        RCLCPP_DEBUG(logger_, "Solve successful. Final cost: %f", static_cast<double>(res.at("f")));
    } else {
        RCLCPP_WARN(logger_, "QP/NLP solver did not find an optimal solution. Status: %s",
                   static_cast<std::string>(solver_func_.stats().at("return_status")).c_str());
        last_solve_successful_ = false;
        last_optimal_X_flat_ = casadi::DM();
        last_optimal_U_flat_ = casadi::DM();
    }

    auto function_end_time = std::chrono::high_resolution_clock::now();
    auto function_duration = std::chrono::duration_cast<std::chrono::microseconds>(function_end_time - function_start_time);
    RCLCPP_DEBUG(logger_, "solveMPC total time: %.3f ms LinParam+Param: %.3f, Guess: %.3f, Bounds: %.3f, Solve: %.3f)", // Updated timing label
                function_duration.count() / 1000.0,
                lin_param_duration.count() / 1000.0, // Renamed/combined timing
                guess_prep_duration.count() / 1000.0,
                bounds_prep_duration.count() / 1000.0,
                solve_duration.count() / 1000.0);

    return success;
}

std::vector<geometry_msgs::msg::Pose> Optimizer::getReferencePoseHorizon(const geometry_msgs::msg::Pose& start_robot_pose, int N, double T)
{
    std::vector<geometry_msgs::msg::Pose> ref_poses;
    ref_poses.reserve(N + 1);

    const auto& global_plan = path_; // Using the xtensor path_ member
    if (global_plan.x.shape(0) < 2) {
        RCLCPP_WARN_THROTTLE(logger_, *clock_, 2000, "Global plan too short (%zu points) for reference generation. Using start pose.", global_plan.x.shape(0));
        for (int i=0; i <= N; ++i) {
            ref_poses.push_back(start_robot_pose);
        }
        return ref_poses;
    }

    // Find the closest point on the path to the current robot pose
    double min_dist_sq = std::numeric_limits<double>::max();
    size_t closest_idx = 0;
    double current_x = start_robot_pose.position.x;
    double current_y = start_robot_pose.position.y;

    for (size_t i = 0; i < global_plan.x.shape(0); ++i) {
        double dx = global_plan.x(i) - current_x;
        double dy = global_plan.y(i) - current_y;
        double dist_sq = dx * dx + dy * dy;
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            closest_idx = i;
        }
    }

    // Estimate a desired speed along the path
    double desired_speed = settings_.constraints.vx_max * 0.8;
    desired_speed = std::max(0.1, desired_speed);

    // Calculate cumulative distances along the path
    std::vector<double> cumulative_dist;
    cumulative_dist.push_back(0.0);
    double total_path_dist = 0;
    for (size_t i = closest_idx; i < global_plan.x.shape(0) - 1; ++i) {
         double dx = global_plan.x(i + 1) - global_plan.x(i);
         double dy = global_plan.y(i + 1) - global_plan.y(i);
         total_path_dist += std::sqrt(dx*dx + dy*dy);
         cumulative_dist.push_back(total_path_dist);
    }

    // Generate reference poses
    size_t current_segment_idx = 0;
    for (int k = 0; k <= N; ++k) {
        double target_dist_k = k * desired_speed * T;
        while (current_segment_idx < cumulative_dist.size() - 1 &&
               cumulative_dist[current_segment_idx + 1] < target_dist_k)
        {
            current_segment_idx++;
        }

        geometry_msgs::msg::Pose ref_pose_k;
        size_t global_idx0 = closest_idx + current_segment_idx;
        size_t global_idx1 = closest_idx + current_segment_idx + 1;

        if (global_idx1 >= global_plan.x.shape(0)) {
             // Clamp to last pose
             size_t last_idx = global_plan.x.shape(0)-1;
             ref_pose_k.position.x = global_plan.x(last_idx);
             ref_pose_k.position.y = global_plan.y(last_idx);
             tf2::Quaternion q;
             if (last_idx < global_plan.yaws.shape(0)) {
                 q.setRPY(0, 0, global_plan.yaws(last_idx));
             } else {
                 if (last_idx > 0) {
                     double dx = global_plan.x(last_idx) - global_plan.x(last_idx - 1);
                     double dy = global_plan.y(last_idx) - global_plan.y(last_idx - 1);
                     q.setRPY(0, 0, std::atan2(dy, dx));
                 } else { q.setRPY(0, 0, 0); }
             }
             ref_pose_k.orientation = tf2::toMsg(q);
        } else {
             // Interpolate
             double segment_len = cumulative_dist[current_segment_idx + 1] - cumulative_dist[current_segment_idx];
             double dist_into_segment = target_dist_k - cumulative_dist[current_segment_idx];
             double ratio = (segment_len > 1e-6) ? (dist_into_segment / segment_len) : 0.0;
             ratio = std::max(0.0, std::min(1.0, ratio));

             double x0 = global_plan.x(global_idx0);
             double y0 = global_plan.y(global_idx0);
             double yaw0 = 0.0;
             if (global_idx0 < global_plan.yaws.shape(0)) yaw0 = global_plan.yaws(global_idx0);
             else if (global_idx0 > 0) yaw0 = atan2(y0-global_plan.y(global_idx0-1), x0-global_plan.x(global_idx0-1));

             double x1 = global_plan.x(global_idx1);
             double y1 = global_plan.y(global_idx1);
             double yaw1 = 0.0;
             if (global_idx1 < global_plan.yaws.shape(0)) yaw1 = global_plan.yaws(global_idx1);
             else yaw1 = atan2(y1-y0, x1-x0);

             ref_pose_k.position.x = x0 + ratio * (x1 - x0);
             ref_pose_k.position.y = y0 + ratio * (y1 - y0);
             double dyaw = yaw1 - yaw0;
             dyaw = atan2(sin(dyaw), cos(dyaw));
             double interpolated_yaw = yaw0 + ratio * dyaw;
             tf2::Quaternion q_interp;
             q_interp.setRPY(0, 0, interpolated_yaw);
             ref_pose_k.orientation = tf2::toMsg(q_interp);
        }
        ref_poses.push_back(ref_pose_k);
    }
    RCLCPP_DEBUG(logger_, "Generated reference horizon with %zu poses.", ref_poses.size());
    return ref_poses;
}

std::vector<std::vector<size_t>> Optimizer::clusterPoints(
    const std::vector<geometry_msgs::msg::Point>& points)
{
    std::vector<std::vector<size_t>> clusters;
    if (points.empty()) {
        return clusters;
    }

    size_t n_points = points.size();
    std::vector<bool> visited(n_points, false);
    // Use the squared tolerance for efficiency
    double tolerance_sq = lidar_cluster_tolerance_ * lidar_cluster_tolerance_;

    for (size_t i = 0; i < n_points; ++i) {
        if (!visited[i]) {
            std::vector<size_t> current_cluster_indices;
            std::vector<size_t> q; // Queue for BFS-like search
            q.push_back(i);
            visited[i] = true;

            size_t head = 0;
            while(head < q.size()) {
                size_t current_idx = q[head++]; // Process point from queue
                current_cluster_indices.push_back(current_idx);
                const auto& p1 = points[current_idx];

                // Find neighbors of the current point
                for (size_t j = 0; j < n_points; ++j) {
                    // Check only unvisited points
                    if (!visited[j]) {
                        const auto& p2 = points[j];
                        double dx = p1.x - p2.x;
                        double dy = p1.y - p2.y;
                        // If within squared tolerance, add to queue and mark visited
                        if ((dx * dx + dy * dy) < tolerance_sq) {
                            visited[j] = true;
                            q.push_back(j);
                        }
                    }
                }
            }
            // Store the cluster if it meets the minimum size requirement
            if (current_cluster_indices.size() >= static_cast<size_t>(lidar_min_cluster_size_)) {
                clusters.push_back(current_cluster_indices);
            }
        }
    }
    return clusters; // Return vector of clusters (each cluster is a vector of indices)
}

// Computes centroid and max squared radius from centroid for a cluster
ClusterInfo Optimizer::computeBoundingCircle(
    const std::vector<geometry_msgs::msg::Point>& cluster_points)
{
    ClusterInfo info; // Create struct to hold results
    if (cluster_points.empty()) {
        return info; // Return empty info if cluster is empty
    }

    info.num_points = cluster_points.size();

    // Calculate centroid (average position)
    double sum_x = 0.0, sum_y = 0.0;
    for (const auto& pt : cluster_points) {
        sum_x += pt.x;
        sum_y += pt.y;
    }
    info.cx = sum_x / info.num_points;
    info.cy = sum_y / info.num_points;

    // Find max squared distance from centroid to any point in the cluster
    double max_dist_sq = 0.0;
    for (const auto& pt : cluster_points) {
        double dx = pt.x - info.cx;
        double dy = pt.y - info.cy;
        max_dist_sq = std::max(max_dist_sq, dx * dx + dy * dy);
    }
    info.radius_sq = max_dist_sq; // Store the squared radius

    return info; // Return the computed cluster information
}

// Main function to process raw lidar points into obstacle info for MPC
std::vector<ClusterInfo> Optimizer::processObstacles(
    const std::vector<geometry_msgs::msg::Point>& points)
{
    std::vector<ClusterInfo> obstacle_info_list; // List to store results

    if (points.empty()) {
        return obstacle_info_list; // Return empty list if no points
    }

    // 1. Cluster the raw lidar points
    std::vector<std::vector<size_t>> cluster_indices = clusterPoints(points);
    RCLCPP_DEBUG(logger_, "Clustering found %zu potential clusters.", cluster_indices.size());


    // 2. Compute bounding circle for each valid cluster
    obstacle_info_list.reserve(cluster_indices.size());
    for (const auto& indices : cluster_indices) {
        // Create a temporary vector containing only the points for this cluster
        std::vector<geometry_msgs::msg::Point> current_cluster_points;
        current_cluster_points.reserve(indices.size());
        for (size_t idx : indices) {
            // Ensure index is valid (safety check)
            if (idx < points.size()) {
                 current_cluster_points.push_back(points[idx]);
            } else {
                 RCLCPP_ERROR(logger_, "Invalid index %zu found during clustering!", idx);
            }
        }

        // Compute bounding circle only if points were added
        if (!current_cluster_points.empty()) {
            ClusterInfo info = computeBoundingCircle(current_cluster_points);
            // Add the computed info to our list if it's valid (has points)
            if (info.num_points > 0) {
                obstacle_info_list.push_back(info);
            }
        }
    }

    // Optional: Sort obstacles here if needed (e.g., by distance to current_state_.pose)
    // std::sort(obstacle_info_list.begin(), obstacle_info_list.end(),
    //           [this](const ClusterInfo& a, const ClusterInfo& b) {
    //               double dist_sq_a = std::pow(a.cx - current_state_.pose.pose.position.x, 2) + std::pow(a.cy - current_state_.pose.pose.position.y, 2);
    //               double dist_sq_b = std::pow(b.cx - current_state_.pose.pose.position.x, 2) + std::pow(b.cy - current_state_.pose.pose.position.y, 2);
    //               return dist_sq_a < dist_sq_b; // Sort by closest first
    //           });


    // 3. Limit the number of obstacles passed to the solver
    if (obstacle_info_list.size() > static_cast<size_t>(max_obstacles_)) {
        RCLCPP_WARN_THROTTLE(logger_, *clock_, 5000, // Use member clock_
            "Detected %zu obstacles, but limiting to %d for constraints. Closest ones used if sorted.",
            obstacle_info_list.size(), max_obstacles_);
        // Resize the list to keep only the first 'max_obstacles_' (which are the closest if sorted)
        obstacle_info_list.resize(max_obstacles_);
    }

    return obstacle_info_list; // Return the final list of obstacles for MPC
}

} // namespace sortham