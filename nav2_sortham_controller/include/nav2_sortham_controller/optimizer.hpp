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

#ifndef NAV2_SORTHAM_CONTROLLER__OPTIMIZER_HPP_
#define NAV2_SORTHAM_CONTROLLER__OPTIMIZER_HPP_

#include <string>
#include <vector>
#include <memory>
#include <limits>
#include <mutex>

#include <xtensor/xtensor.hpp>

#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "rclcpp/rclcpp.hpp"

#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav2_core/goal_checker.hpp"

#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/point.hpp"

#include "nav2_sortham_controller/models/optimizer_settings.hpp"
#include "nav2_sortham_controller/motion_models.hpp"
#include "nav2_sortham_controller/models/path.hpp"
#include "nav2_sortham_controller/tools/parameters_handler.hpp"
#include "nav2_sortham_controller/tools/utils.hpp"

#include <casadi/casadi.hpp>

namespace sortham
{
struct ClusterInfo {
  double cx = 0.0;
  double cy = 0.0;
  double radius_sq = 0.0;
  size_t num_points = 0;
};

struct MpcCurrentState {
    geometry_msgs::msg::PoseStamped pose;
    geometry_msgs::msg::Twist speed;
};

/**
 * @class sortham::Optimizer
 * @brief Main algorithm optimizer of the SORTHAM Controller
 */
class Optimizer
{
public:
  /**
    * @brief Constructor for sortham::Optimizer
    */
  Optimizer() = default;

  /**
   * @brief Initializes optimizer on startup
   * @param parent WeakPtr to node
   * @param name Name of plugin
   * @param costmap_ros Costmap2DROS object of environment
   * @param dynamic_parameter_handler Parameter handler object
   */
  void initialize(
    rclcpp_lifecycle::LifecycleNode::WeakPtr parent, const std::string & name,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros,
    ParametersHandler * dynamic_parameters_handler);

  /**
   * @brief Shutdown for optimizer at process end
   */
  void shutdown();

  /**
   * @brief Compute control using SORTHAM algorithm
   * @param robot_pose Pose of the robot at given time
   * @param robot_speed Speed of the robot at given time
   * @param plan Path plan to track
   * @param goal_checker Object to check if goal is completed
   * @return TwistStamped of the SORTHAM control
   */
  geometry_msgs::msg::TwistStamped evalControl(
    const geometry_msgs::msg::PoseStamped & robot_pose,
    const geometry_msgs::msg::Twist & robot_speed, const nav_msgs::msg::Path & plan,
    const geometry_msgs::msg::Pose & goal,
    const std::vector<geometry_msgs::msg::Point> & obstacle_points,
    nav2_core::GoalChecker * goal_checker);

  /**
   * @brief Get the optimal trajectory for a cycle for visualization
   * @return Optimal trajectory
   */
  xt::xtensor<float, 2> getOptimizedTrajectory();

  /**
   * @brief Set the maximum speed based on the speed limits callback
   * @param speed_limit Limit of the speed for use
   * @param percentage Whether the speed limit is absolute or relative
   */
  void setSpeedLimit(double speed_limit, bool percentage);

  /**
   * @brief Reset the optimization problem to initial conditions
   */
  void reset();

  /**
   * @brief Get last processed obstacles
   */
  const std::vector<ClusterInfo>& getLastProcessedObstacles() const;
  double getRobotRadius() const { return robot_radius_; }
  double getLidarSafetyMargin() const { return lidar_safety_margin_; }

protected:
  rclcpp::Clock::SharedPtr clock_;
  // --- Core MPC Methods ---
  void setupCasADiProblem();
  bool solveMPC(const std::vector<geometry_msgs::msg::Point> & obstacle_points);
  std::vector<geometry_msgs::msg::Pose> getReferencePoseHorizon(
        const geometry_msgs::msg::Pose& current_robot_pose, int N, double dt);

  // --- Obstacle Processing Helpers ---
  std::vector<ClusterInfo> processObstacles(
      const std::vector<geometry_msgs::msg::Point>& points);
  std::vector<std::vector<size_t>> clusterPoints(
      const std::vector<geometry_msgs::msg::Point>& points);
  ClusterInfo computeBoundingCircle(
      const std::vector<geometry_msgs::msg::Point>& cluster_points);
  std::vector<ClusterInfo> last_processed_obstacles_;

  // --- Utility Methods ---
  void getParams();
  void setMotionModel(const std::string & model);
  bool isHolonomic() const;

  // --- Core Members ---
  rclcpp_lifecycle::LifecycleNode::WeakPtr parent_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  nav2_costmap_2d::Costmap2D * costmap_{nullptr};
  std::string name_;
  ParametersHandler * parameters_handler_{nullptr};
  rclcpp::Logger logger_{rclcpp::get_logger("SORTHAMOptimizer")};

  // --- Settings ---
  models::OptimizerSettings settings_;
  std::shared_ptr<MotionModel> motion_model_;

  // --- State/Plan Info used by MPC ---
  MpcCurrentState current_state_;
  models::Path path_;
  geometry_msgs::msg::Pose goal_;

  // --- CasADi Members ---
  casadi::Function solver_func_;
  casadi::MX V_sym_; // Combined state and control vector (decision variables)
  casadi::MX P_sym_; // Parameter vector
  int n_states_;
  int n_controls_;
  int n_params_;
  int n_opt_vars_;
  int n_constraints_;
  int n_obstacle_params_per_obs_ = 3; // cx, cy, r^2
  int n_init_state_params_;
  int n_ref_params_;
  int n_last_control_params_;
  int n_obstacle_params_;
  int n_dyn_lin_params_;
  int n_obs_lin_params_;

  casadi::DM last_optimal_X_flat_;
  casadi::DM last_optimal_U_flat_;
  casadi::DM last_v_{0.0};
  casadi::DM last_w_{0.0};
  casadi::DM last_vy_{0.0};
  casadi::Function dyn_func_;
  casadi::Function dyn_jac_x_func_;
  casadi::Function dyn_jac_u_func_;

  bool mpc_problem_defined_ = false;
  bool last_solve_successful_ = false;

  // --- Precomputed Bounds ---
  std::vector<double> x_flat_lower_bounds_;
  std::vector<double> x_flat_upper_bounds_;
  std::vector<double> g_flat_lower_bounds_;
  std::vector<double> g_flat_upper_bounds_;

  // --- Obstacle Avoidance Parameters ---
  double robot_radius_{0.25};
  double lidar_safety_margin_{0.05};
  double lidar_cluster_tolerance_{0.2};
  int lidar_min_cluster_size_{3};
  int max_obstacles_{10};
  double effective_radius_padding_sq_{0.0};
};

}  // namespace sortham

#endif  // NAV2_SORTHAM_CONTROLLER__OPTIMIZER_HPP_
