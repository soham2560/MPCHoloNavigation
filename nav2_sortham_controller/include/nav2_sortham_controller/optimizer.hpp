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

// C++ Standard Library
#include <string>
#include <vector>
#include <memory>
#include <limits>
#include <mutex>
#include <optional>

// Third-Party
#include <xtensor/xtensor.hpp>
#include <casadi/casadi.hpp>

// ROS 2
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"

// ROS 2 Messages
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "nav_msgs/msg/path.hpp"

// Navigation Stack
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav2_core/goal_checker.hpp"

// SORTHAM Controller
#include "nav2_sortham_controller/models/optimizer_settings.hpp"
#include "nav2_sortham_controller/motion_models.hpp"
#include "nav2_sortham_controller/models/path.hpp"
#include "nav2_sortham_controller/tools/parameters_handler.hpp"
#include "nav2_sortham_controller/tools/utils.hpp"

namespace sortham
{

// ============================================================================
// Supporting Data Structures
// ============================================================================

struct ClusterInfo
{
  double cx = 0.0;
  double cy = 0.0;
  double radius_sq = 0.0;
  size_t num_points = 0;
};

struct MpcCurrentState
{
  geometry_msgs::msg::PoseStamped pose;
  geometry_msgs::msg::Twist speed;
};

// ============================================================================
// Optimizer Class Declaration
// ============================================================================

class Optimizer
{
public:
  // --------------------
  // Public Methods
  // --------------------
  Optimizer() = default;

  void initialize(
    rclcpp_lifecycle::LifecycleNode::WeakPtr parent,
    const std::string & name,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros,
    ParametersHandler * dynamic_parameters_handler);

  void shutdown();

  geometry_msgs::msg::TwistStamped evalControl(
    const geometry_msgs::msg::PoseStamped & robot_pose,
    const geometry_msgs::msg::Twist & robot_speed,
    const nav_msgs::msg::Path & plan,
    const geometry_msgs::msg::Pose & goal,
    const std::vector<geometry_msgs::msg::Point> & obstacle_points,
    nav2_core::GoalChecker * goal_checker);

  xt::xtensor<float, 2> getOptimizedTrajectory();

  void setSpeedLimit(double speed_limit, bool percentage);

  void reset();

  const std::vector<ClusterInfo>& getLastProcessedObstacles() const;

  double getRobotRadius() const { return robot_radius_; }

  double getLidarSafetyMargin() const { return lidar_safety_margin_; }

  std::optional<geometry_msgs::msg::Pose> getLastLocalTarget() const;

  void generateMilestones(const nav_msgs::msg::Path & path);

  geometry_msgs::msg::Pose selectCurrentMilestoneTarget();

  double calculatePathCurvature(const nav_msgs::msg::Path& path, size_t index);

  const std::vector<geometry_msgs::msg::Pose>& getCurrentMilestones() const
  {
    return current_milestones_;
  }

  size_t getCurrentMilestoneIndex() const
  {
    return current_milestone_idx_;
  }

protected:
  // --------------------
  // MPC and Optimization Logic
  // --------------------
  void setupCasADiProblem();
  bool solveMPC(const std::vector<geometry_msgs::msg::Point> & obstacle_points);
  
  // --------------------
  // Obstacle Processing
  // --------------------
  std::vector<ClusterInfo> processObstacles(const std::vector<geometry_msgs::msg::Point>& points);
  std::vector<std::vector<size_t>> clusterPoints(const std::vector<geometry_msgs::msg::Point>& points);
  std::vector<std::vector<geometry_msgs::msg::Point>> clusterPointsInternal(
    const std::vector<geometry_msgs::msg::Point>& points_to_cluster,
    double tolerance, int min_size);
  ClusterInfo computeBoundingCircle(const std::vector<geometry_msgs::msg::Point>& cluster_points);

  // --------------------
  // Path & Milestone Management
  // --------------------
  std::vector<geometry_msgs::msg::Pose> generateReferenceHorizon(const geometry_msgs::msg::Pose& lookahead_pose);

  // --------------------
  // Configuration
  // --------------------
  void getParams();
  void setMotionModel(const std::string & model);
  bool isHolonomic() const;

private:
  // --------------------
  // ROS and Configuration
  // --------------------
  rclcpp_lifecycle::LifecycleNode::WeakPtr parent_;
  rclcpp::Clock::SharedPtr clock_;
  rclcpp::Logger logger_{rclcpp::get_logger("SORTHAMOptimizer")};
  std::string name_;
  ParametersHandler * parameters_handler_{nullptr};

  // --------------------
  // Costmap and Path
  // --------------------
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  nav2_costmap_2d::Costmap2D * costmap_{nullptr};
  models::Path path_;
  geometry_msgs::msg::Pose goal_;

  // --------------------
  // Motion Model
  // --------------------
  std::shared_ptr<MotionModel> motion_model_;
  MpcCurrentState current_state_;
  models::OptimizerSettings settings_;

  // --------------------
  // CasADi MPC Variables
  // --------------------
  casadi::Function solver_func_;
  casadi::Function dyn_func_;

  casadi::MX V_sym_;
  casadi::MX P_sym_;

  casadi::DM last_optimal_X_flat_;
  casadi::DM last_optimal_U_flat_;
  casadi::DM last_v_{0.0};
  casadi::DM last_w_{0.0};
  casadi::DM last_vy_{0.0};

  // --------------------
  // MPC Problem Dimensions
  // --------------------
  int n_states_;
  int n_controls_;
  int n_params_;
  int n_opt_vars_;
  int n_constraints_;

  int n_obstacle_params_per_obs_ = 3;
  int n_init_state_params_;
  int n_local_target_params_;
  int n_last_control_params_;
  int n_obstacle_params_;
  int obs_constraint_start_idx_;

  // --------------------
  // Solver Constraints
  // --------------------
  std::vector<double> x_flat_lower_bounds_;
  std::vector<double> x_flat_upper_bounds_;
  std::vector<double> g_flat_lower_bounds_;
  std::vector<double> g_flat_upper_bounds_;

  // --------------------
  // Obstacle Parameters
  // --------------------
  std::vector<ClusterInfo> last_processed_obstacles_;
  double robot_radius_{0.25};
  double lidar_safety_margin_{0.2};
  double lidar_cluster_tolerance_{0.2};
  int lidar_min_cluster_size_{3};
  int max_obstacles_{10};
  double max_cluster_radius_{1.0};

  // --------------------
  // Lookahead and Milestone
  // --------------------
  double lookahead_point_dist_{1.0};
  double min_lookahead_point_dist_{0.3};
  std::optional<geometry_msgs::msg::Pose> last_local_target_;
  std::vector<geometry_msgs::msg::Pose> current_milestones_;
  size_t current_milestone_idx_;
  double milestone_reached_threshold_;
  double milestone_min_spacing_;
  double milestone_max_spacing_;
  double milestone_curvature_factor_;
  std::string last_path_metadata_;
  std::optional<geometry_msgs::msg::Pose> current_milestones_goal_;

  // --------------------
  // State
  // --------------------
  bool mpc_problem_defined_ = false;
  bool last_solve_successful_ = false;
};

}  // namespace sortham

#endif  // NAV2_SORTHAM_CONTROLLER__OPTIMIZER_HPP_
