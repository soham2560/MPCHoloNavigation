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

#ifndef NAV2_SORTHAM_CONTROLLER__CONTROLLER_HPP_
#define NAV2_SORTHAM_CONTROLLER__CONTROLLER_HPP_

#include <string>
#include <memory>
#include <vector>
#include <mutex>

#include "sensor_msgs/msg/laser_scan.hpp"
#include "geometry_msgs/msg/point.hpp"

#include "nav2_sortham_controller/tools/path_handler.hpp"
#include "nav2_sortham_controller/optimizer.hpp"
#include "nav2_sortham_controller/tools/trajectory_visualizer.hpp"
#include "nav2_sortham_controller/models/constraints.hpp"
#include "nav2_sortham_controller/tools/utils.hpp"

#include "nav2_core/controller.hpp"
#include "nav2_core/goal_checker.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2/utils.h"

namespace nav2_sortham_controller
{

using namespace sortham;  // NOLINT

/**
 * @class sortham::SORTHAMController
 * @brief Main plugin controller for SORTHAM Controller
 */
class SORTHAMController : public nav2_core::Controller
{
public:
  /**
    * @brief Constructor for sortham::SORTHAMController
    */
  SORTHAMController() = default;

  /**
    * @brief Configure controller on bringup
    * @param parent WeakPtr to node
    * @param name Name of plugin
    * @param tf TF buffer to use
    * @param costmap_ros Costmap2DROS object of environment
    */
  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name, const std::shared_ptr<tf2_ros::Buffer> tf,
    const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  /**
    * @brief Cleanup resources
    */
  void cleanup() override;

  /**
    * @brief Activate controller
    */
  void activate() override;

  /**
    * @brief Deactivate controller
    */
  void deactivate() override;

  /**
    * @brief Reset the controller state between tasks
    */
  void reset();

  /**
    * @brief Main method to compute velocities using the optimizer
    * @param robot_pose Robot pose
    * @param robot_speed Robot speed
    * @param goal_checker Pointer to the goal checker for awareness if completed task
    */
  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & robot_pose,
    const geometry_msgs::msg::Twist & robot_speed,
    nav2_core::GoalChecker * goal_checker) override;

  /**
    * @brief Set new reference path to track
    * @param path Path to track
    */
  void setPlan(const nav_msgs::msg::Path & path) override;

  /**
    * @brief Set new speed limit from callback
    * @param speed_limit Speed limit to use
    * @param percentage Bool if the speed limit is absolute or relative
    */
  void setSpeedLimit(const double & speed_limit, const bool & percentage) override;

protected:
  /**
    * @brief Visualize trajectories
    * @param transformed_plan Transformed input plan
    */
  void visualize(nav_msgs::msg::Path transformed_plan);
  /**
    * @brief Callback for the laser scan subscriber
    * @param msg Received LaserScan message
    */
  void laserScanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg);

  std::string name_;
  rclcpp_lifecycle::LifecycleNode::WeakPtr parent_;
  rclcpp::Clock::SharedPtr clock_;
  rclcpp::Logger logger_{rclcpp::get_logger("SORTHAMController")};
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;

  std::unique_ptr<ParametersHandler> parameters_handler_;
  Optimizer optimizer_;
  PathHandler path_handler_;
  TrajectoryVisualizer trajectory_visualizer_;

  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  std::string scan_topic_;
  std::vector<geometry_msgs::msg::Point> obstacle_points_;
  std::mutex obstacle_points_mutex_;
  double lidar_max_range_{3.5};
  double lidar_min_range_{0.1};
  std::string robot_base_frame_;

  bool visualize_;

  double reset_period_;
  // Last time computeVelocityCommands was called
  rclcpp::Time last_time_called_;
};

}  // namespace nav2_sortham_controller

#endif  // NAV2_SORTHAM_CONTROLLER__CONTROLLER_HPP_
