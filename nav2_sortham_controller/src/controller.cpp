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

#include <stdint.h>
#include <chrono>
#include "nav2_sortham_controller/controller.hpp"
#include "nav2_sortham_controller/tools/utils.hpp"

#include "sensor_msgs/msg/laser_scan.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2/utils.h"
#include <cmath>

// #define BENCHMARK_TESTING

namespace nav2_sortham_controller
{

void SORTHAMController::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name, const std::shared_ptr<tf2_ros::Buffer> tf,
  const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  parent_ = parent;
  costmap_ros_ = costmap_ros;
  tf_buffer_ = tf;
  name_ = name;
  parameters_handler_ = std::make_unique<ParametersHandler>(parent);

  auto node = parent_.lock();
  clock_ = node->get_clock();
  last_time_called_ = clock_->now();
  // Get high-level controller parameters
  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(visualize_, "visualize", false);
  getParam(reset_period_, "reset_period", 1.0);
  getParam(scan_topic_, "scan_topic", std::string("scan"));
  getParam(lidar_max_range_, "lidar_max_range", 3.5);
  getParam(lidar_min_range_, "lidar_min_range", 0.1);
  robot_base_frame_ = costmap_ros_->getBaseFrameID();

  // Configure composed objects
  optimizer_.initialize(parent_, name_, costmap_ros_, parameters_handler_.get());
  path_handler_.initialize(parent_, name_, costmap_ros_, tf_buffer_, parameters_handler_.get());
  trajectory_visualizer_.on_configure(
    parent_, name_,
    costmap_ros_->getGlobalFrameID(), parameters_handler_.get());
  scan_sub_ = node->create_subscription<sensor_msgs::msg::LaserScan>(
    scan_topic_,
    rclcpp::SensorDataQoS(),
    std::bind(&SORTHAMController::laserScanCallback, this, std::placeholders::_1));

  RCLCPP_INFO(logger_, "Configured SORTHAM Controller: %s", name_.c_str());
}

void SORTHAMController::cleanup()
{
  optimizer_.shutdown();
  trajectory_visualizer_.on_cleanup();
  parameters_handler_.reset();
  scan_sub_.reset();
  RCLCPP_INFO(logger_, "Cleaned up SORTHAM Controller: %s", name_.c_str());
}

void SORTHAMController::activate()
{
  trajectory_visualizer_.on_activate();
  parameters_handler_->start();
  RCLCPP_INFO(logger_, "Activated SORTHAM Controller: %s", name_.c_str());
}

void SORTHAMController::deactivate()
{
  trajectory_visualizer_.on_deactivate();
  RCLCPP_INFO(logger_, "Deactivated SORTHAM Controller: %s", name_.c_str());
}

void SORTHAMController::reset()
{
  optimizer_.reset();
  std::lock_guard<std::mutex> lock(obstacle_points_mutex_);
}

geometry_msgs::msg::TwistStamped SORTHAMController::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & robot_pose,
  const geometry_msgs::msg::Twist & robot_speed,
  nav2_core::GoalChecker * goal_checker)
{
#ifdef BENCHMARK_TESTING
  auto start = std::chrono::system_clock::now();
#endif

  if (clock_->now() - last_time_called_ > rclcpp::Duration::from_seconds(reset_period_)) {
    reset();
  }
  last_time_called_ = clock_->now();

  std::vector<geometry_msgs::msg::Point> current_obstacle_points;
  {
      std::lock_guard<std::mutex> lock(obstacle_points_mutex_);
      current_obstacle_points = obstacle_points_;
  }

  std::lock_guard<std::mutex> param_lock(*parameters_handler_->getLock());
  geometry_msgs::msg::Pose goal = path_handler_.getTransformedGoal(robot_pose.header.stamp).pose;

  nav_msgs::msg::Path transformed_plan = path_handler_.transformPath(robot_pose);

  nav2_costmap_2d::Costmap2D * costmap = costmap_ros_->getCostmap();
  std::unique_lock<nav2_costmap_2d::Costmap2D::mutex_t> costmap_lock(*(costmap->getMutex()));

  geometry_msgs::msg::TwistStamped cmd =
    optimizer_.evalControl(robot_pose, robot_speed, transformed_plan, goal, current_obstacle_points, goal_checker);

#ifdef BENCHMARK_TESTING
  auto end = std::chrono::system_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  RCLCPP_INFO(logger_, "Control loop execution time: %ld [ms]", duration);
#endif

  if (visualize_) {
    visualize(std::move(transformed_plan));
  }

  return cmd;
}

void SORTHAMController::visualize(nav_msgs::msg::Path transformed_plan)
{
  trajectory_visualizer_.add(optimizer_.getGeneratedTrajectories(), "Candidate Trajectories");
  trajectory_visualizer_.add(optimizer_.getOptimizedTrajectory(), "Optimal Trajectory");
  trajectory_visualizer_.visualize(std::move(transformed_plan));
}

void SORTHAMController::setPlan(const nav_msgs::msg::Path & path)
{
  path_handler_.setPath(path);
}

void SORTHAMController::setSpeedLimit(const double & speed_limit, const bool & percentage)
{
  optimizer_.setSpeedLimit(speed_limit, percentage);
}

void SORTHAMController::laserScanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
  // 1. Get transform from laser frame to robot base frame at the time of the scan
  geometry_msgs::msg::TransformStamped laser_to_base_transform;
  try {
    laser_to_base_transform = tf_buffer_->lookupTransform(
      robot_base_frame_, msg->header.frame_id,
      msg->header.stamp, rclcpp::Duration::from_seconds(0.1)); // Small tolerance
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN_THROTTLE(
       logger_, *clock_, 1000, "Could not transform %s to %s: %s",
       msg->header.frame_id.c_str(), robot_base_frame_.c_str(), ex.what());
    return;
  }

  // 2. Process scan points
  std::vector<geometry_msgs::msg::Point> current_points;
  current_points.reserve(msg->ranges.size());

  for (size_t i = 0; i < msg->ranges.size(); ++i) {
    float range = msg->ranges[i];

    // Filter out invalid points (NaN, Inf) and points outside desired range
    if (std::isnan(range) || std::isinf(range) ||
        range < lidar_min_range_ || range > lidar_max_range_)
    {
      continue;
    }

    // Convert polar to Cartesian in laser frame
    float angle = msg->angle_min + i * msg->angle_increment;
    geometry_msgs::msg::Point point_laser_frame;
    point_laser_frame.x = range * std::cos(angle);
    point_laser_frame.y = range * std::sin(angle);
    point_laser_frame.z = 0.0; // Assuming 2D lidar on a plane

    // Transform point to base frame
    geometry_msgs::msg::Point point_base_frame;
    tf2::doTransform(point_laser_frame, point_base_frame, laser_to_base_transform);

    current_points.push_back(point_base_frame);
  }

  // 3. Store the processed points (thread-safe)
  {
    std::lock_guard<std::mutex> lock(obstacle_points_mutex_);
    obstacle_points_ = std::move(current_points);
    RCLCPP_INFO_THROTTLE(logger_, *clock_, 2000, "Processed %zu obstacle points", obstacle_points_.size());
  }
}

}  // namespace nav2_sortham_controller

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(nav2_sortham_controller::SORTHAMController, nav2_core::Controller)
