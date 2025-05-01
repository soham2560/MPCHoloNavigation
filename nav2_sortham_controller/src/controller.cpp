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
#include "nav2_sortham_controller/optimizer.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "nav2_sortham_controller/tools/utils.hpp"


#include "sensor_msgs/msg/laser_scan.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
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
  getParam(lidar_max_range_, "lidar_max_range", 2.0);
  getParam(lidar_min_range_, "lidar_min_range", 0.5);
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

  obstacle_marker_pub_ = node->create_publisher<visualization_msgs::msg::MarkerArray>(
    "~/processed_obstacles", 1);

  RCLCPP_INFO(logger_, "Configured SORTHAM Controller: %s", name_.c_str());
}

void SORTHAMController::cleanup()
{
  optimizer_.shutdown();
  trajectory_visualizer_.on_cleanup();
  parameters_handler_.reset();
  scan_sub_.reset();
  obstacle_marker_pub_.reset();
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

void SORTHAMController::addDeleteAllMarkers(
    visualization_msgs::msg::MarkerArray& markers, int& id_counter) const
{
    visualization_msgs::msg::Marker clear_marker;
    clear_marker.header.stamp = clock_->now();
    clear_marker.id = id_counter++; // Use a consistent ID? Or unique ok? Let's use unique.
    clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;

    std::vector<std::pair<std::string, std::string>> namespaces_frames = {
        {"obstacle_centers", robot_base_frame_},
        {"obstacle_radii",   robot_base_frame_},
        {"obstacle_padding", robot_base_frame_},
        {"milestones", costmap_ros_->getGlobalFrameID()},
        {"active_target_milestone", costmap_ros_->getGlobalFrameID()}
    };

    for (const auto& nf_pair : namespaces_frames) {
        clear_marker.header.frame_id = nf_pair.second;
        clear_marker.ns = nf_pair.first;
        markers.markers.push_back(clear_marker);
        clear_marker.id = 0;
    }
    id_counter = 1;
}

void SORTHAMController::addObstacleMarkers(
    visualization_msgs::msg::MarkerArray& markers, int& id_counter) const
{
    const auto& obstacles_info = optimizer_.getLastProcessedObstacles();
    if (obstacles_info.empty()) {
        return;
    }

    double robot_radius = optimizer_.getRobotRadius();
    double safety_margin = optimizer_.getLidarSafetyMargin();
    double avoidance_padding = robot_radius + safety_margin;
    auto now = clock_->now();

    for (const auto& obs : obstacles_info) {
        if (obs.num_points == 0 || obs.radius_sq < 0) continue;

        double radius = (obs.radius_sq > 1e-9) ? std::sqrt(obs.radius_sq) : 0.0;
        double avoidance_radius = radius + avoidance_padding;

        std_msgs::msg::Header obs_header;
        obs_header.frame_id = robot_base_frame_;
        obs_header.stamp = now;

        // Center Sphere
        visualization_msgs::msg::Marker center_marker;
        center_marker.header = obs_header;
        center_marker.ns = "obstacle_centers";
        center_marker.id = id_counter++;
        center_marker.type = visualization_msgs::msg::Marker::SPHERE;
        center_marker.action = visualization_msgs::msg::Marker::ADD;
        center_marker.pose.position.x = obs.cx; center_marker.pose.position.y = obs.cy; center_marker.pose.position.z = 0.1;
        center_marker.pose.orientation.w = 1.0;
        center_marker.scale = utils::createScale(0.1, 0.1, 0.1);
        center_marker.color = utils::createColor(1.0f, 0.0f, 0.0f, 0.8f); // Red
        center_marker.lifetime = rclcpp::Duration(1, 0);
        markers.markers.push_back(center_marker);

        // Radius Cylinder
        visualization_msgs::msg::Marker radius_marker;
        radius_marker.header = obs_header;
        radius_marker.ns = "obstacle_radii";
        radius_marker.id = id_counter++;
        radius_marker.type = visualization_msgs::msg::Marker::CYLINDER;
        radius_marker.action = visualization_msgs::msg::Marker::ADD;
        radius_marker.pose.position.x = obs.cx; radius_marker.pose.position.y = obs.cy; radius_marker.pose.position.z = 0.05;
        radius_marker.pose.orientation.w = 1.0;
        radius_marker.scale = utils::createScale(radius * 2.0, radius * 2.0, 0.02);
        radius_marker.color = utils::createColor(1.0f, 1.0f, 0.0f, 0.5f); // Yellow
        radius_marker.lifetime = rclcpp::Duration(1, 0);
        markers.markers.push_back(radius_marker);

        // Padding Cylinder
        visualization_msgs::msg::Marker padding_marker;
        padding_marker.header = obs_header;
        padding_marker.ns = "obstacle_padding";
        padding_marker.id = id_counter++;
        padding_marker.type = visualization_msgs::msg::Marker::CYLINDER;
        padding_marker.action = visualization_msgs::msg::Marker::ADD;
        padding_marker.pose.position.x = obs.cx; padding_marker.pose.position.y = obs.cy; padding_marker.pose.position.z = 0.02;
        padding_marker.pose.orientation.w = 1.0;
        padding_marker.scale = utils::createScale(avoidance_radius * 2.0, avoidance_radius * 2.0, 0.01);
        padding_marker.color = utils::createColor(1.0f, 0.5f, 0.0f, 0.3f); // Orange
        padding_marker.lifetime = rclcpp::Duration(1, 0);
        markers.markers.push_back(padding_marker);
    }
}

void SORTHAMController::addMilestoneMarkers(
    visualization_msgs::msg::MarkerArray& markers, int& id_counter) const
{
    const auto& milestones = optimizer_.getCurrentMilestones();
    if (milestones.empty()) {
        return;
    }

    visualization_msgs::msg::Marker milestones_marker;
    milestones_marker.header.frame_id = costmap_ros_->getGlobalFrameID();
    milestones_marker.header.stamp = clock_->now();
    milestones_marker.ns = "milestones";
    milestones_marker.id = id_counter++;
    milestones_marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    milestones_marker.action = visualization_msgs::msg::Marker::ADD;
    milestones_marker.pose.orientation.w = 1.0;
    milestones_marker.scale = utils::createScale(0.15, 0.15, 0.15);
    milestones_marker.color = utils::createColor(0.0f, 0.7f, 1.0f, 0.6f); // Light blue
    milestones_marker.lifetime = rclcpp::Duration(1, 0);

    for (const auto& pose : milestones) {
        milestones_marker.points.push_back(pose.position);
    }
    markers.markers.push_back(milestones_marker);
}


void SORTHAMController::addActiveTargetMarker(
    visualization_msgs::msg::MarkerArray& markers, int& id_counter) const
{
    std::optional<geometry_msgs::msg::Pose> active_target_pose_opt = optimizer_.getLastLocalTarget();
    if (!active_target_pose_opt.has_value()) {
        return;
    }

    std::string global_frame = costmap_ros_->getGlobalFrameID();
    auto now = clock_->now();
    geometry_msgs::msg::Pose target_pose = active_target_pose_opt.value();
    bool is_fallback = optimizer_.getCurrentMilestones().empty();

    // Arrow Marker
    visualization_msgs::msg::Marker target_arrow_marker;
    target_arrow_marker.header.frame_id = global_frame;
    target_arrow_marker.header.stamp = now;
    target_arrow_marker.ns = "active_target_milestone";
    target_arrow_marker.id = id_counter++;
    target_arrow_marker.type = visualization_msgs::msg::Marker::ARROW;
    target_arrow_marker.action = visualization_msgs::msg::Marker::ADD;
    target_arrow_marker.pose = target_pose;
    target_arrow_marker.scale = utils::createScale(0.7, 0.1, 0.1);
    target_arrow_marker.color = is_fallback ? utils::createColor(1.0f, 0.0f, 0.0f, 0.9f) // Red fallback
                                        : utils::createColor(0.0f, 1.0f, 0.0f, 0.9f); // Green target
    target_arrow_marker.lifetime = rclcpp::Duration(1, 0);
    markers.markers.push_back(target_arrow_marker);

    // Sphere Marker at Target Location
    visualization_msgs::msg::Marker target_sphere_marker;
    target_sphere_marker.header.frame_id = global_frame;
    target_sphere_marker.header.stamp = now;
    target_sphere_marker.ns = "active_target_milestone";
    target_sphere_marker.id = id_counter++;
    target_sphere_marker.type = visualization_msgs::msg::Marker::SPHERE;
    target_sphere_marker.action = visualization_msgs::msg::Marker::ADD;
    target_sphere_marker.pose = target_pose; // Position only matters here
    target_sphere_marker.pose.orientation.w = 1.0; // Ensure valid quat for sphere
    target_sphere_marker.scale = utils::createScale(0.25, 0.25, 0.25);
    target_sphere_marker.color = is_fallback ? utils::createColor(1.0f, 0.0f, 0.0f, 0.9f) // Red fallback
                                         : utils::createColor(0.0f, 1.0f, 0.0f, 0.9f); // Green target
    target_sphere_marker.lifetime = rclcpp::Duration(1, 0);
    markers.markers.push_back(target_sphere_marker);
}


// ==========================================================================
// Main Visualization Function
// ==========================================================================

void SORTHAMController::visualize(nav_msgs::msg::Path transformed_plan)
{
    if (visualize_) {
        const nav_msgs::msg::Path& plan_ref = transformed_plan;
        trajectory_visualizer_.add(optimizer_.getOptimizedTrajectory(), "Optimal Trajectory");
        trajectory_visualizer_.visualize(plan_ref);
    }

    if (!visualize_ || !obstacle_marker_pub_ || obstacle_marker_pub_->get_subscription_count() == 0) {
        return;
    }

    visualization_msgs::msg::MarkerArray marker_array_msg;
    int marker_id = 0;

    addDeleteAllMarkers(marker_array_msg, marker_id);
    addObstacleMarkers(marker_array_msg, marker_id);
    addMilestoneMarkers(marker_array_msg, marker_id);
    addActiveTargetMarker(marker_array_msg, marker_id);

    obstacle_marker_pub_->publish(marker_array_msg);
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
  geometry_msgs::msg::TransformStamped laser_to_base_transform;
  try {
    laser_to_base_transform = tf_buffer_->lookupTransform(
      robot_base_frame_, msg->header.frame_id,
      msg->header.stamp, rclcpp::Duration::from_seconds(0.1));
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN_THROTTLE(
       logger_, *clock_, 1000, "Could not transform laser scan from %s to %s: %s",
       msg->header.frame_id.c_str(), robot_base_frame_.c_str(), ex.what());
    return;
  }

  nav2_costmap_2d::Costmap2D* costmap = costmap_ros_->getCostmap();
  if (!costmap) {
       RCLCPP_WARN_THROTTLE(logger_, *clock_, 1000, "Local costmap pointer is null, cannot filter points.");
       return;
  }
  double origin_x = costmap->getOriginX();
  double origin_y = costmap->getOriginY();
  double size_x_meters = costmap->getSizeInMetersX();
  double size_y_meters = costmap->getSizeInMetersY();
  std::string costmap_frame = costmap_ros_->getGlobalFrameID();

  geometry_msgs::msg::TransformStamped base_to_costmap_transform;
  try {
      base_to_costmap_transform = tf_buffer_->lookupTransform(
          costmap_frame, robot_base_frame_,
          msg->header.stamp, rclcpp::Duration::from_seconds(0.1));
  } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(
         logger_, *clock_, 1000, "Could not get transform from %s to %s needed for filtering: %s",
         robot_base_frame_.c_str(), costmap_frame.c_str(), ex.what());
      return;
  }

  std::vector<geometry_msgs::msg::Point> filtered_points;
  filtered_points.reserve(msg->ranges.size());

  for (size_t i = 0; i < msg->ranges.size(); ++i) {
    float range = msg->ranges[i];

    if (std::isnan(range) || std::isinf(range) ||
        range < lidar_min_range_ || range > lidar_max_range_)
    {
      continue;
    }

    float angle = msg->angle_min + i * msg->angle_increment;
    geometry_msgs::msg::Point point_laser_frame;
    point_laser_frame.x = range * std::cos(angle);
    point_laser_frame.y = range * std::sin(angle);
    point_laser_frame.z = 0.0;

    geometry_msgs::msg::Point point_base_frame;
    tf2::doTransform(point_laser_frame, point_base_frame, laser_to_base_transform);

    geometry_msgs::msg::Point point_costmap_frame;
    tf2::doTransform(point_base_frame, point_costmap_frame, base_to_costmap_transform);

    bool in_bounds = (point_costmap_frame.x >= origin_x &&
                      point_costmap_frame.x < (origin_x + size_x_meters) &&
                      point_costmap_frame.y >= origin_y &&
                      point_costmap_frame.y < (origin_y + size_y_meters));

    if (in_bounds) {
        filtered_points.push_back(point_base_frame);
    }
  }

  {
    std::lock_guard<std::mutex> lock(obstacle_points_mutex_);
    obstacle_points_ = std::move(filtered_points);
    RCLCPP_DEBUG_THROTTLE(logger_, *clock_, 2000, "Processed %zu obstacle points within local costmap", obstacle_points_.size());
  }
}

}  // namespace nav2_sortham_controller

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(nav2_sortham_controller::SORTHAMController, nav2_core::Controller)
