#ifndef _PLAN_TESTER_HPP_
#define _PLAN_TESTER_HPP_

// 核心 ROS 头文件
#include <ros/ros.h>

// 依赖的包和消息类型
#include "plan_env/sdf_map.h"
#include "visualizer/visualizer.hpp"
#include "front_end/jps_planner/jps_planner.h"
#include "back_end/optimizer.h"
#include "carstatemsgs/CarState.h"
#include "carstatemsgs/Polynome.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/PoseArray.h"
#include "nav_msgs/Odometry.h"
#include "std_msgs/Bool.h"
#include "visualization_msgs/MarkerArray.h"
#include "tf/tf.h"
#include "std_msgs/Int32MultiArray.h"

// C++ 标准库
#include <thread>
#include <vector>
#include <algorithm>
#include <limits>

// 本地模块
#include "plan_tester/hungarian.hpp"
#include "plan_tester/branch_and_bound.hpp"

enum StateMachine{
  INIT,
  IDLE,
  PLANNING,
  REPLAN,
  GOINGTOGOAL,
  EMERGENCY_STOP,
};

class PlanTester
{
private:
    ros::NodeHandle nh_;

    // 核心组件
    std::shared_ptr<SDFmap> sdfmap_;
    std::shared_ptr<Visualizer> visualizer_;
    std::shared_ptr<MSPlanner> msplanner_;
    std::shared_ptr<JPS::JPSPlanner> jps_planner_;

    // 任务与路径点管理
    std::vector<Eigen::Vector3d> item_positions_;
    std::vector<Eigen::Vector3d> target_positions_;
    std::vector<Eigen::Vector3d> ordered_waypoints_;
    int current_waypoint_idx_;

    // ROS 通信
    ros::Subscriber items_sub_;
    ros::Subscriber targets_sub_;
    ros::Subscriber current_state_sub_;
    ros::Timer main_thread_timer_;
    ros::Timer map_update_timer_;
    ros::Publisher cmd_pub_;
    ros::Publisher mpc_polynome_pub_;
    ros::Publisher emergency_stop_pub_;
    ros::Publisher record_pub_;
    ros::Publisher marker_pub_;
    ros::Publisher bnb_chair_order_pub_;
    ros::Publisher bnb_target_order_pub_;
    ros::Publisher greedy_chair_order_pub_;
    ros::Publisher greedy_target_order_pub_;

    // 状态变量
    ros::Time current_time_;
    Eigen::Vector3d current_state_XYTheta_;
    Eigen::Vector3d current_state_VAJ_;
    Eigen::Vector3d current_state_OAJ_;
    double plan_start_time_;
    Eigen::Vector3d plan_start_state_XYTheta;
    Eigen::Vector3d plan_start_state_VAJ;
    Eigen::Vector3d plan_start_state_OAJ;
    Eigen::Vector3d goal_state_;
    ros::Time Traj_start_time_;
    double Traj_total_time_;
    ros::Time loop_start_time_;
    bool have_geometry_;
    bool have_goal_;
    bool have_items_;
    bool have_targets_;
    bool if_fix_final_;
    Eigen::Vector3d final_state_;
    double replan_time_;
    double max_replan_time_;
    double predicted_traj_start_time_;
    StateMachine state_machine_ = StateMachine::INIT;

    bool set_obs_done = false;
    bool task_plan_finished = false;
    bool reached_target = false;
    bool going_item = false;
    bool going_target = false;
    Eigen::Vector2d goal_item;
    Eigen::Vector2d goal_target;

public:
    PlanTester(ros::NodeHandle nh);
    ~PlanTester();

    // 任务规划流程函数
    void visualizeItemsAndTargets();
    bool solvePathWithBranchAndBound();
    void solvePathWithGreedy();
    void trigger_planning();

    // ROS 回调与主循环
    void MainThread(const ros::TimerEvent& event);
    void MapUpdateThread(const ros::TimerEvent& event);
    void GeometryCallback(const nav_msgs::Odometry::ConstPtr &msg);
    void items_callback(const geometry_msgs::PoseArray::ConstPtr& msg);
    void targets_callback(const geometry_msgs::PoseArray::ConstPtr& msg);
    
    // 辅助函数
    void printStateMachine();
    bool findJPSRoad();
    void MPCPathPub(const double& traj_start_time);
    void paintSquare (const Eigen::Vector2d& c, bool make_obs, double half_size=0.4);
};

PlanTester::PlanTester(ros::NodeHandle nh) : nh_(nh) {

  sdfmap_ = std::make_shared<SDFmap>(nh);
  visualizer_ = std::make_shared<Visualizer>(nh);
  msplanner_ = std::make_shared<MSPlanner>(Config(ros::NodeHandle("~")), nh_, sdfmap_);
  jps_planner_ = std::make_shared<JPS::JPSPlanner>(sdfmap_, nh_);

  items_sub_ = nh_.subscribe<geometry_msgs::PoseArray>("/mission/items", 1, &PlanTester::items_callback, this);
  targets_sub_ = nh_.subscribe<geometry_msgs::PoseArray>("/mission/targets", 1, &PlanTester::targets_callback, this);
  current_state_sub_ = nh_.subscribe<nav_msgs::Odometry>("odom", 1, &PlanTester::GeometryCallback, this);
  main_thread_timer_ = nh_.createTimer(ros::Duration(0.005), &PlanTester::MainThread, this);
  map_update_timer_ = nh_.createTimer(ros::Duration(0.05),&PlanTester::MapUpdateThread, this);
  
  cmd_pub_ = nh_.advertise<carstatemsgs::CarState>("/simulation/PoseSub", 1);
  emergency_stop_pub_ = nh_.advertise<std_msgs::Bool>("/planner/emergency_stop", 1);
  record_pub_ = nh_.advertise<visualization_msgs::Marker>("/planner/calculator_time", 1);
  mpc_polynome_pub_ = nh_.advertise<carstatemsgs::Polynome>("traj", 1);
  marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/planner/markers", 10, true);
  bnb_chair_order_pub_ = nh_.advertise<std_msgs::Int32MultiArray>("/mission/results/bnb/chair_order", 1, true);
  bnb_target_order_pub_ = nh_.advertise<std_msgs::Int32MultiArray>("/mission/results/bnb/target_order", 1, true);
  greedy_chair_order_pub_ = nh_.advertise<std_msgs::Int32MultiArray>("/mission/results/greedy/chair_order", 1, true);
  greedy_target_order_pub_ = nh_.advertise<std_msgs::Int32MultiArray>("/mission/results/greedy/target_order", 1, true);

  have_geometry_ = false;
  have_goal_ = false;
  have_items_ = false;
  have_targets_ = false;
  current_waypoint_idx_ = -1;

  nh_.param<bool>("if_fix_final", if_fix_final_, false);
  if(if_fix_final_){
    nh_.param<double>("final_x", final_state_(0), 0.0);
    nh_.param<double>("final_y", final_state_(1), 0.0);
    nh_.param<double>("final_yaw", final_state_(2), 0.0);
  }

  nh_.param<double>("replan_time", replan_time_, 10000.0);
  nh_.param<double>("max_replan_time", max_replan_time_, 3.0);

  state_machine_ = StateMachine::IDLE;
  loop_start_time_ = ros::Time::now();
}

PlanTester::~PlanTester() {}

void PlanTester::printStateMachine() {
  if(state_machine_ == INIT) ROS_INFO("state_machine_ == INIT");
  if(state_machine_ == IDLE) ROS_INFO("state_machine_ == IDLE");
  if(state_machine_ == PLANNING) ROS_INFO("state_machine_ == PLANNING");
  if(state_machine_ == REPLAN) ROS_INFO("state_machine_ == REPLAN");
}

void PlanTester::GeometryCallback(const nav_msgs::Odometry::ConstPtr &msg) {
  have_geometry_ = true;
  current_state_XYTheta_ << msg->pose.pose.position.x, msg->pose.pose.position.y, tf::getYaw(msg->pose.pose.orientation);
  current_state_VAJ_ << 0.0, 0.0, 0.0;
  current_state_OAJ_ << 0.0, 0.0, 0.0;
  current_time_ = msg->header.stamp;
}

void PlanTester::items_callback(const geometry_msgs::PoseArray::ConstPtr& msg) {
    ROS_INFO("Received %zu item positions.", msg->poses.size());
    item_positions_.clear();
    for (const auto& pose : msg->poses) {
        item_positions_.emplace_back(pose.position.x, pose.position.y, tf::getYaw(pose.orientation));
    }
    have_items_ = true;
    visualizeItemsAndTargets();
    trigger_planning();
}

void PlanTester::targets_callback(const geometry_msgs::PoseArray::ConstPtr& msg) {
    ROS_INFO("Received %zu target positions.", msg->poses.size());
    target_positions_.clear();
    for (const auto& pose : msg->poses) {
        target_positions_.emplace_back(pose.position.x, pose.position.y, tf::getYaw(pose.orientation));
    }
    have_targets_ = true;
    visualizeItemsAndTargets();
    trigger_planning();
}

void PlanTester::trigger_planning() {
    if (!have_items_ || !have_targets_) {
        ROS_INFO("Waiting for both items and targets before planning...");
        return;
    }

    if (state_machine_ != StateMachine::IDLE) {
        ROS_WARN("Planner is busy. Ignoring new mission start signal.");
        return;
    }
    if (!have_geometry_) {
        ROS_ERROR("No odometry received. Cannot start mission.");
        return;
    }
    if (item_positions_.size() != target_positions_.size()) {
        ROS_ERROR("Mismatch between number of items (%zu) and targets (%zu). Aborting.", item_positions_.size(), target_positions_.size());
        return;
    }

    ROS_INFO("\n\n--- Received All Mission Data, Starting Planning ---");
    
    // Execute and log Branch and Bound approach
    if (solvePathWithBranchAndBound()) {
        ROS_INFO("B&B TSP solved. Starting multi-point traversal.");
        current_waypoint_idx_ = 0;
        goal_state_ = ordered_waypoints_[current_waypoint_idx_];
        have_goal_ = true;
    } else {
        ROS_ERROR("Failed to solve B&B TSP. Aborting mission.");
        state_machine_ = StateMachine::IDLE;
        have_goal_ = false;
        // Reset flags even on failure to allow for a new attempt
        have_items_ = false;
        have_targets_ = false;
        return; // Exit if B&B fails
    }

    // Execute and log Greedy approach for comparison
    solvePathWithGreedy();

    // Reset flags for the next mission
    have_items_ = false;
    have_targets_ = false;

    ROS_INFO("-----------------------------------------------------\n\n");
}

void PlanTester::visualizeItemsAndTargets() {
    ROS_INFO("Visualizing markers... Items: %zu, Targets: %zu", item_positions_.size(), target_positions_.size());
    visualization_msgs::MarkerArray marker_array;
    ros::Time now = ros::Time::now();

    // The most robust way to clear and then draw markers is to use a single
    // MarkerArray message. The DELETEALL action will clear all markers previously
    // published by this node, and the subsequent ADD actions will draw the new ones.
    // This is processed atomically by RViz.
    visualization_msgs::Marker clear_marker;
    clear_marker.header.frame_id = "world";
    clear_marker.header.stamp = now;
    clear_marker.action = visualization_msgs::Marker::DELETEALL;
    marker_array.markers.push_back(clear_marker);

    // Add new markers for items
    for (size_t i = 0; i < item_positions_.size(); ++i) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "world";
        marker.header.stamp = now;
        marker.ns = "items";
        marker.id = i;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = item_positions_[i].x();
        marker.pose.position.y = item_positions_[i].y();
        marker.pose.position.z = 0.5;
        marker.scale.x = 0.5;
        marker.scale.y = 0.5;
        marker.scale.z = 0.5;
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 0.8;
        marker.lifetime = ros::Duration(); // Persist until deleted
        marker_array.markers.push_back(marker);
    }

    // Add new markers for targets
    for (size_t i = 0; i < target_positions_.size(); ++i) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "world";
        marker.header.stamp = now;
        marker.ns = "targets";
        marker.id = i;
        marker.type = visualization_msgs::Marker::SPHERE;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = target_positions_[i].x();
        marker.pose.position.y = target_positions_[i].y();
        marker.pose.position.z = 0.5;
        marker.scale.x = 0.5;
        marker.scale.y = 0.5;
        marker.scale.z = 0.5;
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        marker.color.a = 0.8;
        marker.lifetime = ros::Duration(); // Persist until deleted
        marker_array.markers.push_back(marker);
    }

    marker_pub_.publish(marker_array);
    ROS_INFO("Published %zu markers to topic %s.", marker_array.markers.size(), marker_pub_.getTopic().c_str());
}

void PlanTester::solvePathWithGreedy()
{
    ROS_INFO("--- Starting Greedy Task Planning (Fixed Pairs) ---");

    if (item_positions_.empty()) {
        ROS_ERROR("No item positions available for greedy planning.");
        return;
    }

    int num_tasks = item_positions_.size();
    std::vector<bool> visited_tasks(num_tasks, false);
    
    std::vector<Eigen::Vector3d> greedy_waypoints;
    greedy_waypoints.push_back(current_state_XYTheta_);

    std_msgs::Int32MultiArray chair_order_msg;
    std_msgs::Int32MultiArray target_order_msg;
    std::stringstream chair_ss, target_ss;

    Eigen::Vector3d current_pos = current_state_XYTheta_;

    // Greedy strategy with fixed pairs: Always go to the nearest unvisited item,
    // then to its corresponding target.
    for (int step = 0; step < num_tasks; ++step) {
        // 1. Find the nearest unvisited item from the current position.
        double min_dist = std::numeric_limits<double>::max();
        int next_task_idx = -1;
        for (int i = 0; i < num_tasks; ++i) {
            if (!visited_tasks[i]) {
                // Use JPS path length for more accurate distance
                bool path_found = jps_planner_->plan(current_pos, item_positions_[i]);
                double dist = path_found ? jps_planner_->getPathLength() : (current_pos.head<2>() - item_positions_[i].head<2>()).norm();
                if (dist < min_dist) {
                    min_dist = dist;
                    next_task_idx = i;
                }
            }
        }

        if (next_task_idx == -1) {
            ROS_WARN("Could not find next task for greedy planner.");
            break; 
        }

        // 2. Mark this task as visited and add the item and its fixed target to the path.
        visited_tasks[next_task_idx] = true;
        
        // Add item
        greedy_waypoints.push_back(item_positions_[next_task_idx]);
        chair_order_msg.data.push_back(next_task_idx);
        chair_ss << next_task_idx << " ";

        // Add corresponding target
        greedy_waypoints.push_back(target_positions_[next_task_idx]);
        target_order_msg.data.push_back(next_task_idx);
        target_ss << next_task_idx << " ";

        // 3. Update current position to the target of the completed task pair.
        current_pos = target_positions_[next_task_idx];
    }

    greedy_chair_order_pub_.publish(chair_order_msg);
    greedy_target_order_pub_.publish(target_order_msg);
    ROS_INFO("[Greedy] Chair visit order: [ %s]", chair_ss.str().c_str());
    ROS_INFO("[Greedy] Target visit order: [ %s]", target_ss.str().c_str());

    // Calculate total path length for the greedy path
    double total_path_length = 0.0;
    for (size_t i = 1; i < greedy_waypoints.size(); ++i) {
        bool found = jps_planner_->plan(greedy_waypoints[i-1], greedy_waypoints[i]);
        double seg_len = found ? jps_planner_->getPathLength() : (greedy_waypoints[i-1] - greedy_waypoints[i]).head(2).norm();
        total_path_length += seg_len;
    }
    ROS_INFO("[Greedy] Total planned path length: %.3f m", total_path_length);
    ROS_INFO("--- Greedy Planning Finished ---");

    task_plan_finished = true;
}

bool PlanTester::solvePathWithBranchAndBound() {
    ROS_INFO("--- Starting Branch and Bound Task Planning (with Fixed Assignment) ---");
    if (item_positions_.empty() || target_positions_.empty() || item_positions_.size() != target_positions_.size()) {
        ROS_ERROR("Invalid item or target positions.");
        return false;
    }

    int num_tasks = item_positions_.size();
    
    // 1. 给定固定的指派：item_i 到 target_i
    std::vector<int> fixed_assignment(num_tasks);
    for(int i = 0; i < num_tasks; ++i) {
        fixed_assignment[i] = i;
    }
    ROS_INFO("[B&B] Using fixed assignment (item_i -> target_i).");

    int matrix_size = 1 + 2 * num_tasks; // 0:start, 1..n:items, n+1..2n:targets
    Eigen::MatrixXd all_dists(matrix_size, matrix_size);
    std::vector<Eigen::Vector3d> all_points;
    
    all_points.push_back(current_state_XYTheta_);
    all_points.insert(all_points.end(), item_positions_.begin(), item_positions_.end());
    all_points.insert(all_points.end(), target_positions_.begin(), target_positions_.end());

    for (int i = 0; i < matrix_size; ++i) {
        for (int j = i; j < matrix_size; ++j) {
            if (i == j) {
                all_dists(i, j) = 0;
                continue;
            }
            bool path_found = jps_planner_->plan(all_points[i], all_points[j]);
            double len = path_found ? jps_planner_->getPathLength() : std::numeric_limits<double>::max();
            if (len >= std::numeric_limits<double>::max()) {
                 ROS_WARN("Cannot find path between point %d and %d. Setting cost to infinity.", i, j);
            }
            all_dists(i, j) = all_dists(j, i) = len;
        }
    }
    
    // 3. 实例化并使用带有固定指派的组合式B&B算法
    BranchAndBoundCombined bnb_solver(all_dists, num_tasks);
    std::vector<int> combined_path_indices;
    // 调用新的 solveWithFixedAssignment 方法
    double final_cost = bnb_solver.solve(fixed_assignment, combined_path_indices);
    
    if (final_cost == std::numeric_limits<double>::max()) {
        ROS_ERROR("Failed to find a valid path with fixed assignment.");
        return false;
    }
    
    ROS_INFO("[B&B] Fixed Assignment-Routing solved. Minimum cost: %.3f", final_cost);

    // 4. 重建最终路径并发布任务顺序
    ordered_waypoints_.clear();
    ordered_waypoints_.push_back(current_state_XYTheta_);

    std_msgs::Int32MultiArray chair_order_msg;
    std_msgs::Int32MultiArray target_order_msg;
    std::stringstream chair_ss, target_ss;

    // 遍历返回的路径索引序列
    for (size_t i = 1; i < combined_path_indices.size(); ++i) {
        int global_idx = combined_path_indices[i];
        if (global_idx > 0 && global_idx <= num_tasks) { // 椅子
            int chair_idx = global_idx - 1;
            ordered_waypoints_.push_back(item_positions_[chair_idx]);
            chair_order_msg.data.push_back(chair_idx);
            chair_ss << chair_idx << " ";
        } else if (global_idx > num_tasks && global_idx <= 2 * num_tasks) { // 目标
            int target_idx = global_idx - num_tasks - 1;
            ordered_waypoints_.push_back(target_positions_[target_idx]);
            target_order_msg.data.push_back(target_idx);
            target_ss << target_idx << " ";
        }
    }

    bnb_chair_order_pub_.publish(chair_order_msg);
    bnb_target_order_pub_.publish(target_order_msg);
    ROS_INFO("[B&B] Chair visit order: [ %s]", chair_ss.str().c_str());
    ROS_INFO("[B&B] Target visit order: [ %s]", target_ss.str().c_str());

    double final_path_length = 0;
    for(size_t i = 1; i < ordered_waypoints_.size(); ++i) {
        final_path_length += (ordered_waypoints_[i-1] - ordered_waypoints_[i]).head(2).norm();
    }
    ROS_INFO("[B&B] Total planned path length: %.3f m", final_path_length);
    ROS_INFO("--- B&B Planning Finished ---");

    task_plan_finished = true;

    return true;
}

void PlanTester::paintSquare (const Eigen::Vector2d& c, bool make_obs, double half_size)
{
  for (double x = c.x() - half_size; x <= c.x() + half_size; x += sdfmap_->grid_interval_)
    for (double y = c.y() - half_size; y <= c.y() + half_size; y += sdfmap_->grid_interval_)
      make_obs ? sdfmap_->setObs(Eigen::Vector2d(x,y)) : sdfmap_->setFree(Eigen::Vector2d(x,y));
}

void PlanTester::MapUpdateThread(const ros::TimerEvent& event){
  if(!have_geometry_){
    return;
  }

  if (task_plan_finished && !set_obs_done){
    // items -> OBS
    for (const auto& p : item_positions_) paintSquare(p.head<2>(), /*make_obs=*/true);
    sdfmap_->updateESDF2d();
    set_obs_done = true;
  }

  if (set_obs_done){
    double dist_to_item = (current_state_XYTheta_.head<2>() - goal_item).norm();
    if (going_item && dist_to_item < 0.8)
    {
      paintSquare(goal_item, /*make_obs=*/false, 0.5);
      sdfmap_->updateESDF2d();
      ROS_INFO("Item reached, unlocking the area for passing.");
    }
  }
}

void PlanTester::MainThread(const ros::TimerEvent& event) {
  if (!have_geometry_ || !have_goal_) return;

  // collision check
  if(have_geometry_){
    if(sdfmap_->getDistanceReal(Eigen::Vector2d(current_state_XYTheta_.x(), current_state_XYTheta_.y())) < 0.0){
      std_msgs::Bool emergency_stop;
      emergency_stop.data = true;
      emergency_stop_pub_.publish(emergency_stop);
      state_machine_ = EMERGENCY_STOP;
      ROS_INFO_STREAM("current_state_XYTheta_: " << current_state_XYTheta_.transpose());
      ROS_INFO_STREAM("Dis: " << sdfmap_->getDistanceReal(Eigen::Vector2d(current_state_XYTheta_.x(), current_state_XYTheta_.y())));
      ROS_ERROR("EMERGENCY_STOP!!! too close to obstacle!!!");
      return;
    }
  }
  
  if (state_machine_ == StateMachine::IDLE || 
      ((state_machine_ == StateMachine::PLANNING || state_machine_ == StateMachine::REPLAN) && 
       (ros::Time::now() - loop_start_time_).toSec() > replan_time_)) {
    
    loop_start_time_ = ros::Time::now();
    double current = loop_start_time_.toSec();

    // start new plan
    if (state_machine_ == StateMachine::IDLE) {
      state_machine_ = StateMachine::PLANNING;
      plan_start_time_ = -1;
      predicted_traj_start_time_ = -1;
      plan_start_state_XYTheta = current_state_XYTheta_;
      plan_start_state_VAJ = current_state_VAJ_;
      plan_start_state_OAJ = current_state_OAJ_;
    } 
    // Use predicted distance for replanning in planning state
    else if (state_machine_ == StateMachine::PLANNING || state_machine_ == StateMachine::REPLAN) {
      
      if (((current_state_XYTheta_ - goal_state_).head(2).squaredNorm() + fmod(fabs((plan_start_state_XYTheta - goal_state_)[2]), 2.0 * M_PI)*0.02 < 1.0) ||
          msplanner_->final_traj_.getTotalDuration() < max_replan_time_) {
        state_machine_ = StateMachine::GOINGTOGOAL;
        return;
      }

      state_machine_ = StateMachine::REPLAN;

      predicted_traj_start_time_ = current + max_replan_time_ - plan_start_time_;
      msplanner_->get_the_predicted_state(predicted_traj_start_time_, plan_start_state_XYTheta, plan_start_state_VAJ, plan_start_state_OAJ);

    } 
    
    ROS_INFO("\033[32;40m \n\n\n\n\n-------------------------------------start new plan------------------------------------------ \033[0m");
    
    visualizer_->finalnodePub(plan_start_state_XYTheta, goal_state_);
    ROS_INFO("init_state_: %.10f  %.10f  %.10f", plan_start_state_XYTheta(0), plan_start_state_XYTheta(1), plan_start_state_XYTheta(2));
    ROS_INFO("goal_state_: %.10f  %.10f  %.10f", goal_state_(0), goal_state_(1), goal_state_(2));
    std::cout<<"<arg name=\"start_x_\" value=\""<< plan_start_state_XYTheta(0) <<"\"/>"<<std::endl;
    std::cout<<"<arg name=\"start_y_\" value=\""<< plan_start_state_XYTheta(1) <<"\"/>"<<std::endl;
    std::cout<<"<arg name=\"start_yaw_\" value=\""<< plan_start_state_XYTheta(2) <<"\"/>"<<std::endl;
    std::cout<<"<arg name=\"final_x_\" value=\""<< goal_state_(0) <<"\"/>"<<std::endl;
    std::cout<<"<arg name=\"final_y_\" value=\""<< goal_state_(1) <<"\"/>"<<std::endl;
    std::cout<<"<arg name=\"final_yaw_\" value=\""<< goal_state_(2) <<"\"/>"<<std::endl;

    std::cout<<"plan_start_state_VAJ: "<<plan_start_state_VAJ.transpose()<<std::endl;
    std::cout<<"plan_start_state_OAJ: "<<plan_start_state_OAJ.transpose()<<std::endl;

    ROS_INFO("<arg name=\"start_x_\" value=\"%f\"/>", plan_start_state_XYTheta(0));
    ROS_INFO("<arg name=\"start_y_\" value=\"%f\"/>", plan_start_state_XYTheta(1));
    ROS_INFO("<arg name=\"start_yaw_\" value=\"%f\"/>", plan_start_state_XYTheta(2));
    ROS_INFO("<arg name=\"final_x_\" value=\"%f\"/>", goal_state_(0));
    ROS_INFO("<arg name=\"final_y_\" value=\"%f\"/>", goal_state_(1));
    ROS_INFO("<arg name=\"final_yaw_\" value=\"%f\"/>", goal_state_(2));

    ROS_INFO_STREAM("plan_start_state_VAJ: " << plan_start_state_VAJ.transpose());
    ROS_INFO_STREAM("plan_start_state_OAJ: " << plan_start_state_OAJ.transpose());

    // 找最近点函数
    auto nearest_point = [&](const std::vector<Eigen::Vector3d>& L, const Eigen::Vector2d& p,
                            double& best_dist) -> Eigen::Vector2d {
      best_dist = std::numeric_limits<double>::max();
      Eigen::Vector2d best_pt = p;
      for (const auto& v : L) {
        double d = (p - v.head<2>()).norm();
        if (d < best_dist) {
          best_dist = d;
          best_pt = v.head<2>();
        }
      }
      return best_pt;
    };

    if (set_obs_done)
    {
      Eigen::Vector2d goal_xy = goal_state_.head<2>();

      double dist_item, dist_target;
      goal_item   = nearest_point(item_positions_, goal_xy, dist_item);
      goal_target = nearest_point(target_positions_, goal_xy, dist_target);
      // ROS_INFO("Distance to nearest item: %.2f m, target: %.2f m", dist_item, dist_target);

      // 根据最近的点来决定逻辑
      if (dist_item < dist_target) 
      {
        // 最近的是 item
        paintSquare(goal_item, /*make_obs=*/false, 0.5);
        sdfmap_->updateESDF2d();
        going_item = true;
        going_target = false;
        ROS_INFO("Heading to item.");
      }
      else
      {
        going_item = false;
        going_target = true;
        ROS_INFO("Heading to target.");
      }
    }

    // front end
    ros::Time astar_start_time = ros::Time::now();
    if(!findJPSRoad()){
      state_machine_ = EMERGENCY_STOP;
      ROS_ERROR("EMERGENCY_STOP!!! can not find astar road !!!");
      return;
    }
    ROS_INFO("\033[41;37m all of front end time:%f \033[0m", (ros::Time::now()-astar_start_time).toSec());

    // optimizer
    bool result = msplanner_->minco_plan(jps_planner_->flat_traj_);
    if(!result){
      return;
    }

    ROS_INFO("\033[43;32m all of plan time:%f \033[0m", (ros::Time::now().toSec()-current));

    // visualization
    msplanner_->mincoPathPub(msplanner_->final_traj_, plan_start_state_XYTheta, visualizer_->mincoPathPath);
    msplanner_->mincoPointPub(msplanner_->final_traj_, plan_start_state_XYTheta, visualizer_->mincoPointMarker, Eigen::Vector3d(239, 41, 41));
    
    // for replan
    if(plan_start_time_ < 0){
      Traj_start_time_ = ros::Time::now();
      plan_start_time_ = Traj_start_time_.toSec();
    }
    else{
      plan_start_time_ = current + max_replan_time_;
      Traj_start_time_ = ros::Time(plan_start_time_);
    }
    

    MPCPathPub(plan_start_time_);

    Traj_total_time_ = msplanner_->final_traj_.getTotalDuration();

    if (going_item)
    {
      double dist_to_target;
      Eigen::Vector2d now_target = nearest_point(target_positions_, current_state_XYTheta_.head<2>(), dist_to_target);
      if (dist_to_target < 0.8){
        paintSquare(now_target, /*make_obs=*/true, 0.5);
        sdfmap_->updateESDF2d();
        ROS_INFO("Target reached, locking the area as obstacle.");
      }
    }
  }

  if ((ros::Time::now() - Traj_start_time_).toSec() >= Traj_total_time_) {
    if (current_waypoint_idx_ >= 0) {
      current_waypoint_idx_++;
      if (current_waypoint_idx_ < ordered_waypoints_.size()) {
        ROS_INFO("Waypoint %d/%zu reached. Moving to the next one.", current_waypoint_idx_ + 1, ordered_waypoints_.size());
        goal_state_ = ordered_waypoints_[current_waypoint_idx_];
        have_goal_ = true;
        state_machine_ = StateMachine::IDLE;
      } else {
        ROS_INFO("All waypoints visited. Mission complete.");
        state_machine_ = StateMachine::IDLE;
        have_goal_ = false;
        current_waypoint_idx_ = -1;
      }
    } else {
      state_machine_ = StateMachine::IDLE;
      have_goal_ = false;
    }
  }
}

bool PlanTester::findJPSRoad(){

  ros::Time current = ros::Time::now();
  Eigen::Vector3d start_state;
  std::vector<Eigen::Vector3d> start_path;
  std::vector<Eigen::Vector3d> start_path_both_end;
  bool if_forward = true;
  if(plan_start_time_ > 0){
    start_path = msplanner_->get_the_predicted_state_and_path(predicted_traj_start_time_, predicted_traj_start_time_ + jps_planner_->jps_truncation_time_, plan_start_state_XYTheta, start_state, if_forward);
    u_int start_path_size = start_path.size();
    u_int start_path_i = 0;
    for(; start_path_i < start_path_size; start_path_i++){
      if(!jps_planner_->JPS_check_if_collision(start_path[start_path_i].head(2)))
        break;
    }
    if(start_path_i == 0){
      start_state = plan_start_state_XYTheta;
      start_path_both_end.push_back(start_path.front());
      start_path_both_end.push_back(start_state);
    }
    else if(start_path_i < start_path_size){
      start_path = std::vector<Eigen::Vector3d>(start_path.begin(), start_path.begin() + start_path_i);
      start_state = start_path.back();
      start_path_both_end.push_back(start_path.front());
      start_path_both_end.push_back(start_state);
    }
    else{
      start_path_both_end.push_back(start_path.front());
      start_path_both_end.push_back(start_state);
    }
  }
  else{
    start_state = plan_start_state_XYTheta;
  }

  jps_planner_->plan(start_state, goal_state_);
  
  jps_planner_->getKinoNodeWithStartPath(start_path, if_forward, plan_start_state_VAJ, plan_start_state_OAJ);


  visualization_msgs::Marker marker;
  marker.header.frame_id = "world";
  marker.header.stamp = ros::Time::now();
  marker.ns = "jps_planner";
  marker.id = 0;
  marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  marker.action = visualization_msgs::Marker::ADD;
  marker.pose.position.x = 11;
  marker.pose.position.y = 8;
  marker.pose.position.z = 0;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;
  marker.scale.z = 0.5;
  marker.color.a = 1.0; // Don't forget to set the alpha!
  marker.color.r = 0.0;
  marker.color.g = 0.0;
  marker.color.b = 0.0;
  double search_time = (ros::Time::now()-current).toSec() * 1000.0;
  std::ostringstream out;
  out << std::fixed <<"JPS: \n"<< std::setprecision(2) << search_time<<" ms";
  marker.text = out.str();
  record_pub_.publish(marker);


  ROS_INFO("\033[40;36m jps_planner_ search time:%lf  \033[0m", (ros::Time::now()-current).toSec());

  return true;
}

void PlanTester::MPCPathPub(const double& traj_start_time){
  Eigen::MatrixXd initstate = msplanner_->get_current_iniState();
  Eigen::MatrixXd finState = msplanner_->get_current_finState();
  Eigen::MatrixXd finalInnerpoints = msplanner_->get_current_Innerpoints();
  Eigen::VectorXd finalpieceTime = msplanner_->get_current_finalpieceTime();
  Eigen::Vector3d iniStateXYTheta = msplanner_->get_current_iniStateXYTheta();

  carstatemsgs::Polynome polynome;
  polynome.header.frame_id = "world";
  polynome.header.stamp = ros::Time::now();
  polynome.init_p.x = initstate.col(0).x();
  polynome.init_p.y = initstate.col(0).y();
  polynome.init_v.x = initstate.col(1).x();
  polynome.init_v.y = initstate.col(1).y();
  polynome.init_a.x = initstate.col(2).x();
  polynome.init_a.y = initstate.col(2).y();
  polynome.tail_p.x = finState.col(0).x();
  polynome.tail_p.y = finState.col(0).y();
  polynome.tail_v.x = finState.col(1).x();
  polynome.tail_v.y = finState.col(1).y();
  polynome.tail_a.x = finState.col(2).x();
  polynome.tail_a.y = finState.col(2).y();

  if(plan_start_time_ < 0) polynome.traj_start_time = ros::Time::now();
  else polynome.traj_start_time = ros::Time(plan_start_time_);

  for(u_int i=0; i<finalInnerpoints.cols(); i++){
    geometry_msgs::Vector3 point;
    point.x = finalInnerpoints.col(i).x();
    point.y = finalInnerpoints.col(i).y();
    point.z = 0.0;
    polynome.innerpoints.push_back(point);
  }
  for(u_int i=0; i<finalpieceTime.size(); i++){
    polynome.t_pts.push_back(finalpieceTime[i]);
  }
  polynome.start_position.x = iniStateXYTheta.x();
  polynome.start_position.y = iniStateXYTheta.y();
  polynome.start_position.z = iniStateXYTheta.z();

  if(!msplanner_->if_standard_diff_){
    polynome.ICR.x = msplanner_->ICR_.x();
    polynome.ICR.y = msplanner_->ICR_.y();
    polynome.ICR.z = msplanner_->ICR_.z();
  }
  
  mpc_polynome_pub_.publish(polynome);
}


#endif