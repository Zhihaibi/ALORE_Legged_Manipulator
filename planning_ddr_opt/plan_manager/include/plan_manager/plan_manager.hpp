#ifndef _PLAN_MANAGER_HPP_
#define _PLAN_MANAGER_HPP_

#include "plan_env/sdf_map.h"
#include "visualizer/visualizer.hpp"
#include "front_end/jps_planner/jps_planner.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/PoseArray.h"        // 添加这行
#include "std_msgs/Float32MultiArray.h"    // 添加 Float32MultiArray include
#include "back_end/optimizer.h"
#include "tf/tf.h"
#include "tf/transform_datatypes.h"
#include "carstatemsgs/CarState.h"
#include "carstatemsgs/Polynome.h"
#include "std_msgs/Bool.h"
#include "std_msgs/Int32MultiArray.h"       // 如果使用了 Int32MultiArray 也需要添加


#include <thread>
#include <nav_msgs/Odometry.h>


#include "plan_manager/hungarian.hpp"
#include "plan_manager/branch_and_bound.hpp"

enum StateMachine{
  INIT,
  IDLE,
  PLANNING,
  REPLAN,
  GOINGTOGOAL,
  EMERGENCY_STOP,
};

class PlanManager
{
  private:
    ros::NodeHandle nh_;

    std::shared_ptr<SDFmap> sdfmap_;
    std::shared_ptr<Visualizer> visualizer_;
    std::shared_ptr<MSPlanner> msplanner_;
    std::shared_ptr<JPS::JPSPlanner> jps_planner_;

    std::vector<Eigen::Vector3d> item_positions_;
    std::vector<Eigen::Vector3d> target_positions_;
    std::vector<Eigen::Vector3d> ordered_waypoints_;

    ros::Subscriber goal_sub_;
    ros::Subscriber current_state_sub_;
    ros::Subscriber robotpose_sub_;
    ros::Subscriber objectpose_sub_;
    ros::Timer main_thread_timer_;
    ros::Timer map_update_timer_;
    ros::Publisher cmd_pub_;
    ros::Publisher mpc_polynome_pub_;
    ros::Publisher emergency_stop_pub_;

    ros::Publisher record_pub_;

    ros::Subscriber task_plan_sub_;
    ros::Publisher task_order_pub_;

    ros::Time current_time_;
    Eigen::Vector3d robot_pose;
    Eigen::Vector2d object_pose;
    Eigen::Vector2d target_pose;
    Eigen::Vector2d last_object_pose;
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

    bool task_plan_finished = false;
    bool set_obs_done = false;
    bool going_item = false;
    bool going_target = false;
    Eigen::Vector2d goal_item;
    Eigen::Vector2d goal_target;
    int half_size = 0;

    // new added ============================== // 添加这行
    ros::Subscriber start_sub_;  // 订阅起始点
    bool have_start_;            // 是否有起始点
    Eigen::Vector3d start_state_; // 起始状态
    // ========================================

    bool if_fix_final_;
    Eigen::Vector3d final_state_;

    double replan_time_;
    
    double max_replan_time_;

    double predicted_traj_start_time_;

    StateMachine state_machine_ = StateMachine::INIT;
    bool if_dynamic_task = false;
    bool if_mix_task = true;
    bool get_object = false;

  public:
    PlanManager(ros::NodeHandle nh){
      nh_ = nh;
      
      sdfmap_ = std::make_shared<SDFmap>(nh);
      visualizer_ = std::make_shared<Visualizer>(nh);
      msplanner_ = std::make_shared<MSPlanner>(Config(ros::NodeHandle("~")), nh_, sdfmap_);
      jps_planner_ = std::make_shared<JPS::JPSPlanner>(sdfmap_, nh_);

      start_sub_ = nh_.subscribe<geometry_msgs::PoseStamped>("/planner_start_pose", 1, &PlanManager::start_callback, this); // 添加这行
      // goal_sub_ = nh_.subscribe<geometry_msgs::PoseStamped>("/move_base_simple/goal",1,&PlanManager::goal_callback,this);

      goal_sub_ = nh_.subscribe<geometry_msgs::PoseStamped>("/planner_goal_pose",1,&PlanManager::goal_callback,this);
      robotpose_sub_ = nh_.subscribe<std_msgs::Float32MultiArray>("/env_obs",1,&PlanManager::RobotPoseCallback,this);
      objectpose_sub_ = nh_.subscribe<std_msgs::Float32MultiArray>("/object_6d_pose",1,&PlanManager::ObjectPoseCallback,this);

      // current_state_sub_ = nh_.subscribe<carstatemsgs::CarState>("/simulation/PosePub",1,&PlanManager::GeometryCallback,this);
      current_state_sub_ = nh_.subscribe<nav_msgs::Odometry>("odom",1,&PlanManager::GeometryCallback,this);
      main_thread_timer_ = nh_.createTimer(ros::Duration(0.001),&PlanManager::MainThread, this);
      map_update_timer_ = nh_.createTimer(ros::Duration(0.05),&PlanManager::MapUpdateThread, this);
      cmd_pub_ = nh_.advertise<carstatemsgs::CarState>("/simulation/PoseSub",1);
      emergency_stop_pub_ = nh_.advertise<std_msgs::Bool>("/planner/emergency_stop",1);
      record_pub_ = nh_.advertise<visualization_msgs::Marker>("/planner/calculator_time",1);
      mpc_polynome_pub_ = nh_.advertise<carstatemsgs::Polynome>("traj", 1);

      have_geometry_ = false;
      have_goal_ = false;
      have_start_ = false;  // 添加这行

      task_plan_sub_ = nh_.subscribe<geometry_msgs::PoseArray>("/task_plan/poses", 1, &PlanManager::task_plan_callback, this);
      task_order_pub_ = nh_.advertise<std_msgs::Int32MultiArray>("/task_plan/results", 10);

      nh_.param<bool>("if_fix_final", if_fix_final_, false);
      if(if_fix_final_){
        nh_.param<double>("final_x", final_state_(0), 0.0);
        nh_.param<double>("final_y", final_state_(1), 0.0);
        nh_.param<double>("final_yaw", final_state_(2), 0.0);
      }

      nh_.param<double>("replan_time",replan_time_,10000.0);
      nh_.param<double>("max_replan_time", max_replan_time_, 1.0);

      state_machine_ = StateMachine::IDLE;

      loop_start_time_ = ros::Time::now();
    }

    ~PlanManager(){ 
      sdfmap_->~SDFmap();
      visualizer_->~Visualizer();
    }

    void printStateMachine(){
      if(state_machine_ == INIT) ROS_INFO("state_machine_ == INIT");
      if(state_machine_ == IDLE) ROS_INFO("state_machine_ == IDLE");
      if(state_machine_ == PLANNING) ROS_INFO("state_machine_ == PLANNING");
      if(state_machine_ == REPLAN) ROS_INFO("state_machine_ == REPLAN");
    }

    void GeometryCallback(const nav_msgs::Odometry::ConstPtr &msg){
      have_geometry_ = true;
      current_state_XYTheta_ << msg->pose.pose.position.x, msg->pose.pose.position.y, tf::getYaw(msg->pose.pose.orientation);
      current_state_VAJ_ << 0.0, 0.0, 0.0;
      current_state_OAJ_ << 0.0, 0.0, 0.0;
      current_time_ = msg->header.stamp;
    }

    void RobotPoseCallback(const std_msgs::Float32MultiArray::ConstPtr &msg){
      robot_pose << msg->data[0]-4.4, msg->data[1]+1.2, msg->data[3]; // real to rviz
      // if(if_dynamic_task && (robot_pose.head<2>()-object_pose).norm() < 2.0 && !get_object){
      //   paintSquare(object_pose, /*make_obs=*/false, 1.0);
      //   sdfmap_->updateESDF2d();
      //   get_object = true;
      // }
    }

    void ObjectPoseCallback(const std_msgs::Float32MultiArray::ConstPtr &msg){
      // Currently not used in planning
      float yaw = robot_pose.z();
      float x_base = msg->data[0] + 0.3f;
      float y_base = -msg->data[1];

      object_pose << x_base*std::cos(yaw)-y_base*std::sin(yaw)+robot_pose.x(), 
                  x_base*std::sin(yaw)+y_base*std::cos(yaw)+robot_pose.y();

      paintSquare(object_pose, /*make_obs=*/true, 0.3);
      get_object = false;
      sdfmap_->updateESDF2d();
      
      // print robot and object pose
      ROS_INFO("Robot Pose in world frame: (%.2f, %.2f, %.2f)", robot_pose.x(), robot_pose.y(), robot_pose.z());
      ROS_INFO("Object Pose in world frame: (%.2f, %.2f)", object_pose.x(), object_pose.y());
    }

    void task_plan_callback(const geometry_msgs::PoseArray::ConstPtr& msg) {
      if (task_plan_finished){
        return;
      }
      ROS_INFO("Received %zu item positions.", msg->poses.size());
      item_positions_.clear();
      target_positions_.clear();
      sdfmap_->updateESDF2d();

      // 检查数据数量是否为偶数（应该是items和targets数量相等）
      if (msg->poses.size() % 2 != 0) {
          ROS_ERROR("Invalid data format: total poses count should be even (items + targets)");
          return;
      }
      half_size = msg->poses.size() / 2;

      // 前半部分是 item_positions_ (items)
      for (int i = 0; i < half_size; i++) {
          const auto& pose = msg->poses[i];
          item_positions_.emplace_back(pose.position.x, pose.position.y, tf::getYaw(pose.orientation));
      }
      
      // 后半部分是 target_positions_ (targets)
      for (int i = half_size; i < msg->poses.size(); i++) {
          const auto& pose = msg->poses[i];
          target_positions_.emplace_back(pose.position.x, pose.position.y, tf::getYaw(pose.orientation));
      }
      // solvePathWithBranchAndBound();
      solvePathWithGreedy();

      // print item and target positions for verification
      ROS_INFO("Item Positions:");
      for (size_t i = 0; i < item_positions_.size(); ++i) {
          ROS_INFO("  Item %zu: (%.2f, %.2f, %.2f)", i, item_positions_[i](0), item_positions_[i](1), item_positions_[i](2));
      }
      ROS_INFO("Target Positions:");
      for (size_t i = 0; i < target_positions_.size(); ++i) {
          ROS_INFO("  Target %zu: (%.2f, %.2f, %.2f)", i, target_positions_[i](0), target_positions_[i](1), target_positions_[i](2));
      }
      
    }

    void solvePathWithBranchAndBound() {

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

        ROS_INFO("\n\n--- Received All Mission Data, Starting task Planning ---");

        ROS_INFO("Starting combined Branch and Bound optimization...");

        int num_tasks = item_positions_.size();
        
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


        // 2. Solve with the combined B&B solver
        BranchAndBoundCombined bnb_solver(all_dists, num_tasks);
        std::vector<int> best_path_indices;
        double final_cost = bnb_solver.solve(fixed_assignment, best_path_indices);

        if (final_cost >= std::numeric_limits<double>::max()) {
            ROS_ERROR("Combined B&B failed to find a valid solution.");
        }

        ROS_INFO("Combined B&B solution found with cost: %.2f", final_cost);

        ordered_waypoints_.clear();
        ordered_waypoints_.push_back(current_state_XYTheta_);

        // 4. Extract and publish the visit order
        std_msgs::Int32MultiArray task_order_msg;
        std::stringstream item_ss, target_ss;

        for (int idx : best_path_indices) {
            if (idx > 0 && idx <= num_tasks) { // item index (1 to n)
                int item_idx = idx - 1;
                task_order_msg.data.push_back(item_idx);
                ordered_waypoints_.push_back(item_positions_[item_idx]);
                item_ss << item_idx << " ";
            } else if (idx > num_tasks) { // Target index (n+1 to 2n)
                int target_idx = idx - (num_tasks + 1);
                ordered_waypoints_.push_back(target_positions_[target_idx]);
                task_order_msg.data.push_back(target_idx);
                item_ss << target_idx << " ";
            }
        }

        task_order_pub_.publish(task_order_msg);
        ROS_INFO("Published optimal item visit order: [ %s]", item_ss.str().c_str());

        double final_path_length = 0;
        for(size_t i = 1; i < ordered_waypoints_.size(); ++i) {
            final_path_length += (ordered_waypoints_[i-1] - ordered_waypoints_[i]).head(2).norm();
        }
        ROS_INFO("[B&B] Total planned path length: %.3f m", final_path_length);
        ROS_INFO("--- B&B Planning Finished ---");
        task_plan_finished = true;
    }

    void solvePathWithGreedy() {
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

      ROS_INFO("Starting Greedy task planning with JPS path cost (item-target-item-target...)");

      int num_tasks = item_positions_.size();
      std::vector<bool> item_visited(num_tasks, false);
      std::vector<bool> target_visited(num_tasks, false);
      std::vector<int> visit_order;

      Eigen::Vector3d current_pos = robot_pose;

      for (int step = 0; step < num_tasks; ++step) {
          // 1. 选最近未访问item (使用JPS路径代价)
          double min_item_cost = std::numeric_limits<double>::max();
          int min_item_idx = -1;
          for (int i = 0; i < num_tasks; i++) {
              if (!item_visited[i]) {
                  bool path_found = jps_planner_->plan(current_pos, item_positions_[i]);
                  double cost = path_found ? jps_planner_->getPathLength() : std::numeric_limits<double>::max();
                  if (cost < min_item_cost) {
                      min_item_cost = cost;
                      min_item_idx = i;
                  }
              }
          }
          if (min_item_idx == -1) {
              ROS_WARN("No reachable item found at step %d", step);
              break;
          }
          item_visited[min_item_idx] = true;
          current_pos = item_positions_[min_item_idx];
          visit_order.push_back(min_item_idx);
          ROS_INFO("Selected item %d with path cost: %.2f", min_item_idx, min_item_cost);

          // 2. 选最近未访问target (使用JPS路径代价)
          double min_target_cost = std::numeric_limits<double>::max();
          int min_target_idx = -1;
          for (int i = 0; i < num_tasks; i++) {
              if (!target_visited[i]) {
                  bool path_found = jps_planner_->plan(current_pos, target_positions_[i]);
                  double cost = path_found ? jps_planner_->getPathLength() : std::numeric_limits<double>::max();
                  if (cost < min_target_cost) {
                      min_target_cost = cost;
                      min_target_idx = i;
                  }
              }
          }
          if (min_target_idx == -1) {
              ROS_WARN("No reachable target found at step %d", step);
              break;
          }
          target_visited[min_target_idx] = true;
          current_pos = target_positions_[min_target_idx];
          visit_order.push_back(num_tasks + min_target_idx);
          ROS_INFO("Selected target %d with path cost: %.2f", min_target_idx, min_target_cost);
      }

      // 发布顺序
      std_msgs::Int32MultiArray greedy_order_msg;
      std::stringstream ss;
      for (int idx : visit_order) {
          if (idx >= 0 && idx < num_tasks) {
              greedy_order_msg.data.push_back(idx);
              ss << idx << " ";
          } else if (idx >= num_tasks) {
              int target_idx = idx - num_tasks;
              greedy_order_msg.data.push_back(target_idx);
              ss << target_idx << " ";
          }
      }
      task_order_pub_.publish(greedy_order_msg);
      ROS_INFO("Published greedy visit order (JPS-based): [ %s]", ss.str().c_str());
      task_plan_finished = true;
  }

    void goal_callback(const geometry_msgs::PoseStamped::ConstPtr &msg){
      // Ignore the given goal at runtime, commenting out this check may cause unexpected bugs
      // Especially when there is no re-planning
      if(state_machine_ != StateMachine::IDLE){
        ROS_ERROR("Haven't reached the goal yet!!");
        return;
      }
      // ROS_INFO("\n\n\n\n\n\n\n\n");
      // ROS_INFO("---------------------------------------------------------------");
      // ROS_INFO("---------------------------------------------------------------");

      ROS_INFO("get goal!");
      state_machine_ = StateMachine::IDLE;
      have_goal_ = true;
      goal_state_<<msg->pose.position.x, msg->pose.position.y, tf::getYaw(msg->pose.orientation);
      if(if_fix_final_) goal_state_ = final_state_;
      ROS_INFO_STREAM("goal state: " << goal_state_.transpose());

      // ROS_INFO("---------------------------------------------------------------");
      // ROS_INFO("---------------------------------------------------------------");
      // ROS_INFO("\n\n\n\n\n\n\n\n");
    }


    void start_callback(const geometry_msgs::PoseStamped::ConstPtr &msg){
      if(state_machine_ != StateMachine::IDLE){
          ROS_ERROR("System is busy, cannot set new start point!");
          return;
      }
      
      ROS_INFO("Get start point!");
      have_start_ = true;
      start_state_ << msg->pose.position.x, msg->pose.position.y, tf::getYaw(msg->pose.orientation);
      ROS_INFO_STREAM("start state: " << start_state_.transpose());
    }

    void paintSquare (const Eigen::Vector2d& c, bool make_obs, double half_size=0.4)
    {
      for (double x = c.x() - half_size; x <= c.x() + half_size; x += sdfmap_->grid_interval_)
        for (double y = c.y() - half_size; y <= c.y() + half_size; y += sdfmap_->grid_interval_)
          make_obs ? sdfmap_->setObs(Eigen::Vector2d(x,y)) : sdfmap_->setFree(Eigen::Vector2d(x,y));
    }

    void paintBox (const Eigen::Vector2d&c, bool make_obs)
    {
      for (double x=c.x()-0.25; x<=c.x()+0.25; x+=sdfmap_->grid_interval_)
        for (double y=c.y()-0.15; y<=c.y()+0.15; y+=sdfmap_->grid_interval_)
          make_obs ? sdfmap_->setObs(Eigen::Vector2d(x,y)) : sdfmap_->setFree(Eigen::Vector2d(x,y));
    }

    void paintTable (const Eigen::Vector2d&c, bool make_obs)
    {
      for (double x=c.x()-0.5; x<=c.x()+0.5; x+=sdfmap_->grid_interval_)
        for (double y=c.y()-0.25; y<=c.y()+0.25; y+=sdfmap_->grid_interval_)
          make_obs ? sdfmap_->setObs(Eigen::Vector2d(x,y)) : sdfmap_->setFree(Eigen::Vector2d(x,y));
    }

    void paintChair (const Eigen::Vector2d&c, bool make_obs)
    {
      for (double x=c.x()-0.2; x<=c.x()+0.2; x+=sdfmap_->grid_interval_)
        for (double y=c.y()-0.2; y<=c.y()+0.2; y+=sdfmap_->grid_interval_)
          make_obs ? sdfmap_->setObs(Eigen::Vector2d(x,y)) : sdfmap_->setFree(Eigen::Vector2d(x,y));
    }

    void MapUpdateThread(const ros::TimerEvent& event){
      if (!have_geometry_){
        return;
      }
      if(!have_geometry_){
        ROS_ERROR("no geometry info!");
        return;
      }

      if (task_plan_finished && !set_obs_done){
        // items -> OBS
        if (if_mix_task)
        {
          paintBox(item_positions_[0].head<2>(), /*make_obs=*/true);
          paintTable(item_positions_[1].head<2>(), /*make_obs=*/true);
          paintChair(item_positions_[2].head<2>(), /*make_obs=*/true);
        }
        else
        {
          for (const auto& p : item_positions_) 
            paintSquare(p.head<2>(), /*make_obs=*/true, 0.5); // 0.3 for box
        }
        sdfmap_->updateESDF2d();
        set_obs_done = true;
      }

      if (set_obs_done){
        double dist_to_item = (robot_pose.head<2>() - goal_item).norm();
        if (going_item)
        {
          ROS_INFO("Goal item position: (%.2f, %.2f)", goal_item.x(), goal_item.y());
          ROS_INFO("Current xy: (%.2f, %.2f)", robot_pose.x(), robot_pose.y());
          ROS_INFO("Distance to current item: %.2f m", dist_to_item);
          if (dist_to_item < 2.0)
          {
            paintSquare(goal_item, /*make_obs=*/false, 0.5);
            sdfmap_->updateESDF2d();
            ROS_INFO("Item reached, unlocking the area for passing.");
          }
        }

        double dist_to_target = (robot_pose.head<2>() - goal_target).norm();
        if (going_target)
        {
          ROS_INFO("Goal target position: (%.2f, %.2f)", goal_target.x(), goal_target.y());
          ROS_INFO("Current xy: (%.2f, %.2f)", robot_pose.x(), robot_pose.y());
          ROS_INFO("Distance to current target: %.2f m", dist_to_target);
          if (dist_to_target < 0.5)
          {
            paintSquare(goal_target, /*make_obs=*/true, 0.3);
            sdfmap_->updateESDF2d();
            ROS_INFO("Target reached, locking the area.");
          }
          
        }
      }
    }

    void MainThread(const ros::TimerEvent& event){

      if(!have_geometry_ || !have_goal_ || !have_start_) return;

      if(state_machine_ == StateMachine::IDLE || 
          ((state_machine_ == StateMachine::PLANNING||state_machine_ == StateMachine::REPLAN) 
            && (ros::Time::now() - loop_start_time_).toSec() > replan_time_)){
        loop_start_time_ = ros::Time::now();
        double current = loop_start_time_.toSec();
        // start new plan
        if(state_machine_ == StateMachine::IDLE){
          state_machine_ = StateMachine::PLANNING;
          plan_start_time_ = -1;
          predicted_traj_start_time_ = -1;

          // 使用订阅到的起始点而不是当前位置
          plan_start_state_XYTheta = start_state_;
          plan_start_state_VAJ = Eigen::Vector3d::Zero(); // 起始速度为零
          plan_start_state_OAJ = Eigen::Vector3d::Zero(); // 起始角速度为零
        } 
        // Use predicted distance for replanning in planning state
        else if(state_machine_ == StateMachine::PLANNING || state_machine_ == StateMachine::REPLAN){
          
          if(((current_state_XYTheta_ - goal_state_).head(2).squaredNorm() + fmod(fabs((plan_start_state_XYTheta - goal_state_)[2]), 2.0 * M_PI)*0.02 < 1.0) ||
             msplanner_->final_traj_.getTotalDuration() < max_replan_time_){
            state_machine_ = StateMachine::GOINGTOGOAL;
            return;
          }

          state_machine_ = StateMachine::REPLAN;

          predicted_traj_start_time_ = current + max_replan_time_ - plan_start_time_;
          msplanner_->get_the_predicted_state(predicted_traj_start_time_, plan_start_state_XYTheta, plan_start_state_VAJ, plan_start_state_OAJ);

        } 
        
        // ROS_INFO("\033[32;40m \n\n\n\n\n-------------------------------------start new plan------------------------------------------ \033[0m");
        
        visualizer_->finalnodePub(plan_start_state_XYTheta, goal_state_);
        ROS_INFO("init_state_: %.10f  %.10f  %.10f", plan_start_state_XYTheta(0), plan_start_state_XYTheta(1), plan_start_state_XYTheta(2));
        ROS_INFO("goal_state_: %.10f  %.10f  %.10f", goal_state_(0), goal_state_(1), goal_state_(2));

        // std::cout<<"<arg name=\"start_x_\" value=\""<< plan_start_state_XYTheta(0) <<"\"/>"<<std::endl;
        // std::cout<<"<arg name=\"start_y_\" value=\""<< plan_start_state_XYTheta(1) <<"\"/>"<<std::endl;
        // std::cout<<"<arg name=\"start_yaw_\" value=\""<< plan_start_state_XYTheta(2) <<"\"/>"<<std::endl;
        // std::cout<<"<arg name=\"final_x_\" value=\""<< goal_state_(0) <<"\"/>"<<std::endl;
        // std::cout<<"<arg name=\"final_y_\" value=\""<< goal_state_(1) <<"\"/>"<<std::endl;
        // std::cout<<"<arg name=\"final_yaw_\" value=\""<< goal_state_(2) <<"\"/>"<<std::endl;

        // std::cout<<"plan_start_state_VAJ: "<<plan_start_state_VAJ.transpose()<<std::endl;
        // std::cout<<"plan_start_state_OAJ: "<<plan_start_state_OAJ.transpose()<<std::endl;

        // ROS_INFO("<arg name=\"start_x_\" value=\"%f\"/>", plan_start_state_XYTheta(0));
        // ROS_INFO("<arg name=\"start_y_\" value=\"%f\"/>", plan_start_state_XYTheta(1));
        // ROS_INFO("<arg name=\"start_yaw_\" value=\"%f\"/>", plan_start_state_XYTheta(2));
        // ROS_INFO("<arg name=\"final_x_\" value=\"%f\"/>", goal_state_(0));
        // ROS_INFO("<arg name=\"final_y_\" value=\"%f\"/>", goal_state_(1));
        // ROS_INFO("<arg name=\"final_yaw_\" value=\"%f\"/>", goal_state_(2));

        // ROS_INFO_STREAM("plan_start_state_VAJ: " << plan_start_state_VAJ.transpose());
        // ROS_INFO_STREAM("plan_start_state_OAJ: " << plan_start_state_OAJ.transpose());

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

        if (set_obs_done && !if_dynamic_task)
        {
          Eigen::Vector2d goal_xy = goal_state_.head<2>();

          double dist_item, dist_target;
          goal_item   = nearest_point(item_positions_, goal_xy, dist_item);
          goal_target = nearest_point(target_positions_, goal_xy, dist_target);
          // ROS_INFO("Distance to nearest item: %.2f m, target: %.2f m", dist_item, dist_target);

          // Determine the next goal: item or target
          if (dist_item < dist_target) 
          {
            // paintSquare(goal_item, /*make_obs=*/false, 0.5);
            // sdfmap_->updateESDF2d();
            // paintSquare(goal_item, /*make_obs=*/true, 0.1);
            // sdfmap_->updateESDF2d();
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
        // ROS_INFO("\033[41;37m all of front end time:%f \033[0m", (ros::Time::now()-astar_start_time).toSec());

        // optimizer
        bool result = msplanner_->minco_plan(jps_planner_->flat_traj_);
        if(!result){
          return;
        }

        // ROS_INFO("\033[43;32m all of plan time:%f \033[0m", (ros::Time::now().toSec()-current));

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

        if (going_item && !if_dynamic_task)
        {
          double dist_to_target;
          Eigen::Vector2d now_target = nearest_point(target_positions_, robot_pose.head<2>(), dist_to_target);
          if (dist_to_target < 0.3){
            paintSquare(now_target, /*make_obs=*/true, 0.2);
            sdfmap_->updateESDF2d();
            ROS_INFO("Target reached, locking the area as obstacle.");
          }
        }
      }

      if((ros::Time::now() - Traj_start_time_).toSec() >= Traj_total_time_){
        state_machine_ = StateMachine::IDLE;
        have_goal_ = false;
        have_start_ = false;  // Reset start point status
      }
    }

    bool findJPSRoad(){
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
      marker.pose.position.x = 17.5;
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

    void MPCPathPub(const double& traj_start_time){
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

  };

#endif