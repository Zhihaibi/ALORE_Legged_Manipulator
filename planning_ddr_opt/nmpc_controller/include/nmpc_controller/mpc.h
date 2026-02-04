#ifndef _MPC_H_
#define _MPC_H_

#include <vector>
#include <iostream>
#include <string>

#include <ros/ros.h>
#include <ros/package.h>

#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

#include "carstatemsgs/CarState.h"
#include "carstatemsgs/CarControl.h"
#include "std_msgs/Bool.h"
#include "geometry_msgs/PointStamped.h"

#include "carstatemsgs/Polynome.h"
#include "nmpc_controller/traj_anal.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include "sensor_msgs/PointCloud2.h"
#include "visualization_msgs/Marker.h"
#include "nav_msgs/Odometry.h"

#include <osqp/osqp.h>
#include <OsqpEigen/OsqpEigen.h>

#include <thread>

#include "nmpc_controller/mpc_wrapper.h"

using namespace Tracked_nmpc;

struct CarState{
    double x;
    double y;
    double theta;
    double vl;
    double vr;
};

struct CarICR{
    double xv;
    double yl;
    double yr;
};

struct TrajectoryPoint{
    double x;
    double y;
    double theta;
    double vl;
    double vr;
    double time;
};


enum STATE {
  kX = 0,
  kY = 1,
  kPsi = 2,
};


enum CONTROL {
  kVr = 0,
  kVl = 1,
};


enum ONLINEDATA {
  kXv = 0,
  kYr = 1,
  kYl = 2,
};

class MpcController{
    
    private:
        ros::NodeHandle nh_;

        // for mpc wrapper
        MpcWrapper mpc_wrapper_;
        Eigen::MatrixXd est_state_;
        Eigen::MatrixXd reference_states_;
        Eigen::MatrixXd reference_inputs_;
        Eigen::MatrixXd predicted_states_;
        Eigen::MatrixXd predicted_inputs_;
        Eigen::MatrixXd point_of_interest_;

        bool solve_from_scratch_;
        std::thread preparation_thread_;

        double timing_preparation_;
        double timing_feedback_;

        ros::Publisher predicted_path_pub_;
        ros::Publisher car_control_pub_;

        void run();
        // bool setStateEstimate(const CarState& state_estimate);
        // bool setReference(const std::vector<TrajectoryPoint>& reference_trajectory);
        bool publishPrediction(const Eigen::MatrixXd &predicted_states, const Eigen::MatrixXd &predicted_inputs, const ros::Time &call_time);
        void preparationThread();

        int N_;
        double dt_;

        ros::Timer cmd_timer_;
        double cmd_timer_rate_;
        void CmdCallback(const ros::TimerEvent& event);

        Eigen::Vector3d current_state_;
        ros::Subscriber odom_sub_;
        ros::Subscriber traj_sub_;

        // void OdomCallback(const carstatemsgs::CarState::ConstPtr& msg);
        void OdomCallback(const nav_msgs::Odometry::ConstPtr& msg);
        void TrajCallback(const carstatemsgs::Polynome::ConstPtr& msg);
        ros::Subscriber emergency_stop_sub_;

        ros::Publisher sequence_pub_;
        void sequencePub();

        ros::Publisher cmd_pub_;
        
        // FOR ICR
        ros::Subscriber ICR_sub_;
        CarICR car_icr_;
        void ICRCallback(const geometry_msgs::PointStamped::ConstPtr& msg);

        // debug
        ros::Publisher Ref_path_pub_;
        ros::Publisher cmd_path_pub_;
        ros::Publisher Ref_velocity_pub_;
        ros::Publisher Real_velocity_pub_;
        ros::Publisher Ref_path_marker_pub_;

        TrajAnal traj_;
        TrajAnal new_traj_;
        double new_traj_start_time_;

        double max_omega;
        double max_domega;
        // Maximum speed, minimum speed, maximum speed change per step, acceleration
        double max_speed;
        double min_speed;
        double max_accel;

        int delay_num_;

        double Last_plan_time = -1;

        // State variables
        // Whether the robot has localization
        bool has_odom;
        // Whether the trajectory has been received
        bool receive_traj_ = false;
        // Current robot state
        ros::Time now_state_time_;
        // Used to predict the future state of the robot
        // MPCState xopt[500];

        // Trajectory start time
        double start_time;
        // Trajectory duration
        double traj_duration; 
        
        // Whether the goal has been reached
        bool at_goal = false;
        // Maximum computation time, current condition for reaching the goal
        double max_mpc_time_;

        // Whether to use MPC
        bool use_mpc_;

        // Normalize the input angle to the range of -pi to pi
        void normlize_theta(double& th);

        // Smooth the yaw angle changes in xref to avoid 2pi jumps
        void smooth_yaw(void);
        
        // Send control commands
        void cmdPub();
        
        // Get a series of reference points for MPC
        void getRefPoints(const int T, double dt);

        void emergencyStop(const std_msgs::Bool::ConstPtr &msg);
        
    public:
        MpcController(const ros::NodeHandle &nh);

};



#endif