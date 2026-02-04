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
#include "std_msgs/Bool.h"
#include "carstatemsgs/Polynome.h"
#include "mpc_controller/traj_anal.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include "sensor_msgs/PointCloud2.h"
#include "visualization_msgs/Marker.h"
#include "nav_msgs/Odometry.h"

#include <osqp/osqp.h>
#include <OsqpEigen/OsqpEigen.h>

class MPCState{
    public:
        double x = 0;
        double y = 0;
        double v = 0;
        double theta = 0;
        double w = 0;
};

class TrajPoint{
    public:
        double x = 0;
        double y = 0;
        double v = 0;
        double a = 0;
        double theta = 0;
        double w = 0;
};

class MpcController{
    
    private:
        ros::NodeHandle nh_;

        ros::Timer cmd_timer_;
        double cmd_timer_rate_;
        void CmdCallback(const ros::TimerEvent& event);
        
        ros::Subscriber odom_sub_;
        ros::Subscriber traj_sub_;
        // void OdomCallback(const carstatemsgs::CarState::ConstPtr& msg);
        void OdomCallback(const nav_msgs::Odometry::ConstPtr& msg);
        void TrajCallback(const carstatemsgs::Polynome::ConstPtr& msg);
        ros::Subscriber emergency_stop_sub_;

        ros::Publisher sequence_pub_;
        void sequencePub();

        ros::Publisher cmd_pub_;
        
        // debug
        ros::Publisher Ref_path_pub_;
        ros::Publisher cmd_path_pub_;
        ros::Publisher Ref_velocity_pub_;
        ros::Publisher Real_velocity_pub_;
        ros::Publisher Ref_path_marker_pub_;

        TrajAnal traj_;
        TrajAnal new_traj_;
        double new_traj_start_time_;

        // One of the convergence conditions: if the change in control quantity during the iteration is less than this value, it is considered converged
        double du_th;
        // One of the convergence conditions: if the number of iterations is greater than this value, it is considered converged
        int max_iter;
        // Time step of MPC prediction
        double dt;
        // Number of time steps of MPC prediction
        int T;
        // Due to calculation time delay, send the control quantity of the delay_num-th step, generally the value is calculation time divided by step length
        int delay_num;
        // Tracking error weights, respectively x y v yaw
        std::vector<double> Q;
        // Control quantity weights, the purpose is to minimize the control quantity, respectively v w
        std::vector<double> R;
        // Control quantity change weights, the purpose is to minimize the change in control quantity, respectively v w
        std::vector<double> Rd;
        // Kinematic constraints
        // Maximum angular velocity, angular acceleration, maximum change in angular velocity per step
        double max_omega;
        double max_domega;
        double max_comega;
        // Maximum speed, minimum speed, maximum change in speed per step, acceleration
        double max_speed;
        double min_speed;
        double max_cv;
        double max_accel;

        // MPC linear state transition function matrix
        Eigen::MatrixXd A;
        Eigen::MatrixXd B;
        Eigen::VectorXd C;

        // MPC state matrix
        MPCState xbar[500];
        // Reference state
        Eigen::MatrixXd xref;
        // Reference input
        Eigen::MatrixXd dref;
        // Output buffer
        Eigen::MatrixXd output;
        // Store the last output as the initial value for MPC
        Eigen::MatrixXd last_output;
        // MPC calculation results
        std::vector<Eigen::Vector2d> output_buff;

        Eigen::Vector2d current_output;
        double Last_plan_time = -1;

        // State variables
        // Whether the robot has localization
        bool has_odom;
        // Whether the trajectory has been received
        bool receive_traj_ = false;
        // Current robot state
        MPCState now_state;
        ros::Time now_state_time_;
        // Used to predict the future state of the robot
        MPCState xopt[500];

        // Trajectory start time
        double start_time;
        // Trajectory duration
        double traj_duration; 
        
        // Whether the goal has been reached
        bool at_goal = false;
        std::vector<TrajPoint> P;

        // Maximum calculation time, current condition for reaching the goal
        double max_mpc_time_;

        // Whether to use MPC
        bool use_mpc_;

        // Normalize the input angle to the range of -pi to pi
        void normlize_theta(double& th);

        // Smooth the yaw angle changes in xref to avoid 2pi jumps
        void smooth_yaw(void);

        // Get the parameters of the linear model based on the input state
        void getLinearModel(const MPCState& s);

        // Get the output state based on the input state and control quantity
        void stateTrans(MPCState& s, double a, double yaw_dot);

        // Get all states based on the input state and control quantity b
        void predictMotion(void);
        void predictMotion(MPCState* b);

        // MPC model for speed control
        void solveMPCV(void);

        // Output the solution of the MPC problem
        void getCmd(void);
        
        // Publish control commands
        void cmdPub();
        
        // Get a series of reference points for MPC // TBD
        void getRefPoints(const int T, double dt);

        void emergencyStop(const std_msgs::Bool::ConstPtr &msg);

    public:
        MpcController(const ros::NodeHandle &nh);

};



#endif