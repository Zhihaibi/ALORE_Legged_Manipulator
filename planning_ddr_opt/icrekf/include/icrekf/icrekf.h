#ifndef _ICREKF_H_
#define _ICREKF_H_

#include <ros/ros.h>
#include <ros/package.h>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

#include <carstatemsgs/CarState.h>
#include <carstatemsgs/CarControl.h>
#include <carstatemsgs/SimulatedCarState.h>

#include <geometry_msgs/PointStamped.h>

#include "tf/transform_datatypes.h"

#include <geometry_msgs/PointStamped.h>
#include <nav_msgs/Odometry.h>


class FirstOrderFilter {
private:
    double cutoffFrequency;
    double samplingFrequency;
    double a;
    double b;
    double previousOutput;

public:
    FirstOrderFilter(double cutoff, double sampling) {
        cutoffFrequency = cutoff;
        samplingFrequency = sampling;
        a = exp(-2 * M_PI * cutoffFrequency / samplingFrequency);
        b = 1 - a;
        previousOutput = 0.0;
    }

    double filter(double input) {
        double output = b * input + a * previousOutput;
        previousOutput = output;
        return output;
    }
};


class ICREKF{
    private:
        ros::NodeHandle nh_;

        bool get_state_;
        Eigen::Vector3d current_state_;
        Eigen::Vector3d current_state_VVXVY_;
        double current_state_omega_;
        ros::Time current_time_;
        ros::Time current_u_time_;
        double pre_u_duration_;

        bool get_u_;
        Eigen::Vector2d current_u_;

        // Eigen::Vector3d ICRs_;

        Eigen::VectorXd x_;
        Eigen::MatrixXd conv_;
        
        Eigen::MatrixXd Q_;
        Eigen::MatrixXd L_;
        Eigen::MatrixXd H_;
        Eigen::MatrixXd M_;
        Eigen::MatrixXd R_;

        ros::Subscriber Pose_sub_;
        int Pose_sub_Reduce_frequency_;
        int Pose_sub_Reduce_count_;
        ros::Subscriber Pose_odom_sub_;

        ros::Subscriber control_sub_;


        double state_pub_frequency_, state_pub_rate_;
        ros::Timer state_pub_timer_;
        ros::Publisher state_XYTheta_pub_;
        ros::Publisher state_ICR_pub_;
        ros::Publisher simple_state_ICR_pub_;
        ros::Publisher ICR_eigenvalues_pub_;
        ros::Publisher ALL_ICR_eigenvalues_pub_;


        std::shared_ptr<FirstOrderFilter> filter_yl_;
        std::shared_ptr<FirstOrderFilter> filter_yr_;
        std::shared_ptr<FirstOrderFilter> filter_xv_;


        double yr_standard_, yl_standard_, xv_standard_;
        bool if_yr_conver_, if_yl_conver_, if_xv_conver_;
        int index_yr_standard_, index_yl_standard_, index_xv_standard_;

        bool if_update_;

        double start_time_ = -1;

    public:
        ICREKF(ros::NodeHandle nh){
            nh_ = nh;

            // Pose_sub_ = nh_.subscribe<carstatemsgs::SimulatedCarState>("/ref_pose", 1, &ICREKF::PoseSubCallback, this);
            Pose_odom_sub_ = nh_.subscribe<nav_msgs::Odometry>("/odom", 1, &ICREKF::PoseOdomSubCallback, this);
            nh_.param<int>(ros::this_node::getName()+"/Pose_sub_Reduce_frequency_", Pose_sub_Reduce_frequency_, 10);
            Pose_sub_Reduce_count_ = 0;
            control_sub_ = nh_.subscribe<carstatemsgs::CarControl>("/control", 1, &ICREKF::ControlSubCallback, this);

            nh_.param<double>(ros::this_node::getName()+"/state_pub_frequency", state_pub_frequency_, 1.0);
            state_pub_rate_ = 1.0 / state_pub_frequency_;
            std::cout<<state_pub_frequency_<<std::endl;
            state_pub_timer_ = nh_.createTimer(ros::Duration(state_pub_rate_), &ICREKF::state_pub_timer_callback, this);
            state_XYTheta_pub_ = nh_.advertise<nav_msgs::Odometry>("EKF_XYTheta", 1);
            state_ICR_pub_ = nh_.advertise<geometry_msgs::PointStamped>("EKF_ICR", 1);
            simple_state_ICR_pub_ = nh_.advertise<geometry_msgs::PointStamped>("Simple_EKF_ICR", 1);
            ICR_eigenvalues_pub_ = nh_.advertise<geometry_msgs::PointStamped>("EKF_ICR_eigenvalues", 1);
            ALL_ICR_eigenvalues_pub_ = nh_.advertise<geometry_msgs::PointStamped>("ALL_EKF_ICR_eigenvalues", 1);

            filter_yl_ = std::make_shared<FirstOrderFilter>(1, 1.0/state_pub_rate_);
            filter_yr_ = std::make_shared<FirstOrderFilter>(1, 1.0/state_pub_rate_);
            filter_xv_ = std::make_shared<FirstOrderFilter>(1, 1.0/state_pub_rate_);

            nh_.param<double>(ros::this_node::getName()+"/yr_standard", yr_standard_, 0.15);
            nh_.param<double>(ros::this_node::getName()+"/yl_standard", yl_standard_, 0.15);
            nh_.param<double>(ros::this_node::getName()+"/xv_standard", xv_standard_, 0);
            if_yr_conver_ = false;
            if_yl_conver_ = false;
            if_xv_conver_ = false;
            index_yr_standard_ = 0;
            index_yl_standard_ = 0;
            index_xv_standard_ = 0;

            Q_.resize(6, 6);
            Q_.setZero();
            nh_.param<double>(ros::this_node::getName()+"/Q_x", Q_(0,0), 0.2);
            nh_.param<double>(ros::this_node::getName()+"/Q_y", Q_(1,1), 0.2);
            nh_.param<double>(ros::this_node::getName()+"/Q_psi", Q_(2,2), 0.314);
            nh_.param<double>(ros::this_node::getName()+"/Q_yr", Q_(3,3), 0.01);
            nh_.param<double>(ros::this_node::getName()+"/Q_yl", Q_(4,4), 0.01);
            nh_.param<double>(ros::this_node::getName()+"/Q_xv", Q_(5,5), 0.01);
            Q_ = Q_*Q_;

            R_.resize(3, 3);
            R_.setZero();
            nh_.param<double>(ros::this_node::getName()+"/R_x", R_(0,0), 0.01);
            nh_.param<double>(ros::this_node::getName()+"/R_y", R_(1,1), 0.01);
            nh_.param<double>(ros::this_node::getName()+"/R_psi", R_(2,2), 0.0157);
            R_ = R_*R_;
            
            L_.resize(6, 6);
            L_.setIdentity();
            // L_ = L_ * delta_t_;

            H_.resize(3, 6);
            H_.setZero();
            H_.block(0, 0, 3, 3).setIdentity();

            M_.resize(3, 3);
            M_.setIdentity();


            x_.resize(6);
            nh_.param<double>(ros::this_node::getName()+"/init_x_yr", x_[3], 0.01);
            nh_.param<double>(ros::this_node::getName()+"/init_x_yl", x_[4], 0.01);
            nh_.param<double>(ros::this_node::getName()+"/init_x_xv", x_[5], 0.01);
            conv_.resize(6, 6);
            conv_.setZero();

            nh_.param<bool>(ros::this_node::getName()+"/if_update", if_update_, true);
        
            get_state_ = false;
            get_u_ = false;

        }

        void PoseSubCallback(const carstatemsgs::SimulatedCarState::ConstPtr &msg);
        void ControlSubCallback(const carstatemsgs::CarControl::ConstPtr &msg);
        void PoseOdomSubCallback(const nav_msgs::Odometry::ConstPtr &msg);

        void get_forecast_x(Eigen::VectorXd& _x, Eigen::MatrixXd& _conv, const Eigen::Vector2d& input_u, const double& u_duration);

        void get_update_x(Eigen::VectorXd& _x, Eigen::MatrixXd& _conv,  const Eigen::Vector3d& current_state);
        
        void state_pub_timer_callback(const ros::TimerEvent& event);
};

#endif