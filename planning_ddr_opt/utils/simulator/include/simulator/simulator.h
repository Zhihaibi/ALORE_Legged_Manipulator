#ifndef _SIMULATOR_H_
#define _SIMULATOR_H_

#include "ros/forwards.h"
#include "visualization_msgs/MarkerArray.h"
#include <cmath>
#include <ros/ros.h>
#include <ros/package.h>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <random>

#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

#include <carstatemsgs/CarState.h>
#include <carstatemsgs/KinematicState.h>
#include <carstatemsgs/CarControl.h>
#include <carstatemsgs/SimulatedCarState.h>
#include <visualization_msgs/Marker.h>
#include <tf/transform_datatypes.h>

#include <geometry_msgs/Point.h>
#include <nav_msgs/Odometry.h>

class Simulation{
    private:
        ros::NodeHandle nh_;
        ros::Subscriber Pose_sub_;
        ros::Publisher Pose_pub_;
        ros::Publisher Rviz_pub_;
        ros::Publisher Rviz_Arrow_pub_;
        ros::Publisher KinematicState_pub_;
        ros::Publisher radar_range_pub_;

        ros::Publisher Pose_pose_pub_;
        ros::Publisher Pose_odom_pub_;

        ros::Subscriber Clear_traj_sub_;

        ros::Timer Pose_pub_timer_;
        double Pose_pub_rate_;
        ros::Timer Pose_odom_pub_timer_;
        double Pose_odom_pub_rate_;
        ros::Timer State_Propa_timer_;
        double State_Propa_rate_;

        ros::Time current_time_;
        Eigen::Vector3d current_XYTheta_;
        Eigen::Vector4d current_SVAJ_;
        Eigen::Vector4d current_YOAJ_;
        double current_vy_;

        double desired_v_;
        double desired_omega_;

        double length_;
        double width_;
        double height_;
        double wheel_base_;
        double tread_;
        double front_suspension_;
        double rear_suspension_;


        double max_v_;
        double min_v_;
        double max_omega_;
        double max_a_;
        double max_domega_;
        double max_centripetal_acc_;

        double offset_;
        bool if_add_noise_;
        double noise_stddev_;

        bool hrz_limited_;
        double hrz_laser_range_dgr_;
        double detection_range_;

        ros::Publisher accu_traj_pub_;
        visualization_msgs::Marker accu_traj_;



        ros::Subscriber PreciseEKFPram_sub_;
        double PreciseChi_;
        double PreciseChiConv_;

        ros::Publisher Simcarstate_pub_;
        ros::Publisher Control_pub_;
        ros::Subscriber Param_sub_;
        ros::Subscriber Conrtrol_sub_;

        double ICR_yr_;
        double ICR_yl_;
        double ICR_xv_;

    public:
        Simulation(ros::NodeHandle nh){
            nh_ = nh;
            
            double start_x, start_y, start_yaw;
            nh_.param<double>(ros::this_node::getName()+"/start_x",start_x,0.0);
            nh_.param<double>(ros::this_node::getName()+"/start_y",start_y,0.0);
            nh_.param<double>(ros::this_node::getName()+"/start_yaw",start_yaw,0.0);
            current_XYTheta_ << start_x, start_y, start_yaw;
            current_SVAJ_.setZero();
            current_YOAJ_.setZero();
            current_YOAJ_[0] = start_yaw;
            current_vy_ = 0.0;

            desired_v_ = 0.0;
            desired_omega_ = 0.0;

            nh_.param<double>(ros::this_node::getName()+"/length",length_,0.0);
            nh_.param<double>(ros::this_node::getName()+"/width",width_,0.0);
            nh_.param<double>(ros::this_node::getName()+"/height",height_,0.0);
            nh_.param<double>(ros::this_node::getName()+"/wheel_base",wheel_base_,0.0);
            nh_.param<double>(ros::this_node::getName()+"/tread",tread_,0.0);
            nh_.param<double>(ros::this_node::getName()+"/front_suspension",front_suspension_,0.0);
            nh_.param<double>(ros::this_node::getName()+"/front_suspension",rear_suspension_,0.0);
            offset_ = length_/2 - rear_suspension_;

            nh_.param<double>(ros::this_node::getName()+"/max_vel",max_v_,0.0);
            nh_.param<double>(ros::this_node::getName()+"/min_vel",min_v_,0.0);
            nh_.param<double>(ros::this_node::getName()+"/max_omega",max_omega_,0.0);
            nh_.param<double>(ros::this_node::getName()+"/max_acc",max_a_,0.0);
            nh_.param<double>(ros::this_node::getName()+"/max_domega",max_domega_,0.0);
            nh_.param<double>(ros::this_node::getName()+"/max_centripetal_acc",max_centripetal_acc_,0.0);

            int State_Propa_rate, Pose_pub_rate;
            nh_.param<int>(ros::this_node::getName()+"/State_Propa_rate",State_Propa_rate,50);
            nh_.param<int>(ros::this_node::getName()+"/Pose_pub_rate",Pose_pub_rate,50);
            State_Propa_rate_ = 1.0/State_Propa_rate;
            Pose_pub_rate_ = 1.0/Pose_pub_rate;

            int Pose_odom_pub_rate;
            nh_.param<int>(ros::this_node::getName()+"/Pose_odom_pub_rate",Pose_odom_pub_rate,50);
            Pose_odom_pub_rate_ = 1.0/Pose_odom_pub_rate;

            nh_.param<bool>(ros::this_node::getName()+"/if_add_noise",if_add_noise_,false);
            nh_.param<double>(ros::this_node::getName()+"/noise_stddev",noise_stddev_,0.01);

            nh_.param<bool>(ros::this_node::getName()+"/hrz_limited",hrz_limited_,false);
            nh_.param<double>(ros::this_node::getName()+"/hrz_laser_range_dgr",hrz_laser_range_dgr_,360.0);
            nh_.param<double>(ros::this_node::getName()+"/detection_range",detection_range_,10.0);

            Pose_sub_ = nh_.subscribe<carstatemsgs::CarState>("/simulation/PoseSub",1,&Simulation::PoseSubCallback,this,ros::TransportHints().unreliable());
            // Pose_sub_ = nh_.subscribe<carstatemsgs::CarState>("/simulation/PoseSub",1,&Simulation::PoseSubCallback,this);
            Pose_pub_ = nh_.advertise<carstatemsgs::CarState>("/simulation/PosePub",1);
            Pose_pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/simulation/PosePosePub",1);
            Pose_odom_pub_ = nh_.advertise<nav_msgs::Odometry>("/simulation/PoseOdomPub",1);

            Rviz_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("simulation/RvizCar",1);
            Rviz_Arrow_pub_ = nh_.advertise<visualization_msgs::Marker>("simulation/RvizCarArrow",1);
            KinematicState_pub_ = nh_.advertise<carstatemsgs::KinematicState>("/simulation/KinematicState",1);
            radar_range_pub_ = nh_.advertise<visualization_msgs::Marker>("simulation/RvizRadarRange",10);

            Clear_traj_sub_ = nh_.subscribe<geometry_msgs::PoseStamped>("/move_base_simple/goal",1,&Simulation::ClearTrajCallback,this);

            State_Propa_timer_ = nh_.createTimer(ros::Duration(State_Propa_rate_),&Simulation::StatePropaCallback, this);
            Pose_pub_timer_ = nh_.createTimer(ros::Duration(Pose_pub_rate_),&Simulation::PosePubCallback, this);
            Pose_odom_pub_timer_ = nh_.createTimer(ros::Duration(Pose_odom_pub_rate_),&Simulation::PoseOdomPubCallback, this);

            accu_traj_pub_ = nh_.advertise<visualization_msgs::Marker>("simulation/accu_traj",1);
            accu_traj_.header.frame_id = "world";
            accu_traj_.ns = "accu_traj";
            accu_traj_.lifetime = ros::Duration();
            accu_traj_.type = visualization_msgs::Marker::LINE_STRIP;
            accu_traj_.action = visualization_msgs::Marker::ADD;
            accu_traj_.color.a = 1;
            accu_traj_.color.r = 0.512;
            accu_traj_.color.g = 0.654;
            accu_traj_.color.b = 0.854; 
            accu_traj_.scale.x = 0.05;
            accu_traj_.pose.orientation.w = 1.0;
            accu_traj_.id = 0;

            nh_.param<double>(ros::this_node::getName()+"/tread",tread_,0.0);
            nh_.param<double>(ros::this_node::getName()+"/ICR_yr", ICR_yr_, 0.0);
            nh_.param<double>(ros::this_node::getName()+"/ICR_yl", ICR_yl_, 0.0);
            nh_.param<double>(ros::this_node::getName()+"/ICR_xv", ICR_xv_, 0.0);

            Control_pub_ = nh_.advertise<carstatemsgs::CarControl>("/simulation/ControlPub",1);
            Simcarstate_pub_ = nh_.advertise<carstatemsgs::SimulatedCarState>("/simulation/SimulatedCarState",1);

            Param_sub_ = nh_.subscribe<geometry_msgs::Point>("/simulation/change_param",1,&Simulation::changeParamCallback,this);

            Conrtrol_sub_ = nh_.subscribe<carstatemsgs::CarControl>("/simulation/ControlSub",1,&Simulation::ControlSubCallback,this);

            PreciseEKFPram_sub_ = nh_.subscribe<geometry_msgs::PointStamped>("/EFFEKF/EKF_ICR",1,&Simulation::EKFPreciseCallback,this);

            current_time_ = ros::Time::now();


        }

        void PoseSubCallback(const carstatemsgs::CarState::ConstPtr &msg){

            if(fabs(msg->v - current_SVAJ_[1]) > max_a_ * Pose_pub_rate_){
                current_SVAJ_.tail(3) << current_SVAJ_[1] + Pose_pub_rate_ * max_a_ * (msg->v - current_SVAJ_[1])/fabs(msg->v - current_SVAJ_[1]),
                                         msg->a, msg->js;
            }
            else{
                current_SVAJ_.tail(3) << msg->v, msg->a, msg->js;
            }
            if(fabs(msg->omega - current_YOAJ_[1]) > max_domega_ * Pose_pub_rate_){
                current_YOAJ_.tail(3) << current_YOAJ_[1] + Pose_pub_rate_ * max_domega_ * (msg->omega - current_YOAJ_[1])/fabs(msg->omega - current_YOAJ_[1]),
                                         msg->alpha, msg->jyaw;
            }
            else{
                current_YOAJ_.tail(3) << msg->omega, msg->alpha, msg->jyaw;
            }
            // Generally, only information at the speed level and above will be received
            current_SVAJ_.tail(3) << msg->v, msg->a, msg->js;
            current_YOAJ_.tail(3) << msg->omega, msg->alpha, msg->jyaw;
            if(if_add_noise_){
                std::random_device rd;
                std::mt19937 gen(rd());
                std::normal_distribution<double> distributionv(0.0, current_SVAJ_[1] * noise_stddev_);
                current_SVAJ_[1] += distributionv(gen);
                std::normal_distribution<double> distributiono(0.0, current_YOAJ_[1] * noise_stddev_);
                current_YOAJ_[1] += distributiono(gen);
            }

            current_time_ = msg->Header.stamp;
        }

        void ControlSubCallback(const carstatemsgs::CarControl::ConstPtr &msg){
            current_time_ = msg->Header.stamp;
            double left_wheel_ome = msg->left_wheel_ome;
            double right_wheel_ome = msg->right_wheel_ome;
            desired_v_ = (left_wheel_ome + right_wheel_ome) / 2.0 - (right_wheel_ome - left_wheel_ome)/(ICR_yl_ - ICR_yr_) * (ICR_yl_ + ICR_yr_)/2.0;
            current_vy_ = -(right_wheel_ome - left_wheel_ome) / (ICR_yl_ - ICR_yr_) * ICR_xv_;
            // std::cout<<"current_vy_: "<<current_vy_<<std::endl;
            desired_omega_ = (right_wheel_ome - left_wheel_ome) / (ICR_yl_ - ICR_yr_);
        }

        void StatePropaCallback(const ros::TimerEvent& event){
            
            if(fabs(current_SVAJ_[1] - desired_v_) >= Pose_pub_rate_ * max_a_){
                current_SVAJ_[1] += Pose_pub_rate_ * max_a_ * (desired_v_ - current_SVAJ_[1])/fabs(desired_v_ - current_SVAJ_[1]);
                current_SVAJ_[2] = max_a_ * (desired_v_ - current_SVAJ_[1])/fabs(desired_v_ - current_SVAJ_[1]);
            }
            else{
                current_SVAJ_[2] = (desired_v_ - current_SVAJ_[1])/Pose_pub_rate_;
                current_SVAJ_[1] = desired_v_;
                
            }
            if(fabs(current_YOAJ_[1] - desired_omega_) >= Pose_pub_rate_ * max_domega_){
                current_YOAJ_[1] += Pose_pub_rate_ * max_domega_ * (desired_omega_ - current_YOAJ_[1])/fabs(desired_omega_ - current_YOAJ_[1]);
                current_YOAJ_[2] = max_domega_ * (desired_omega_ - current_YOAJ_[1])/fabs(desired_omega_ - current_YOAJ_[1]);
            }
            else{
                current_YOAJ_[2] = (desired_omega_ - current_YOAJ_[1])/Pose_pub_rate_;
                current_YOAJ_[1] = desired_omega_;
            }

            // Currently, only speed is used for state transition
            current_XYTheta_.x() += current_SVAJ_[1]*State_Propa_rate_*cos(current_XYTheta_.z());
            current_XYTheta_.y() += current_SVAJ_[1]*State_Propa_rate_*sin(current_XYTheta_.z());
            current_XYTheta_.z() += current_YOAJ_[1]*State_Propa_rate_;

            current_XYTheta_.x() -= current_vy_*State_Propa_rate_*sin(current_XYTheta_.z());
            current_XYTheta_.y() += current_vy_*State_Propa_rate_*cos(current_XYTheta_.z());

            current_SVAJ_[0] += current_SVAJ_[1] * State_Propa_rate_;
            current_YOAJ_[0] += current_YOAJ_[1] * State_Propa_rate_;
        }

        void PoseOdomPubCallback(const ros::TimerEvent& event){
            geometry_msgs::PoseStamped pose;
            pose.header.frame_id = "world";
            pose.header.stamp = ros::Time::now();
            pose.pose.position.x = current_XYTheta_.x();
            pose.pose.position.y = current_XYTheta_.y();
            pose.pose.position.z = height_/2;
            tf::Quaternion q = tf::createQuaternionFromRPY(0, 0, current_XYTheta_.z());
            pose.pose.orientation.x = q.x();
            pose.pose.orientation.y = q.y();
            pose.pose.orientation.z = q.z();
            pose.pose.orientation.w = q.w();
            Pose_pose_pub_.publish(pose);

            nav_msgs::Odometry odom;
            odom.header.frame_id = "world";
            odom.header.stamp = ros::Time::now();
            odom.pose.pose.position.x = current_XYTheta_.x();
            odom.pose.pose.position.y = current_XYTheta_.y();
            odom.pose.pose.position.z = height_/2;
            odom.pose.pose.orientation.x = q.x();
            odom.pose.pose.orientation.y = q.y();
            odom.pose.pose.orientation.z = q.z();
            odom.pose.pose.orientation.w = q.w();
            Pose_odom_pub_.publish(odom);
        }

        void PosePubCallback(const ros::TimerEvent& event){
            carstatemsgs::CarState carstate;
            carstate.Header.frame_id = "world";
            carstate.Header.stamp = ros::Time::now();
            carstate.x = current_XYTheta_.x();
            carstate.y = current_XYTheta_.y();
            carstate.yaw = current_XYTheta_.z();

            carstate.s = current_SVAJ_[0];
            carstate.v = current_SVAJ_[1];
            carstate.a = current_SVAJ_[2];
            carstate.js = current_SVAJ_[3];


            carstate.omega = current_YOAJ_[1];
            carstate.alpha = current_YOAJ_[2];
            carstate.jyaw = current_YOAJ_[3];

            Pose_pub_.publish(carstate);

            carstatemsgs::SimulatedCarState simcarstate;
            simcarstate.Header.frame_id = "world";
            simcarstate.Header.stamp = ros::Time::now();
            simcarstate.x = current_XYTheta_.x();
            simcarstate.y = current_XYTheta_.y();
            simcarstate.yaw = current_XYTheta_.z();

            simcarstate.v = current_SVAJ_[1];
            simcarstate.omega = current_YOAJ_[1];

            simcarstate.vy = current_vy_;
            simcarstate.vx = current_SVAJ_[1];

            simcarstate.ICR_xv = ICR_xv_;
            simcarstate.ICR_yl = ICR_yl_;
            simcarstate.ICR_yr = ICR_yr_;

            Simcarstate_pub_.publish(simcarstate);

            carstatemsgs::CarControl carcontrol;
            carcontrol.Header.frame_id = "world";
            carcontrol.Header.stamp = ros::Time::now();
            carcontrol.left_wheel_ome = simcarstate.vx - current_YOAJ_[1] * ICR_yl_;
            carcontrol.right_wheel_ome = simcarstate.vx - current_YOAJ_[1] * ICR_yr_;
            Control_pub_.publish(carcontrol);


            carstatemsgs::KinematicState kinematicstate;
            kinematicstate.Header.frame_id = "world";
            kinematicstate.Header.stamp = ros::Time::now();
            kinematicstate.centripetal_acc = current_YOAJ_[1] * current_SVAJ_[1];
            kinematicstate.max_centripetal_acc = max_centripetal_acc_;
            kinematicstate.min_centripetal_acc = -max_centripetal_acc_;
            kinematicstate.moment = fabs(current_SVAJ_[1]) * max_omega_ + fabs(current_YOAJ_[1]) * max_v_;
            kinematicstate.max_moment = max_v_ * max_omega_;
            kinematicstate.min_moment = -max_v_ * max_omega_;
            KinematicState_pub_.publish(kinematicstate);

            Eigen::Vector2d offset;
            Eigen::Matrix2d R;
            double yaw = current_XYTheta_.z();
            R<<cos(yaw), -sin(yaw), sin(yaw), cos(yaw);

            visualization_msgs::MarkerArray rviz_marker_array;

            visualization_msgs::Marker marker;
            marker.header.frame_id = "world";
            marker.ns = "rvizCar";
            marker.lifetime = ros::Duration();
            marker.type = visualization_msgs::Marker::MESH_RESOURCE;
            marker.action = visualization_msgs::Marker::ADD;
            marker.color.a = 1;
            marker.color.r = 0.512;
            marker.color.g = 0.654;
            marker.color.b = 0.854;
            marker.header.stamp = ros::Time::now();

            // Car body
            marker.mesh_resource = "package://simulator/urdf/scout_mini.dae";
            // marker.mesh_resource = "package://simulator/urdf/b2z1.dae";
            
            // marker.mesh_resource = "package://simulator/urdf/track2.dae";
            // marker.mesh_resource = "package://simulator/urdf/yunjing.dae";
            offset << 0.246, 0.21; // here -> new  
            // offset << -0.05478, -0.15;
            // offset << 0.12, -0.01;
            marker.scale.x = 0.000666;
            marker.scale.y = 0.000666;
            marker.scale.z = 0.000666;
            // marker.scale.x = 0.000666*0.66;
            // marker.scale.y = 0.000666*0.66;
            // marker.scale.z = 0.000666*0.66;
            // marker.scale.x = 0.75;
            // marker.scale.y = 0.75;
            // marker.scale.z = 0.75;
            marker.id = 0;
            marker.pose.position.x = (R*offset).x() + current_XYTheta_.x();
            marker.pose.position.y = (R*offset).y() + current_XYTheta_.y();
            marker.pose.position.z = height_/2-0.32;
            marker.color.a = 0.8;
            marker.color.r = 0.3;
            marker.color.g = 0.3;
            marker.color.b = 0.3;
            tf::Quaternion q = tf::createQuaternionFromRPY(- M_PI / 2.0, M_PI, -3*M_PI/2 + yaw);
            // tf::Quaternion q = tf::createQuaternionFromRPY(- M_PI / 2.0, M_PI, yaw);
            // tf::Quaternion q = tf::createQuaternionFromRPY(0, 0, yaw);
            marker.pose.orientation.x = q.x();
            marker.pose.orientation.y = q.y();
            marker.pose.orientation.z = q.z();
            marker.pose.orientation.w = q.w();

            rviz_marker_array.markers.push_back(marker);

            Rviz_pub_.publish(rviz_marker_array);
            marker.mesh_resource.clear();

            // arrow
            marker.header.frame_id = "world";
            marker.ns = "rvizCarArrow";
            marker.lifetime = ros::Duration();
            marker.type = visualization_msgs::Marker::ARROW;
            marker.action = visualization_msgs::Marker::ADD;
            marker.scale.x = 0.2;
            marker.scale.y = 0.06;
            marker.scale.z = 0.06;
            marker.color.a = 1;
            marker.color.r = 0.512;
            marker.color.g = 0.654;
            marker.color.b = 0.854;
            marker.header.stamp = ros::Time::now();
            marker.id = 0;
            marker.pose.position.x = current_XYTheta_.x();
            marker.pose.position.y = current_XYTheta_.y();
            marker.pose.position.z = 0.175;
            marker.pose.orientation = tf::createQuaternionMsgFromYaw(yaw);
            Rviz_Arrow_pub_.publish(marker);

            static int i=0;
            if(i>=1000) i-=1000;
            else i++;
            geometry_msgs::Point p;
            p.x = current_XYTheta_.x();
            p.y = current_XYTheta_.y();
            // p.z = height_/2;
            p.z = 0.1;
            accu_traj_.points.push_back(p);


            Eigen::Vector3d low_speed_color;
            Eigen::Vector3d high_speed_color;
            std_msgs::ColorRGBA color;
            if(current_SVAJ_[1]>=0){
                color.r = std::min(pow(current_SVAJ_[1]/max_v_, 1.1), 1.0);
                color.g = 1 - color.r;
            }
            else if(min_v_ == 0){
                color.r = 0.0;
                color.g = 1.0;
            }
            else{
                color.r = std::min(pow(current_SVAJ_[1]/min_v_, 1.1), 1.0);
                color.g = 1 - color.r;
            }

            color.b = 0.0;
            color.a = 1.0;
            // color.r = i / 1000.0;
            // color.g = i / 1000.0;
            // color.b = 1.0;
            // color.a = 1.0;
            accu_traj_.colors.push_back(color);

            accu_traj_pub_.publish(accu_traj_);

            if(hrz_limited_){
                visualization_msgs::Marker RadarRange;
                RadarRange.header.frame_id = "world";
                RadarRange.ns = "RadarRange";
                RadarRange.action = visualization_msgs::Marker::DELETEALL;
                radar_range_pub_.publish(RadarRange);
    
                RadarRange.lifetime = ros::Duration();
                RadarRange.type = visualization_msgs::Marker::LINE_LIST;
                RadarRange.action = visualization_msgs::Marker::ADD;
                RadarRange.id = 0;
                RadarRange.scale.x = 0.02;
                RadarRange.color.a = 1;
                RadarRange.color.r = 0.0;
                RadarRange.color.g = 0.0;
                RadarRange.color.b = 0.0;
                RadarRange.header.stamp = ros::Time::now();
                geometry_msgs::Point p1, p2;
                p1.x = current_XYTheta_.x();
                p1.y = current_XYTheta_.y();
                p1.z = height_/2;
                p2.x = current_XYTheta_.x() + detection_range_ * cos(yaw + hrz_laser_range_dgr_/2.0/180.0*M_PI);
                p2.y = current_XYTheta_.y() + detection_range_ * sin(yaw + hrz_laser_range_dgr_/2.0/180.0*M_PI);
                p2.z = height_/2;
                RadarRange.points.push_back(p1);
                RadarRange.points.push_back(p2);
                p2.x = current_XYTheta_.x() + detection_range_ * cos(yaw - hrz_laser_range_dgr_/2.0/180.0*M_PI);
                p2.y = current_XYTheta_.y() + detection_range_ * sin(yaw - hrz_laser_range_dgr_/2.0/180.0*M_PI);
                p2.z = height_/2;
                RadarRange.points.push_back(p1);
                RadarRange.points.push_back(p2);
                radar_range_pub_.publish(RadarRange);

                RadarRange.points.clear();
                RadarRange.type = visualization_msgs::Marker::LINE_STRIP;
                RadarRange.action = visualization_msgs::Marker::ADD;
                RadarRange.id = 2;
                RadarRange.header.stamp = ros::Time::now();
                for (double angle = yaw - hrz_laser_range_dgr_/2.0/180.0*M_PI; angle <= yaw + hrz_laser_range_dgr_/2.0/180.0*M_PI; angle += 0.03) {
                    geometry_msgs::Point p;
                    p.x = current_XYTheta_.x() + detection_range_ * cos(angle);
                    p.y = current_XYTheta_.y() + detection_range_ * sin(angle);
                    p.z = height_/2;
                    RadarRange.points.push_back(p);
                }
                radar_range_pub_.publish(RadarRange);
            }

        }

        void ClearTrajCallback(const geometry_msgs::PoseStampedConstPtr &msg){
            accu_traj_.points.clear();
            accu_traj_.colors.clear();
        }

        void changeParamCallback(const geometry_msgs::Point::ConstPtr& msg){
            if(msg->x > 0){
                ROS_ERROR("error! msg->x > 0 || ICR_yr_ should smaller than 0");
            }
            if(msg->y < 0){
                ROS_ERROR("error! msg->y < 0 || ICR_yl_ should bigger than 0");
            }

            ICR_yr_ = msg->x;
            ICR_yl_ = msg->y;
            ICR_xv_ = msg->z;

        }

        void EKFPreciseCallback(const geometry_msgs::PointStamped::ConstPtr& msg){
            PreciseChi_ = msg->point.x;
            PreciseChiConv_ = msg->point.y;
        }

};

#endif