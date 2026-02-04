#ifndef _VISUALIZER_H_
#define _VISUALIZER_H_

#include <ros/ros.h>
#include <ros/package.h>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <map>

#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include "sensor_msgs/PointCloud2.h"
#include "visualization_msgs/MarkerArray.h"
#include "visualization_msgs/Marker.h"
#include "geometry_msgs/PoseStamped.h"
#include "nav_msgs/Path.h"
#include "tf/transform_datatypes.h"

#include "gcopter/trajectory.hpp"

// debug
#include <fstream>

class Visualizer
{
  private:
    ros::NodeHandle nh_;

    ros::Publisher kinoastarPubPCL;
    ros::Publisher kinoastarPubPath;
    ros::Publisher finalnodePubMarker;
    ros::Publisher meshPub;
    ros::Publisher edgePub;
    
  public:
    ros::Publisher mincoPathPath;
    ros::Publisher mincoPointMarker;
    Visualizer(ros::NodeHandle nh){
      nh_ = nh;
      kinoastarPubPCL = nh_.advertise<sensor_msgs::PointCloud2>("/visualizer/kinoastarPathPCL",10);
      kinoastarPubPath = nh_.advertise<nav_msgs::Path>("/visualizer/kinoastarPath",10);
      finalnodePubMarker = nh_.advertise<visualization_msgs::Marker>("/visualizer/finalnode",10);
      mincoPathPath = nh_.advertise<nav_msgs::Path>("/visualizer/mincoPath",10);
      mincoPointMarker = nh_.advertise<visualization_msgs::Marker>("/visualizer/mincoPoint",10);

      
    }

    ~Visualizer(){};

    // Only the end point
    void finalnodePub(const geometry_msgs::PoseStamped::ConstPtr &msg){
      visualization_msgs::Marker marker;
      marker.header.frame_id = "world";
      marker.ns = "finalnode";
      marker.lifetime = ros::Duration();
      marker.type = visualization_msgs::Marker::ARROW;
      marker.action = visualization_msgs::Marker::ADD;
      marker.scale.x = 0.3;
      marker.scale.y = 0.05;
      marker.scale.z = 0.05;
      marker.color.a = 0.3;
      marker.color.r = rand() / double(RAND_MAX);
      marker.color.g = rand() / double(RAND_MAX);
      marker.color.b = rand() / double(RAND_MAX);
      marker.header.stamp = ros::Time::now();
      marker.id = 0;
      marker.pose.position.x = msg->pose.position.x;
      marker.pose.position.y = msg->pose.position.y;
      marker.pose.position.z = 0.15;
      marker.pose.orientation.w = msg->pose.orientation.w;
      marker.pose.orientation.x = msg->pose.orientation.x;
      marker.pose.orientation.y = msg->pose.orientation.y;
      marker.pose.orientation.z = msg->pose.orientation.z;
      finalnodePubMarker.publish(marker);
    }

    // the start point and end point
    void finalnodePub(const Eigen::Vector3d &init_point, const Eigen::Vector3d &final_point){
      visualization_msgs::Marker marker;
      marker.header.frame_id = "world";
      marker.ns = "init_point";
      marker.lifetime = ros::Duration();
      marker.type = visualization_msgs::Marker::ARROW;
      marker.action = visualization_msgs::Marker::ADD;
      marker.scale.x = 0.15;
      marker.scale.y = 0.06;
      marker.scale.z = 0.06;
      marker.color.a = 0.8;
      // marker.color.r = rand() / double(RAND_MAX);
      // marker.color.g = rand() / double(RAND_MAX);
      // marker.color.b = rand() / double(RAND_MAX);
      marker.color.r = 0;
      marker.color.g = 1;
      marker.color.b = 0;
      marker.header.stamp = ros::Time::now();
      marker.id = 0;
      marker.pose.position.x = init_point.x();
      marker.pose.position.y = init_point.y();
      marker.pose.position.z = 0.02;      
      marker.pose.orientation = tf::createQuaternionMsgFromYaw(init_point.z());
      finalnodePubMarker.publish(marker);

      marker.ns = "final_point";
      // marker.color.r = rand() / double(RAND_MAX);
      // marker.color.g = rand() / double(RAND_MAX);
      // marker.color.b = rand() / double(RAND_MAX);
      marker.color.r = 1;
      marker.color.g = 0;
      marker.color.b = 0;
      marker.pose.position.x = final_point.x();
      marker.pose.position.y = final_point.y();
      marker.pose.position.z = 0.02;      
      marker.pose.orientation = tf::createQuaternionMsgFromYaw(final_point.z());
      finalnodePubMarker.publish(marker);
    }

    void initnodePub(const Eigen::Vector3d &point){
      visualization_msgs::Marker marker;
      marker.header.frame_id = "world";
      marker.ns = "initnode";
      marker.lifetime = ros::Duration();
      marker.type = visualization_msgs::Marker::ARROW;
      marker.action = visualization_msgs::Marker::ADD;
      marker.scale.x = 0.2;
      marker.scale.y = 0.04;
      marker.scale.z = 0.04;
      marker.color.a = 0.7;
      marker.color.r = 1;
      marker.color.g = 0;
      marker.color.b = 0;
      marker.header.stamp = ros::Time::now();
      marker.id = 0;
      marker.pose.position.x = point.x();
      marker.pose.position.y = point.y();
      marker.pose.position.z = 0.02;      
      marker.pose.orientation = tf::createQuaternionMsgFromYaw(point.z());
      finalnodePubMarker.publish(marker);
    }

    void mincoPathPub(const Trajectory<5, 2> &final_traj, const Eigen::Vector3d &start_state_XYTheta){
      double ini_x = start_state_XYTheta.x();
      double ini_y = start_state_XYTheta.y();

      double s1;
      int K = 50;
      int SamNumEachPart = 2 * K;
      double sumT = 0.0;

      int TrajNum = final_traj.getPieceNum();
      Eigen::VectorXd pieceTime = final_traj.getDurations();

      std::vector<Eigen::VectorXd> VecIntegralX(TrajNum);
      std::vector<Eigen::VectorXd> VecIntegralY(TrajNum);
      std::vector<Eigen::VectorXd> VecYaw(TrajNum);
      std::vector<Eigen::Vector2d> VecTrajFinalXY(TrajNum+1);
      VecTrajFinalXY[0] = Eigen::Vector2d(ini_x, ini_y);

      for(int i=0; i<TrajNum; i++){
        double step = pieceTime[i] / K;
        double halfstep = step / 2.0;
        double CoeffIntegral = pieceTime[i] / K / 6.0;

        Eigen::VectorXd IntegralX(K);IntegralX.setZero();
        Eigen::VectorXd IntegralY(K);IntegralY.setZero();
        Eigen::VectorXd Yaw(K);Yaw.setZero();
        s1 = 0.0;
        for(int j=0; j<=SamNumEachPart; j++){
          if(j%2 == 0){
            Eigen::Vector2d currPos = final_traj.getPos(s1+sumT);
            Eigen::Vector2d currVel = final_traj.getVel(s1+sumT);
            s1 += halfstep;
            if(j!=0){
              IntegralX[j/2-1] += CoeffIntegral * currVel.y() * cos(currPos.x());
              IntegralY[j/2-1] += CoeffIntegral * currVel.y() * sin(currPos.x());
              Yaw[j/2-1] = currPos.x();
            }
            if(j!=SamNumEachPart){
              IntegralX[j/2] += CoeffIntegral * currVel.y() * cos(currPos.x());
              IntegralY[j/2] += CoeffIntegral * currVel.y() * sin(currPos.x());
            }
          }
          else{
            Eigen::Vector2d currPos = final_traj.getPos(s1+sumT);
            Eigen::Vector2d currVel = final_traj.getVel(s1+sumT);
            s1 += halfstep;
            IntegralX[j/2] += 4.0 * CoeffIntegral * currVel.y() * cos(currPos.x());
            IntegralY[j/2] += 4.0 * CoeffIntegral * currVel.y() * sin(currPos.x());
          }
        }
        VecIntegralX[i] = IntegralX;
        VecIntegralY[i] = IntegralY;
        VecYaw[i] = Yaw;
        // VecTrajFinalXY[i+1] = Eigen::Vector2d(IntegralX[IntegralX.size()-1], IntegralY[IntegralX.size()-1]);
        sumT += pieceTime[i];
      }

      nav_msgs::Path path;
      path.header.frame_id = "world";
      path.header.stamp = ros::Time::now();
      Eigen::Vector2d pos(ini_x, ini_y);
      for(u_int i=0; i<VecIntegralX.size(); i++){
        for(u_int j=0; j<VecIntegralX[i].size(); j++){
          pos.x() += VecIntegralX[i][j];
          pos.y() += VecIntegralY[i][j];
          geometry_msgs::PoseStamped pose;
          pose.header.frame_id = "world";
          pose.header.stamp = ros::Time::now();
          pose.pose.position.x = pos.x();
          pose.pose.position.y = pos.y();
          pose.pose.position.z = 0.15;
          pose.pose.orientation = tf::createQuaternionMsgFromYaw(VecYaw[i][j]);
          path.poses.push_back(pose);
        }
      }
      mincoPathPath.publish(path);
      ROS_INFO("\033[40;33m iter real finStateXY:%f  %f  \033[0m", pos.x(), pos.y());
    }

};



#endif