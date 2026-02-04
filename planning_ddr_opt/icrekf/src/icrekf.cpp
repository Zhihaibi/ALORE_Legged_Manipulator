#include "icrekf/icrekf.h"
#include "Eigen/Eigenvalues"
#include "nav_msgs/Odometry.h"

void ICREKF::PoseSubCallback(const carstatemsgs::SimulatedCarState::ConstPtr &msg){

    if(!if_update_) return;

    if(!get_state_){
        current_state_ << msg->x, msg->y, msg->yaw;
        current_state_VVXVY_ << msg->v, msg->vx, msg->vy;
        current_state_omega_ = msg->omega;
        current_time_ = msg->Header.stamp;

        x_.head(3) = current_state_;

        get_state_ = true;
        return;
    }
    if(Pose_sub_Reduce_count_ < Pose_sub_Reduce_frequency_){
        Pose_sub_Reduce_count_++;
        return;
    }
    else{
        Pose_sub_Reduce_count_ -= Pose_sub_Reduce_frequency_;
    }
// std::cout<<"PoseSubCallback"<<std::endl;
    current_state_ << msg->x, msg->y, msg->yaw;
    current_state_VVXVY_ << msg->v, msg->vx, msg->vy;
    current_state_omega_ = msg->omega;

    while(current_state_[2] - x_[2] > M_PI) current_state_[2] -= 2 * M_PI;
    while(current_state_[2] - x_[2] < -M_PI) current_state_[2] += 2 * M_PI;

    current_time_ = msg->Header.stamp;
// std::cout<<"current_state_: "<<current_state_.transpose()<<std::endl;
    get_update_x(x_, conv_, current_state_);
}



void ICREKF::PoseOdomSubCallback(const nav_msgs::Odometry::ConstPtr &msg){
    if(!if_update_) return;

    if(!get_state_){
        current_state_ << msg->pose.pose.position.x, msg->pose.pose.position.y, tf::getYaw(msg->pose.pose.orientation);
        current_state_VVXVY_ << msg->twist.twist.linear.x, msg->twist.twist.linear.y, msg->twist.twist.angular.z;
        current_state_omega_ = msg->twist.twist.angular.z;
        current_time_ = msg->header.stamp;
        
        x_.head(3) = current_state_;

        get_state_ = true;
        return;
    }
    if(Pose_sub_Reduce_count_ < Pose_sub_Reduce_frequency_){
        Pose_sub_Reduce_count_++;
        return;
    }
    else{
        Pose_sub_Reduce_count_ -= Pose_sub_Reduce_frequency_;
    }

    current_state_ << msg->pose.pose.position.x, msg->pose.pose.position.y, tf::getYaw(msg->pose.pose.orientation);
    current_state_VVXVY_ << msg->twist.twist.linear.x, msg->twist.twist.linear.y, msg->twist.twist.angular.z;
    current_state_omega_ = msg->twist.twist.angular.z;

    while(current_state_[2] - x_[2] > M_PI) current_state_[2] -= 2 * M_PI;
    while(current_state_[2] - x_[2] < -M_PI) current_state_[2] += 2 * M_PI;

    current_time_ = msg->header.stamp;
    get_update_x(x_, conv_, current_state_);
}


void ICREKF::ControlSubCallback(const carstatemsgs::CarControl::ConstPtr &msg){
    if(!if_update_) return;

    if(!get_u_){
        current_u_ << msg->left_wheel_ome, msg->right_wheel_ome;
        current_u_time_ = msg->Header.stamp;
        get_u_ = true;
        
        return;
    }
    if(!get_state_) return;
// std::cout<<"ControlSubCallback"<<std::endl;
    pre_u_duration_ = (msg->Header.stamp - current_u_time_).toSec();
    get_forecast_x(x_, conv_, current_u_, pre_u_duration_);

    current_u_ << msg->left_wheel_ome, msg->right_wheel_ome;
    // std::cout<<"current_u_: "<<current_u_.transpose()<<std::endl;
    current_u_time_ = msg->Header.stamp;

    if(start_time_ < 0 && current_u_.squaredNorm()>0.1) start_time_ = ros::Time::now().toSec();
}


void ICREKF::get_forecast_x(Eigen::VectorXd& _x, Eigen::MatrixXd& _conv, const Eigen::Vector2d& input_u, const double& u_duration){

    if(!if_update_) return;

    double x = _x[0];
    double y = _x[1];
    double psi = _x[2];
    double yr = _x[3];
    double yl = _x[4];
    double xv = _x[5];

    double vl = input_u.x();
    double vr = input_u.y();


    _x[0] = x + u_duration * ((vr*yl-vl*yr)/(yl-yr)*cos(psi) + (vr-vl)*xv/(yl-yr)*sin(psi));
    _x[1] = y + u_duration * ((vr*yl-vl*yr)/(yl-yr)*sin(psi) - (vr-vl)*xv/(yl-yr)*cos(psi));
    _x[2] = psi + u_duration * (vr-vl)/(yl-yr);

    Eigen::MatrixXd F(6, 6);
    F(0, 0) = 1;
    F(1, 0) = 0;
    F(2, 0) = u_duration * (-(vr*yl-vl*yr)/(yl-yr)*sin(psi) + (vr-vl)*xv/(yl-yr)*cos(psi));
    F(3, 0) = u_duration * (-vl*cos(psi)/(yl-yr) + ((vr*yl-vl*yr)*cos(psi) + (vr-vl)*xv*sin(psi))/(yl-yr)/(yl-yr));
    F(4, 0) = u_duration * (vr*cos(psi)/(yl-yr) - ((vr*yl-vl*yr)*cos(psi) + (vr-vl)*xv*sin(psi))/(yl-yr)/(yl-yr));
    F(5, 0) = u_duration * (vr-vl)/(yl-yr)*sin(psi);

    F(0, 1) = 0;
    F(1, 1) = 1;
    F(2, 1) = u_duration * ((vr*yl-vl*yr)/(yl-yr)*cos(psi) + (vr-vl)*xv/(yl-yr)*sin(psi));
    F(3, 1) = u_duration * (-vl*sin(psi)/(yl-yr) + ((vr*yl-vl*yr)*sin(psi) - (vr-vl)*xv*cos(psi))/(yl-yr)/(yl-yr));
    F(4, 1) = u_duration * (vr*sin(psi)/(yl-yr) - ((vr*yl-vl*yr)*sin(psi) - (vr-vl)*xv*cos(psi))/(yl-yr)/(yl-yr));
    F(5, 1) = u_duration * -(vr-vl)/(yl-yr)*cos(psi);

    F(0, 2) = 0;
    F(1, 2) = 0;
    F(2, 2) = 1;
    F(3, 2) = u_duration * (vr-vl)/(yl-yr)/(yl-yr);
    F(4, 2) = u_duration * -(vr-vl)/(yl-yr)/(yl-yr);
    F(5, 2) = 0;

    F(0, 3) = 0;
    F(1, 3) = 0;
    F(2, 3) = 0;
    F(3, 3) = 1;
    F(4, 3) = 0;
    F(5, 3) = 0;

    F(0, 4) = 0;
    F(1, 4) = 0;
    F(2, 4) = 0;
    F(3, 4) = 0;
    F(4, 4) = 1;
    F(5, 4) = 0;

    F(0, 5) = 0;
    F(1, 5) = 0;
    F(2, 5) = 0;
    F(3, 5) = 0;
    F(4, 5) = 0;
    F(5, 5) = 1;

    // Eigen::MatrixXd F(6, 6);
    // F(0, 0) = 1;
    // F(1, 0) = 0;
    // F(2, 0) =  (-(vr*yl-vl*yr)/(yl-yr)*sin(psi) + (vr-vl)*xv/(yl-yr)*cos(psi));
    // F(3, 0) =  (-vl*cos(psi)/(yl-yr) + ((vr*yl-vl*yr)*cos(psi) + (vr-vl)*xv*sin(psi))/(yl-yr)/(yl-yr));
    // F(4, 0) =  (vr*cos(psi)/(yl-yr) - ((vr*yl-vl*yr)*cos(psi) + (vr-vl)*xv*sin(psi))/(yl-yr)/(yl-yr));
    // F(5, 0) =  (vr-vl)/(yl-yr)*sin(psi);

    // F(0, 1) = 0;
    // F(1, 1) = 1;
    // F(2, 1) = ((vr*yl-vl*yr)/(yl-yr)*cos(psi) - (vr-vl)*xv/(yl-yr)*sin(psi));
    // F(3, 1) = (-vl*sin(psi)/(yl-yr) + ((vr*yl-vl*yr)*sin(psi) - (vr-vl)*xv*cos(psi))/(yl-yr)/(yl-yr));
    // F(4, 1) = (vr*sin(psi)/(yl-yr) - ((vr*yl-vl*yr)*sin(psi) + (vr-vl)*xv*cos(psi))/(yl-yr)/(yl-yr));
    // F(5, 1) = -(vr-vl)/(yl-yr)*cos(psi);

    // F(0, 2) = 0;
    // F(1, 2) = 0;
    // F(2, 2) = 1;
    // F(3, 2) = (vr-vl)/(yl-yr)/(yl-yr);
    // F(4, 2) = -(vr-vl)/(yl-yr)/(yl-yr);
    // F(5, 2) = 0;

    // F(0, 3) = 0;
    // F(1, 3) = 0;
    // F(2, 3) = 0;
    // F(3, 3) = 1;
    // F(4, 3) = 0;
    // F(5, 3) = 0;

    // F(0, 4) = 0;
    // F(1, 4) = 0;
    // F(2, 4) = 0;
    // F(3, 4) = 0;
    // F(4, 4) = 1;
    // F(5, 4) = 0;

    // F(0, 5) = 0;
    // F(1, 5) = 0;
    // F(2, 5) = 0;
    // F(3, 5) = 0;
    // F(4, 5) = 0;
    // F(5, 5) = 1;

    // _conv = F * _conv * F.transpose() + L_ * u_duration  * Q_ * L_.transpose() * u_duration;
    _conv = F.transpose() * _conv * F + L_ * u_duration  * Q_ * L_.transpose() * u_duration;

    return;
}

void ICREKF::get_update_x(Eigen::VectorXd& _x, Eigen::MatrixXd& _conv,  const Eigen::Vector3d& current_state){

    if(!if_update_) return;

    Eigen::MatrixXd K_k = _conv * H_.transpose() * (H_ * _conv * H_.transpose() + M_ * R_ *  M_.transpose()).inverse();
// std::cout<<"get_update_x"<<std::endl;
    _x = _x + K_k * (current_state - H_ * _x);
// std::cout<<"x: "<<_x.transpose()<<std::endl;
    _conv  = (Eigen::MatrixXd::Identity(6, 6) - K_k * H_) * _conv;

    return;

}


void ICREKF::state_pub_timer_callback(const ros::TimerEvent& event){
    geometry_msgs::PointStamped ps;
    ps.header.frame_id = "base";
    ps.header.stamp = ros::Time::now();
    // ps.point.x = x_[0];
    // ps.point.y = x_[1];
    // ps.point.z = x_[2];
    // state_XYTheta_pub_.publish(ps);

    nav_msgs::Odometry odom;
    odom.header.frame_id = "world";
    odom.header.stamp = ros::Time::now();
    odom.pose.pose.position.x = x_[0];
    odom.pose.pose.position.y = x_[1];
    odom.pose.pose.position.z = 0;
    tf::Quaternion q = tf::createQuaternionFromYaw(x_[2]);
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();
    state_XYTheta_pub_.publish(odom);
    

    // geometry_msgs::PoseStamped ps1;
    // ps1.header.frame_id = "world";
    // ps1.header.stamp = ros::Time::now();
    // ps1.pose.position.x = x_[0];
    // ps1.pose.position.y = x_[1];
    // ps1.pose.position.z = 0;
    // ps1.pose.orientation = tf::createQuaternionMsgFromYaw(x_[2]);
    // state_XYTheta_pub_.publish(ps1);

    ps.point.x = x_[3];
    ps.point.y = x_[4];
    ps.point.z = x_[5];
    state_ICR_pub_.publish(ps);
    
    ps.point.x = conv_(3, 3);
    ps.point.z = conv_(4, 4);
    ps.point.y = conv_(5, 5);
    ALL_ICR_eigenvalues_pub_.publish(ps);

    ps.point.x = conv_(0, 0);
    ps.point.y = conv_(1, 1);
    ps.point.z = conv_(2, 2);
    ICR_eigenvalues_pub_.publish(ps);

    if(!if_yr_conver_ && fabs(x_[3] - yr_standard_)/fabs(yr_standard_) < 0.01){
        if(index_yr_standard_ ++ >10){
            if_yr_conver_ = true;
            std::cout<<"ICREKF    alpha_l conver!!!!"<<std::endl;
            std::cout<< "converge time: "<<ros::Time::now().toSec() - start_time_<<" s \n\n";
        }
    }
    else{
        index_yr_standard_ = 0;
    }

    if(!if_yl_conver_ && fabs(x_[4] - yl_standard_)/fabs(yl_standard_) < 0.01){
        if(index_yl_standard_ ++ >10){
            if_yl_conver_ = true;
            std::cout<<"ICREKF    alpha_r conver!!!!"<<std::endl;
            std::cout<< "converge time: "<<ros::Time::now().toSec() - start_time_<<" s \n\n";
        }
    }
    else{
        index_yl_standard_ = 0;
    }

    if(!if_xv_conver_ && fabs(x_[5] - xv_standard_)/fabs(xv_standard_) < 0.01){
        if(index_xv_standard_ ++ >10){
            if_xv_conver_ = true;
            std::cout<<"ICREKF   xv conver!!!!"<<std::endl;
            std::cout<< "converge time: "<<ros::Time::now().toSec() - start_time_<<" s \n\n";
        }
    }
    else{
        index_xv_standard_ = 0;
    }

    if(get_state_ && get_u_){
        if(fabs(current_state_omega_)>1e-1){
            ps.point.x = filter_yl_->filter((current_state_VVXVY_.y() - current_u_.x())/current_state_omega_);
            // ps.point.x = (current_state_VVXVY_.y() -current_u_.x())/current_state_omega_;
        }
        else{
            ps.point.x = 0;
        }
        if(fabs(current_state_omega_)>1e-1){
            ps.point.y = filter_yr_->filter((current_state_VVXVY_.y() - current_u_.y())/current_state_omega_);
            // ps.point.y = (current_state_VVXVY_.y() -current_u_.y())/current_state_omega_;
        }
        else{
            ps.point.y = 0;
        }
        if(fabs(current_state_omega_)>1e-1){
            ps.point.z = filter_xv_->filter(-current_state_VVXVY_.z()/current_state_omega_);
            // ps.point.z = -current_state_VVXVY_.z()/current_state_omega_;
        }
        else{
            ps.point.z = 0;
        }
        simple_state_ICR_pub_.publish(ps);

    }
    
// std::cout<<"state_pub_timer_callback"<<std::endl;
}