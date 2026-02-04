#include "nmpc_controller/mpc.h"
#include <nav_msgs/Path.h>
#include "tf/tf.h"
#include "tf/transform_datatypes.h"

MpcController::MpcController(const ros::NodeHandle &nh){

    nh_ = nh;


    nh.param("max_omega", max_omega, -1.0);
    nh.param("max_domega", max_domega, -1.0);
    nh.param("max_vel", max_speed, -1.0);
    nh.param("min_vel", min_speed, -1.0);
    nh.param("max_acc", max_accel, -1.0);
    nh.param("cmd_timer_rate", cmd_timer_rate_, 100.0);
    nh.param("max_mpc_time", max_mpc_time_, 10.0);
    nh.param("if_mpc", use_mpc_, true);

    nh.param("delay_num", delay_num_, 0);

    emergency_stop_sub_ = nh_.subscribe<std_msgs::Bool>("/planner/emergency_stop", 1, &MpcController::emergencyStop, this);

    cmd_timer_ = nh_.createTimer(ros::Duration(1.0/cmd_timer_rate_),&MpcController::CmdCallback, this);

    has_odom = false;
    receive_traj_ = false;


    double state_seq_res;
    int Integral_appr_resInt;
    nh.param("state_seq_res", state_seq_res, 1.0);
    nh.param("Integral_appr_resInt", Integral_appr_resInt, 10);
    traj_.setRes(state_seq_res, Integral_appr_resInt);
    new_traj_.setRes(state_seq_res, Integral_appr_resInt);

    traj_sub_ = nh_.subscribe<carstatemsgs::Polynome>("traj", 1, &MpcController::TrajCallback, this);

    std::cout<<"trajectory topic name: "<<traj_sub_.getTopic()<<std::endl;

    // odom_sub_ = nh_.subscribe<carstatemsgs::CarState>("odom", 1, &MpcController::OdomCallback, this);
    odom_sub_ = nh_.subscribe<nav_msgs::Odometry>("odom", 1, &MpcController::OdomCallback, this);
    sequence_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("sequence",1);
    cmd_pub_ = nh_.advertise<carstatemsgs::CarState>("cmd",1);

    Ref_path_pub_ = nh_.advertise<nav_msgs::Path>("Ref_path",10);
    cmd_path_pub_ = nh_.advertise<nav_msgs::Path>("cmd_path",10);
    Ref_path_marker_pub_ = nh_.advertise<visualization_msgs::Marker>("Ref_path_marker",10);

    Ref_velocity_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("Ref_velocity",10);
    Real_velocity_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("Real_velocity",10);

    ICR_sub_ = nh_.subscribe<geometry_msgs::PointStamped>("EKF_ICR", 1, &MpcController::ICRCallback, this);

    {
        geometry_msgs::PoseStamped ref_v;
        Ref_velocity_pub_.publish(ref_v);
    }

    est_state_ = Eigen::Matrix<double, kStateSize, 1>::Zero();
    reference_states_ = Eigen::Matrix<double, kStateSize, kSamples + 1>::Zero();
    reference_inputs_ = Eigen::Matrix<double, kInputSize, kSamples + 1>::Zero();
    predicted_states_ = Eigen::Matrix<double, kStateSize, kSamples + 1>::Zero();
    predicted_inputs_ = Eigen::Matrix<double, kInputSize, kSamples>::Zero();
    point_of_interest_ = Eigen::Matrix<double, 3, 1>::Zero();

    std::vector<double> Q;
    std::vector<double> R;
    nh.param("matrix_q", Q, std::vector<double>());
    nh.param("matrix_r", R, std::vector<double>());

    Eigen::MatrixXd Q_mat(kStateSize, kStateSize);
    Eigen::MatrixXd R_mat(kInputSize, kInputSize);
    Q_mat.setZero();
    R_mat.setZero();

    for (int i = 0; i < kStateSize; i++) {
        Q_mat(i, i) = Q[i];
    }
    for (int i = 0; i < kInputSize; i++) {
        R_mat(i, i) = R[i];
    }


    mpc_wrapper_.setCosts(Q_mat, R_mat);

    solve_from_scratch_ = true;
    preparation_thread_ = std::thread(&MpcWrapper::prepare, mpc_wrapper_);
    predicted_path_pub_ = nh_.advertise<nav_msgs::Path>("predicted_path", 1);

    N_ = kSamples;
    dt_ = mpc_wrapper_.getTimestep();

    car_control_pub_ = nh_.advertise<carstatemsgs::CarControl>("wheel_cmd", 1);


    start_time = -1;
}

// void MpcController::OdomCallback(const carstatemsgs::CarState::ConstPtr& msg){    
//     has_odom = true;

//     current_state_.x() = msg->x;
//     current_state_.y() = msg->y;
//     current_state_.z() = msg->yaw;

//     est_state_ = current_state_;

//     now_state_time_ = msg->Header.stamp;
// }

void MpcController::OdomCallback(const nav_msgs::Odometry::ConstPtr& msg){    
    has_odom = true;

    current_state_.x() = msg->pose.pose.position.x;
    current_state_.y() = msg->pose.pose.position.y;
    current_state_.z() = tf::getYaw(msg->pose.pose.orientation);

    est_state_ = current_state_;

    now_state_time_ = msg->header.stamp;
}

void MpcController::ICRCallback(const geometry_msgs::PointStamped::ConstPtr& msg){
    car_icr_.yr = msg->point.x;
    car_icr_.yl = msg->point.y;
    car_icr_.xv = msg->point.z;
}

void MpcController::TrajCallback(const carstatemsgs::Polynome::ConstPtr& msg){
    Eigen::Vector3d start_state(msg->start_position.x, msg->start_position.y, msg->start_position.z);
    Eigen::MatrixXd initstate(2, 3);
    initstate.col(0) << msg->init_p.x, msg->init_p.y;
    initstate.col(1) << msg->init_v.x, msg->init_v.y;
    initstate.col(2) << msg->init_a.x, msg->init_a.y;
    Eigen::MatrixXd finalstate(2, 3);
    finalstate.col(0) << msg->tail_p.x, msg->tail_p.y;
    finalstate.col(1) << msg->tail_v.x, msg->tail_v.y;
    finalstate.col(2) << msg->tail_a.x, msg->tail_a.y;

    Eigen::MatrixXd InnerPoints;
    InnerPoints.resize(2, msg->innerpoints.size());
    for(u_int i=0; i<msg->innerpoints.size(); i++){
        InnerPoints.col(i) << msg->innerpoints[i].x, msg->innerpoints[i].y;
    }
    Eigen::VectorXd t_pts(msg->t_pts.size());
    for(u_int i=0; i<msg->t_pts.size(); i++){
        t_pts[i] = msg->t_pts[i];
    }

    if(new_traj_.if_get_traj_){
        traj_ = new_traj_;
        traj_duration = traj_.get_traj_duration();
        start_time = new_traj_start_time_;
        new_traj_.if_get_traj_ = false;
    }

    new_traj_.setTraj(start_state, initstate, finalstate, InnerPoints, t_pts, Eigen::Vector3d(msg->ICR.x, msg->ICR.y, msg->ICR.z));

    sequencePub();
    new_traj_start_time_ = msg->traj_start_time.toSec();

    new_traj_.if_get_traj_ = true;

    // traj_duration = traj_.get_traj_duration();
    receive_traj_ = true;
    at_goal = false;

    ROS_INFO("get new traj!!");
    // start_time = msg->traj_start_time.toSec();
}

void MpcController::CmdCallback(const ros::TimerEvent& event){
    if (!has_odom || !receive_traj_)
        return;
    
    if (new_traj_.if_get_traj_ && ros::Time::now().toSec() > new_traj_start_time_){
        traj_ = new_traj_;
        traj_duration = traj_.get_traj_duration();
        start_time = new_traj_start_time_;
        new_traj_.if_get_traj_ = false;
    }

    if (at_goal){
        carstatemsgs::CarState cmd;
        cmd.Header.frame_id = "world";
        cmd.Header.stamp = ros::Time::now();
        cmd.v = 0.0;
        cmd.omega = 0.0;

        cmd.a = 0.0;
        cmd.alpha = 0.0;
        cmd.js = 0.0;
        cmd.jyaw = 0.0;
        cmd_pub_.publish(cmd);
        receive_traj_ = false;
        start_time = -1;
    
        carstatemsgs::CarControl car_control_cmd;
        cmd.Header.frame_id = "world";
        cmd.Header.stamp = ros::Time::now();
        car_control_cmd.right_wheel_ome = predicted_inputs_(kVr, delay_num_);
        car_control_cmd.left_wheel_ome = predicted_inputs_(kVl, delay_num_);
        car_control_pub_.publish(car_control_cmd);
    }
    else{
        if(use_mpc_){
            getRefPoints(N_, dt_);
            smooth_yaw();
            run();

            cmdPub();
        }
        else{
            carstatemsgs::CarState cmd;
            double t_cur = ros::Time::now().toSec() - start_time;
            
            if(t_cur > traj_duration){
                at_goal = true;
            }

            Eigen::Vector2d curr_v = traj_.getVstate(t_cur);
            Eigen::Vector2d curr_a = traj_.getAstate(t_cur);

            cmd.Header.frame_id = "world";
            cmd.Header.stamp = ros::Time::now();
            cmd.v = curr_v.y();
            cmd.omega = curr_v.x();

            cmd.a = curr_a.y();
            cmd.alpha = curr_a.x();
            cmd.js = 0.0;
            cmd.jyaw = 0.0;
            cmd_pub_.publish(cmd);
        }

    }
    sequencePub();
    
}


void MpcController::normlize_theta(double &th){
    while (th > M_PI) th -= 2 * M_PI;
    while (th < -M_PI) th += 2 * M_PI;
}

void MpcController::smooth_yaw(void)
{
    double dyaw = reference_states_(kPsi, 0) - est_state_(kPsi);

    while (dyaw >= M_PI / 2)
    {
        reference_states_(kPsi, 0) -= M_PI * 2;
        dyaw = reference_states_(kPsi, 0) - est_state_(kPsi);
    }
    while (dyaw <= -M_PI / 2)
    {
        reference_states_(kPsi, 0) += M_PI * 2;
        dyaw = reference_states_(kPsi, 0) - est_state_(kPsi);
    }

    for (int i = 0; i < N_; i++)
    {
        dyaw = reference_states_(kPsi, i + 1) - reference_states_(kPsi, i);
        while (dyaw >= M_PI / 2)
        {
            reference_states_(kPsi, i + 1) -= M_PI * 2;
            dyaw = reference_states_(kPsi, i + 1) - reference_states_(kPsi, i);
        }
        while (dyaw <= -M_PI / 2)
        {
            reference_states_(kPsi, i + 1) += M_PI * 2;
            dyaw = reference_states_(kPsi, i + 1) - reference_states_(kPsi, i);
        }
    }
}

void MpcController::emergencyStop(const std_msgs::Bool::ConstPtr &msg){
    carstatemsgs::CarState cmd;
    cmd.Header.frame_id = "world";
    cmd.Header.stamp = ros::Time::now();
    cmd.v = 0.0;
    cmd.omega = 0.0;

    cmd.a = 0.0;
    cmd.alpha = 0.0;
    cmd.js = 0.0;
    cmd.jyaw = 0.0;
    cmd_pub_.publish(cmd);

    receive_traj_ = false;
    start_time = -1;
}

void MpcController::run(){

    ros::Time call_time = ros::Time::now();

    const clock_t start = clock();
    preparation_thread_.join();

    // setStateEstimate(state_estimate);

    Eigen::Vector3d car_icr;
    car_icr(kXv) = car_icr_.xv;
    car_icr(kYr) = car_icr_.yr;
    car_icr(kYl) = car_icr_.yl;

    mpc_wrapper_.setICRParameters(car_icr);
    // setReference(reference_trajectory);

    mpc_wrapper_.setTrajectory(reference_states_, reference_inputs_);

    static const bool do_preparation_step(false);

    if (solve_from_scratch_) {
        ROS_INFO("Solving MPC with hover as initial guess.");
        mpc_wrapper_.solve(est_state_);
        solve_from_scratch_ = false;
    } else {
        mpc_wrapper_.update(est_state_, do_preparation_step);
    }
    mpc_wrapper_.getStates(predicted_states_);
    mpc_wrapper_.getInputs(predicted_inputs_);

    // std::cout<<"reference_states_: \n"<<reference_states_<<std::endl;
    // std::cout<<"reference_inputs_: \n"<<reference_inputs_<<std::endl;
    // std::cout<<"predicted_states_: \n"<<predicted_states_<<std::endl;
    // std::cout<<"predicted_inputs_: \n"<<predicted_inputs_<<std::endl;

    publishPrediction(predicted_states_, predicted_inputs_, call_time);


    // Start a thread to prepare for the next execution.
    preparation_thread_ = std::thread(&MpcController::preparationThread, this);

    // Timing
    const clock_t end = clock();
    timing_feedback_ = 0.9 * timing_feedback_ +
                        0.1 * double(end - start) / CLOCKS_PER_SEC;
    
    ROS_INFO_THROTTLE(1.0, "MPC Timing: Latency: %1.1f ms  |  Total: %1.1f ms",
                      timing_feedback_ * 1000, (timing_feedback_ + timing_preparation_) * 1000);

    // Return the input control command.
    // return updateControlCommand(predicted_states_.col(0),
    //                             predicted_inputs_.col(0),
    //                             call_time);
}

// bool MpcController::setStateEstimate(const CarState& state_estimate) {
//     est_state_(kX) = state_estimate.x;
//     est_state_(kY) = state_estimate.y;
//     est_state_(kPsi) = state_estimate.theta;
//     return true;
// }

// bool MpcController::setReference(const std::vector<TrajectoryPoint>& reference_trajectory) {
//     reference_states_.setZero();
//     reference_inputs_.setZero();

//     for (int i = 0; i < kSamples + 1; i++) {
//         reference_states_(kX, i) = reference_trajectory[i].x;
//         reference_states_(kY, i) = reference_trajectory[i].y;
//         reference_states_(kPsi, i) = reference_trajectory[i].theta;
//         reference_inputs_(kVl, i) = reference_trajectory[i].vl;
//         reference_inputs_(kVr, i) = reference_trajectory[i].vr;
//     }


//   return true;
// }

bool MpcController::publishPrediction(const Eigen::MatrixXd &predicted_states, const Eigen::MatrixXd &predicted_inputs, const ros::Time &call_time){
    nav_msgs::Path predicted_path;
    predicted_path.header.frame_id = "world";
    predicted_path.header.stamp = call_time;

    geometry_msgs::PoseStamped pose;
    for (int i = 0; i < kSamples + 1; i++) {
        pose.pose.position.x = predicted_states(kX, i);
        pose.pose.position.y = predicted_states(kY, i);
        pose.pose.position.z = 0;
        pose.pose.orientation = tf::createQuaternionMsgFromYaw(predicted_states(kPsi, i));
        predicted_path.poses.push_back(pose);
    }

    predicted_path_pub_.publish(predicted_path);

    return true;
}

void MpcController::preparationThread() {
  const clock_t start = clock();

  mpc_wrapper_.prepare();

  // Timing
  const clock_t end = clock();
  timing_preparation_ = 0.9 * timing_preparation_ +
                        0.1 * double(end - start) / CLOCKS_PER_SEC;
}


// important
void MpcController::getRefPoints(const int T, double dt){
    double t_cur = ros::Time::now().toSec() - start_time;
    int j = 0;

    geometry_msgs::PoseStamped ref_v;
    auto current_a = traj_.getAstate(std::min(ros::Time::now().toSec() - start_time, traj_duration));
    auto current_v = traj_.getVstate(std::min(ros::Time::now().toSec() - start_time, traj_duration));
    auto current_p = traj_.getPstate(std::min(ros::Time::now().toSec() - start_time, traj_duration));
    ref_v.header.stamp = ros::Time::now();
    ref_v.header.frame_id = "world";
    ref_v.pose.position.x = current_p.x();
    ref_v.pose.position.y = current_p.y();
    ref_v.pose.position.z = current_p.z();
    ref_v.pose.orientation.w = current_v.x();
    ref_v.pose.orientation.x = current_v.y();
    ref_v.pose.orientation.y = current_a.x();
    ref_v.pose.orientation.z = current_a.y();
    Ref_velocity_pub_.publish(ref_v);

    if (t_cur > traj_duration + 1.0){
        at_goal = true;
    }
    else{
        at_goal = false;
    }
    for (double temp_t = t_cur + dt; j <= T; j++, temp_t += dt){
        if (temp_t <= traj_duration){
            Eigen::Vector3d curP = traj_.getPstate(temp_t);
            Eigen::Vector2d curV = traj_.getVstate(temp_t);
            Eigen::Vector2d curA = traj_.getAstate(temp_t);

            reference_states_(kX, j) = curP.x();
            reference_states_(kY, j) = curP.y();
            reference_states_(kPsi, j) = curP.z();

            reference_inputs_(kVl, j) = curV.y() - curV.x()*car_icr_.yl;
            reference_inputs_(kVr, j) = curV.y() - curV.x()*car_icr_.yr;

            normlize_theta(reference_states_(kPsi, j));
        }
        else{
            Eigen::Vector3d curP = traj_.getPstate(traj_duration);
            Eigen::Vector2d curV = traj_.getVstate(traj_duration);
            Eigen::Vector2d curA = traj_.getAstate(traj_duration);

            reference_states_(kX, j) = curP.x();
            reference_states_(kY, j) = curP.y();
            reference_states_(kPsi, j) = curP.z();

            reference_inputs_(kVl, j) = 0.0;
            reference_inputs_(kVr, j) = 0.0;

            normlize_theta(reference_states_(kPsi, j));
        }
    }

    nav_msgs::Path path;
    path.header.frame_id = "world";
    path.header.stamp = ros::Time::now();
    geometry_msgs::PoseStamped pose;
    for (u_int i = 0; i < reference_states_.cols(); i++){
        pose.pose.position.x = reference_states_(kX, i);
        pose.pose.position.y = reference_states_(kY, i);
        pose.pose.position.z = 0;
        pose.pose.orientation = tf::createQuaternionMsgFromYaw(reference_states_(kPsi, i));
        path.poses.push_back(pose);
    }
    Ref_path_pub_.publish(path);
    visualization_msgs::Marker path_marker;
    path_marker.header.frame_id = "world";
    path_marker.header.stamp = ros::Time::now();
    path_marker.ns = "path_marker";
    path_marker.id = 0;
    path_marker.type = visualization_msgs::Marker::LINE_STRIP;
    path_marker.action = visualization_msgs::Marker::ADD;
    path_marker.pose.orientation.w = 1.0;
    path_marker.scale.x = 0.1;

    path_marker.color.r = 0.0;
    path_marker.color.g = 1.0;
    path_marker.color.b = 0.0;
    path_marker.color.a = 1.0;


    for (u_int i = 0; i < reference_states_.cols(); i++){
        geometry_msgs::Point point;
        point.x = reference_states_(kX, i);
        point.y = reference_states_(kY, i);
        point.z = 0;
        path_marker.points.push_back(point);
    }

    Ref_path_marker_pub_.publish(path_marker);
}

void MpcController::cmdPub(){
    carstatemsgs::CarControl cmd;
    cmd.Header.frame_id = "world";
    cmd.Header.stamp = ros::Time::now();
    cmd.right_wheel_ome = predicted_inputs_(kVr, delay_num_);
    cmd.left_wheel_ome = predicted_inputs_(kVl, delay_num_);
    car_control_pub_.publish(cmd);
}

void MpcController::sequencePub(){
    sensor_msgs::PointCloud2 globalMap_pcd;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr  colored_pcl_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZ> cloudMap;
    pcl::PointXYZRGB  pt;

    std::vector<Eigen::Vector4d> seq = traj_.get_state_sequence_();
    for(u_int i=0; i<seq.size(); i++){
        pt.x = seq[i].x();
        pt.y = seq[i].y();
        pt.z = 0.1;
        pt.r = 255;
        pt.g = 0;
        pt.b = 255;
        colored_pcl_ptr->points.push_back(pt); 
    }
    cloudMap.height = colored_pcl_ptr->points.size();
    cloudMap.width = 1;
    pcl::toROSMsg(*colored_pcl_ptr,globalMap_pcd);
    globalMap_pcd.header.stamp = ros::Time::now();
    globalMap_pcd.header.frame_id = "world";
    sequence_pub_.publish(globalMap_pcd);
}