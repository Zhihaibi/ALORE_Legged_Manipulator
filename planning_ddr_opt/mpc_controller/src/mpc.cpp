#include "mpc_controller/mpc.h"
#include <nav_msgs/Path.h>
#include "nav_msgs/Odometry.h"
#include "tf/tf.h"
#include "tf/transform_datatypes.h"

MpcController::MpcController(const ros::NodeHandle &nh){
    nh_ = nh;

    nh.param("/mpc/du_threshold", du_th, -1.0);
    nh.param("/mpc/dt", dt, -1.0);
    nh.param("/mpc/max_iter", max_iter, -1);
    nh.param("/mpc/predict_steps", T, -1);
    nh.param("/mpc/max_omega", max_omega, -1.0);
    nh.param("/mpc/max_domega", max_domega, -1.0);
    nh.param("/mpc/max_vel", max_speed, -1.0);
    nh.param("/mpc/min_vel", min_speed, -1.0);
    nh.param("/mpc/max_acc", max_accel, -1.0);
    nh.param("/mpc/delay_num", delay_num, -1);
    nh.param("/mpc/cmd_timer_rate", cmd_timer_rate_, 100.0);
    nh.param("/mpc/max_mpc_time", max_mpc_time_, 10.0);

    nh.param("/mpc/if_mpc", use_mpc_, true);

    emergency_stop_sub_ = nh_.subscribe<std_msgs::Bool>("/planner/emergency_stop", 1, &MpcController::emergencyStop, this);


    nh.param<std::vector<double>>("/mpc/matrix_q", Q, std::vector<double>());
    nh.param<std::vector<double>>("/mpc/matrix_r", R, std::vector<double>());
    nh.param<std::vector<double>>("/mpc/matrix_rd", Rd, std::vector<double>());

    cmd_timer_ = nh_.createTimer(ros::Duration(1.0/cmd_timer_rate_),&MpcController::CmdCallback, this);

    has_odom = false;
    receive_traj_ = false;
    max_comega = max_domega * dt;
    max_cv = max_accel * dt;
    xref = Eigen::Matrix<double, 4, 500>::Zero(4, 500);
    last_output = output = dref = Eigen::Matrix<double, 2, 500>::Zero(2, 500);
    current_output = Eigen::Vector2d::Zero();
    for (int i=0; i<delay_num; i++) output_buff.push_back(Eigen::Vector2d::Zero());


    double state_seq_res;
    int Integral_appr_resInt;
    nh.param("/mpc/state_seq_res", state_seq_res, 1.0);
    nh.param("/mpc/Integral_appr_resInt", Integral_appr_resInt, 10);
    traj_.setRes(state_seq_res, Integral_appr_resInt);
    new_traj_.setRes(state_seq_res, Integral_appr_resInt);

    traj_sub_ = nh_.subscribe<carstatemsgs::Polynome>("traj", 1, &MpcController::TrajCallback, this);
    // odom_sub_ = nh_.subscribe<carstatemsgs::CarState>("odom", 1, &MpcController::OdomCallback, this);
    odom_sub_ = nh_.subscribe<nav_msgs::Odometry>("odom", 1, &MpcController::OdomCallback, this);
    sequence_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/mpc/sequence",1);
    cmd_pub_ = nh_.advertise<carstatemsgs::CarState>("cmd",1);

    Ref_path_pub_ = nh_.advertise<nav_msgs::Path>("/mpc/Ref_path",10);
    cmd_path_pub_ = nh_.advertise<nav_msgs::Path>("/mpc/cmd_path",10);
    Ref_path_marker_pub_ = nh_.advertise<visualization_msgs::Marker>("/mpc/Ref_path_marker",10);

    Ref_velocity_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/mpc/Ref_velocity",10);
    Real_velocity_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/mpc/Real_velocity",10);

    {
        geometry_msgs::PoseStamped ref_v;
        Ref_velocity_pub_.publish(ref_v);
    }

    start_time = -1;
}

// void MpcController::OdomCallback(const carstatemsgs::CarState::ConstPtr& msg){    
//     has_odom = true;

//     now_state.x = msg->x;
//     now_state.y = msg->y;
//     now_state.theta = msg->yaw;
//     now_state.v = msg->v;
//     now_state.w = msg->omega;

//     now_state_time_ = msg->Header.stamp;
// }

void MpcController::OdomCallback(const nav_msgs::Odometry::ConstPtr& msg){    
    has_odom = true;

    now_state.x = msg->pose.pose.position.x;
    now_state.y = msg->pose.pose.position.y;
    now_state.theta = tf::getYaw(msg->pose.pose.orientation);
    now_state.v = 0.0;
    now_state.w = 0.0;

    now_state_time_ = msg->header.stamp;

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

    new_traj_.setTraj(start_state, initstate, finalstate, InnerPoints, t_pts);

    sequencePub();
    new_traj_start_time_ = msg->traj_start_time.toSec();

    new_traj_.if_get_traj_ = true;

    // traj_duration = traj_.get_traj_duration();
    receive_traj_ = true;
    at_goal = false;
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
        current_output << 0.0, 0.0;
        start_time = -1;
    }
    else{
        if(use_mpc_){
            getRefPoints(T, dt);
            for (int i = 0; i < T; i++){
                xref(0, i) = P[i].x;
                xref(1, i) = P[i].y;
                xref(3, i) = P[i].theta;
                dref(0, i) = P[i].v;
                dref(1, i) = P[i].w;
            }
            smooth_yaw();
            getCmd();
            carstatemsgs::CarState cmd;
            cmd.Header.frame_id = "world";
            cmd.Header.stamp = ros::Time::now();
            cmd.v = output(0, delay_num);
            cmd.omega = output(1, delay_num);

            cmd.a = 0.0;
            cmd.alpha = 0.0;
            cmd.js = 0.0;
            cmd.jyaw = 0.0;
            cmd_pub_.publish(cmd);

            current_output = output.col(delay_num);
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
    
}


void MpcController::normlize_theta(double &th){
    while (th > M_PI) th -= 2 * M_PI;
    while (th < -M_PI) th += 2 * M_PI;
}

void MpcController::getLinearModel(const MPCState& s)
{
    B = Eigen::Matrix<double, 3, 2>::Zero();
    B(0, 0) = cos(s.theta) * dt;
    B(1, 0) = sin(s.theta) * dt;
    B(2, 1) = dt;

    A = Eigen::Matrix3d::Identity();
    A(0, 2) = -B(1, 0) * s.v;
    A(1, 2) = B(0, 0) * s.v;

    C = Eigen::Vector3d::Zero();
    C(0) = -A(0, 2) * s.theta;
    C(1) = -A(1, 2) * s.theta;
}

void MpcController::stateTrans(MPCState& s, double a, double yaw_dot)
{
    if (yaw_dot >= max_omega)
    {
        yaw_dot = max_omega;
    }
    else if (yaw_dot <= -max_omega)
    {
        yaw_dot = -max_omega;
    }
    if (s.v >= max_speed)
    {
        s.v = max_speed;
    }
    else if (s.v <= min_speed)
    {
        s.v = min_speed;
    }

    s.x = s.x + a * cos(s.theta) * dt;
    s.y = s.y + a * sin(s.theta) * dt;
    s.theta = s.theta + yaw_dot * dt;
    s.v = a;
}


void MpcController::predictMotion(void)
{
    xbar[0] = now_state;

    MPCState temp = now_state;
    for (int i = 1; i < T + 1; i++)
    {
        stateTrans(temp, output(0, i - 1), output(1, i - 1));
        xbar[i] = temp;
    }
}

void MpcController::predictMotion(MPCState* b)
{
    b[0] = xbar[0];

    Eigen::MatrixXd Ax;
    Eigen::MatrixXd Bx;
    Eigen::MatrixXd Cx;
    Eigen::MatrixXd xnext;
    MPCState temp = xbar[0];

    for (int i = 1; i < T + 1; i++)
    {
        Bx = Eigen::Matrix<double, 3, 2>::Zero();
        Bx(0, 0) = cos(xbar[i - 1].theta) * dt;
        Bx(1, 0) = sin(xbar[i - 1].theta) * dt;
        Bx(2, 1) = dt;

        Ax = Eigen::Matrix3d::Identity();
        Ax(0, 2) = -Bx(1, 0) * xbar[i - 1].v;
        Ax(1, 2) = Bx(0, 0) * xbar[i - 1].v;

        Cx = Eigen::Vector3d::Zero();
        Cx(0) = -Ax(0, 2) * xbar[i - 1].theta;
        Cx(1) = -Ax(1, 2) * xbar[i - 1].theta;
        xnext = Ax * Eigen::Vector3d(temp.x, temp.y, temp.theta) +
                Bx * Eigen::Vector2d(output(0, i - 1), output(1, i - 1)) + Cx;
        temp.x = xnext(0);
        temp.y = xnext(1);
        temp.theta = xnext(2);
        b[i] = temp;
    }
}

void MpcController::solveMPCV(void)
{
    const int dimx = 3 * (T - delay_num);
    const int dimu = 2 * (T - delay_num);
    const int nx = dimx + dimu;

    Eigen::SparseMatrix<double> hessian;
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(nx);
    Eigen::SparseMatrix<double> linearMatrix;
    Eigen::VectorXd lowerBound;
    Eigen::VectorXd upperBound;

    // first-order
    for (int i=0, j=delay_num, k=0; i<dimx; i+=3, j++, k+=2)
    {
        gradient[i] = -2 * Q[0] * xref(0, j);
        gradient[i+1] = -2 * Q[1] * xref(1, j);
        gradient[i+2] = -2 * Q[3] * xref(3, j);
        gradient[dimx+k] = -2 * Q[2] * dref(0, j);
    }

    // second-order
    const int nnzQ = nx + dimu - 2;
    int irowQ[nnzQ];
    int jcolQ[nnzQ];
    double dQ[nnzQ];
    for (int i=0; i<nx; i++)
    {
        irowQ[i] = jcolQ[i] = i;
    }
    for (int i=nx; i<nnzQ; i++)
    {
        irowQ[i] = i - dimu + 2;
        jcolQ[i] = i - dimu;
    }
    for (int i=0; i<dimx; i+=3)
    {
        dQ[i] = Q[0] * 2.0;
        dQ[i+1] = Q[1] * 2.0;
        dQ[i+2] = Q[3] * 2.0;
    }
    dQ[dimx] = dQ[nx-2] = (R[0] + Rd[0] + Q[2]) * 2.0;
    dQ[dimx + 1] = dQ[nx-1] = (R[1] + Rd[1]) * 2.0;
    for (int i=dimx+2; i<nx-2; i+=2)
    {
        dQ[i] = 2 * (R[0] + 2 * Rd[0] + Q[2]);
        dQ[i+1] = 2 * (R[1] + 2 * Rd[1]);
    }
    for (int i=nx; i<nnzQ; i+=2)
    {
        dQ[i] = -Rd[0] * 2.0;
        dQ[i+1] = -Rd[1] * 2.0;
    }
    hessian.resize(nx, nx);
    Eigen::MatrixXd QQ(nx, nx);
    for (int i=0; i<nx; i++)
    {
        hessian.insert(irowQ[i], jcolQ[i]) = dQ[i];
    }
    for (int i=nx; i<nnzQ; i++)
    {
        hessian.insert(irowQ[i], jcolQ[i]) = dQ[i];
        hessian.insert(jcolQ[i], irowQ[i]) = dQ[i];
    }

    // equality constraints
    MPCState temp = xbar[delay_num];
    getLinearModel(temp);
    int my = dimx;
    double b[my];
    const int nnzA = 11 * (T-delay_num) - 5;
    int irowA[nnzA];
    int jcolA[nnzA];
    double dA[nnzA];
    Eigen::Vector3d temp_vec(temp.x, temp.y, temp.theta);
    Eigen::Vector3d temp_b = A*temp_vec + C;
    
    for (int i=0; i<dimx; i++)
    {
        irowA[i] = jcolA[i] = i;
        dA[i] = 1;
    }
    b[0] = temp_b[0];
    b[1] = temp_b[1];
    b[2] = temp_b[2];
    irowA[dimx] = 0;
    jcolA[dimx] = dimx;
    dA[dimx] = -B(0, 0);
    irowA[dimx+1] = 1;
    jcolA[dimx+1] = dimx;
    dA[dimx+1] = -B(1, 0);
    irowA[dimx+2] = 2;
    jcolA[dimx+2] = dimx+1;
    dA[dimx+2] = -B(2, 1);
    int ABidx = 8*(T-delay_num) - 8;
    int ABbegin = dimx+3;
    for (int i=0, j=1; i<ABidx; i+=8, j++)
    {
        getLinearModel(xbar[j+delay_num]);
        for (int k=0; k<3; k++)
        {
            b[3*j+k] = C[k];
            irowA[ABbegin + i + k] = 3*j + k;
            jcolA[ABbegin + i + k] = irowA[ABbegin + i + k] - 3;
            dA[ABbegin + i + k] = -A(k, k);
        }
        irowA[ABbegin + i + 3] = 3*j;
        jcolA[ABbegin + i + 3] = 3*j - 1;
        dA[ABbegin + i + 3] = -A(0, 2);

        irowA[ABbegin + i + 4] = 3*j + 1;
        jcolA[ABbegin + i + 4] = 3*j - 1;
        dA[ABbegin + i + 4] = -A(1, 2);
        
        irowA[ABbegin + i + 5] = 3*j;
        jcolA[ABbegin + i + 5] = dimx + 2*j;
        dA[ABbegin + i + 5] = -B(0, 0);
        
        irowA[ABbegin + i + 6] = 3*j + 1;
        jcolA[ABbegin + i + 6] = dimx + 2*j;
        dA[ABbegin + i + 6] = -B(1, 0);
        
        irowA[ABbegin + i + 7] = 3*j + 2;
        jcolA[ABbegin + i + 7] = dimx + 2*j + 1;
        dA[ABbegin + i + 7] = -B(2, 1);
    }

    // iequality constraints
    const int mz  = 2 * (T-delay_num) - 2;
    const int nnzC = 2 * dimu - 4;
    int   irowC[nnzC];
    int   jcolC[nnzC];
    double   dC[nnzC];
    for (int i=0, k=0; i<mz; i+=2, k+=4)
    {
        irowC[k] = i;
        jcolC[k] = dimx  + i;
        dC[k] = -1.0;

        irowC[k+1] = i;
        jcolC[k+1] = jcolC[k] +2;
        dC[k+1] = 1.0;

        irowC[k+2] = i + 1;
        jcolC[k+2] = dimx + 1 + i;
        dC[k+2] = -1.0;

        irowC[k+3] = i + 1;
        jcolC[k+3] = jcolC[k+2] +2;
        dC[k+3] = 1.0;
    }

    // xlimits and all
    int mx = dimu;
    int nc = mx+my+mz;
    lowerBound.resize(nc);
    upperBound.resize(nc);
    linearMatrix.resize(nc, nx);
    for (int i=0; i<mx; i+=2)
    {
        lowerBound[i] = -max_speed;
        lowerBound[i+1] = -max_omega;
        upperBound[i] = max_speed;
        upperBound[i+1] = max_omega;
        linearMatrix.insert(i, dimx+i) = 1;
        linearMatrix.insert(i+1, dimx+i+1) = 1;
    }

    for (int i=0; i<nnzA; i++)
    {
        linearMatrix.insert(irowA[i]+mx, jcolA[i]) = dA[i];
    }

    for (int i=0; i<my; i++)
    {
        lowerBound[mx+i] = upperBound[mx+i] = b[i];
    }

    for (int i=0; i<nnzC; i++)
    {
        linearMatrix.insert(irowC[i]+mx+my, jcolC[i]) = dC[i];
    }

    for (int i=0; i<mz; i+=2)
    {
        lowerBound[mx+my+i] = -max_cv;
        upperBound[mx+my+i] = max_cv;
        lowerBound[mx+my+i+1] = -max_comega;
        upperBound[mx+my+i+1] = max_comega;
    }
    // instantiate the solver
    OsqpEigen::Solver solver;

    // settings
    solver.settings()->setVerbosity(false);
    solver.settings()->setWarmStart(true);
    solver.settings()->setAbsoluteTolerance(1e-6);
    solver.settings()->setMaxIteration(30000);
    solver.settings()->setRelativeTolerance(1e-6);
    // set the initial data of the QP solver
    solver.data()->setNumberOfVariables(nx);
    solver.data()->setNumberOfConstraints(nc);
    if (!solver.data()->setHessianMatrix(hessian)) return;
    if (!solver.data()->setGradient(gradient)) return;
    if (!solver.data()->setLinearConstraintsMatrix(linearMatrix)) return;
    if (!solver.data()->setLowerBound(lowerBound)) return;
    if (!solver.data()->setUpperBound(upperBound)) return;

    // instantiate the solver
    if (!solver.initSolver()) return;

    // controller input and QPSolution vector
    Eigen::VectorXd QPSolution;

    // solve the QP problem
    if (!solver.solve()) return;

    // get the controller input
    QPSolution = solver.getSolution();
    // ROS_INFO("Solution: v0=%f     omega0=%f", QPSolution[dimx], QPSolution[dimx+1]);
    for (int i = 0; i < delay_num; i++)
    {
        output.col(i) = output_buff[i];
        // output(0, i) = output_buff[i][0];
        // output(1, i) = output_buff[i][1];
    }
    for (int i = 0, j = 0; i < dimu; i += 2, j++)
    {
        output(0, j + delay_num) = QPSolution[dimx + i];
        output(1, j + delay_num) = QPSolution[dimx + i + 1];
    }
}


void MpcController::smooth_yaw(void)
{
    double dyaw = xref(3, 0) - now_state.theta;

    while (dyaw >= M_PI / 2)
    {
        xref(3, 0) -= M_PI * 2;
        dyaw = xref(3, 0) - now_state.theta;
    }
    while (dyaw <= -M_PI / 2)
    {
        xref(3, 0) += M_PI * 2;
        dyaw = xref(3, 0) - now_state.theta;
    }

    for (int i = 0; i < T - 1; i++)
    {
        dyaw = xref(3, i + 1) - xref(3, i);
        while (dyaw >= M_PI / 2)
        {
            xref(3, i + 1) -= M_PI * 2;
            dyaw = xref(3, i + 1) - xref(3, i);
        }
        while (dyaw <= -M_PI / 2)
        {
            xref(3, i + 1) += M_PI * 2;
            dyaw = xref(3, i + 1) - xref(3, i);
        }
    }
}

void MpcController::getCmd(void)
{
    int iter;
    Last_plan_time = ros::Time::now().toSec();
    for (iter = 0; iter < max_iter; iter++)
    {
        predictMotion();
        last_output = output;
        solveMPCV();
        double du = 0;
        for (int i = 0; i < output.cols(); i++)
        {
            du = du + fabs(output(0, i) - last_output(0, i)) + fabs(output(1, i) - last_output(1, i));
        }
        // break;
        // if (du <= du_th || (ros::Time::now() - begin).toSec() > max_mpc_time_)
        if (ros::Time::now().toSec() - Last_plan_time > max_mpc_time_)
        {
            break;
        }
    }
    // if (iter == max_iter)
    // {
    //     ROS_WARN("MPC Iterative is max iter");
    // }
    predictMotion(xopt);

    nav_msgs::Path path;
    path.header.frame_id = "world";
    path.header.stamp = ros::Time::now();
    geometry_msgs::PoseStamped pose;
    for (int i = 0; i < T; i++){
        pose.pose.position.x = xopt[i].x;
        pose.pose.position.y = xopt[i].y;
        pose.pose.position.z = 0;
        pose.pose.orientation = tf::createQuaternionMsgFromYaw(xopt[i].theta);
        path.poses.push_back(pose);
    }
    cmd_path_pub_.publish(path);
    
    if (delay_num > 0)
    {
        output_buff.erase(output_buff.begin());
        output_buff.push_back(Eigen::Vector2d(output(0, delay_num), output(1, delay_num)));
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
    current_output << 0.0, 0.0;
    start_time = -1;
}

void MpcController::getRefPoints(const int T, double dt){
    P.clear();
    TrajPoint tp;
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
    for (double temp_t = t_cur + dt; j < T; j++, temp_t += dt){
        if (temp_t <= traj_duration){
            Eigen::Vector3d curP = traj_.getPstate(temp_t);
            Eigen::Vector2d curV = traj_.getVstate(temp_t);
            Eigen::Vector2d curA = traj_.getAstate(temp_t);

            tp.v = curV.y();
            tp.x = curP.x();
            tp.y = curP.y();
            tp.a = curA.y();
            tp.theta = curP.z();
            normlize_theta(tp.theta);
            tp.w = curV.x();
            P.push_back(tp);
        }
        else{
            Eigen::Vector3d curP = traj_.getPstate(traj_duration);
            Eigen::Vector2d curV = traj_.getVstate(traj_duration);
            Eigen::Vector2d curA = traj_.getAstate(traj_duration);

            tp.v = curV.y();
            tp.x = curP.x();
            tp.y = curP.y();
            tp.a = curA.y();
            tp.theta = curP.z();
            normlize_theta(tp.theta);
            tp.w = curV.x();
            P.push_back(tp);
        }
    }

    nav_msgs::Path path;
    path.header.frame_id = "world";
    path.header.stamp = ros::Time::now();
    geometry_msgs::PoseStamped pose;
    for (u_int i = 0; i < P.size(); i++){
        pose.pose.position.x = P[i].x;
        pose.pose.position.y = P[i].y;
        pose.pose.position.z = 0;
        pose.pose.orientation = tf::createQuaternionMsgFromYaw(P[i].theta);
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


    for (u_int i = 0; i < P.size(); i++){
        geometry_msgs::Point point;
        point.x = P[i].x;
        point.y = P[i].y;
        point.z = 0;
        path_marker.points.push_back(point);
    }

    Ref_path_marker_pub_.publish(path_marker);
}

void MpcController::sequencePub(){
    sensor_msgs::PointCloud2 globalMap_pcd;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr  colored_pcl_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZ> cloudMap;
    pcl::PointXYZRGB  pt;

    std::vector<Eigen::Vector4d> seq = new_traj_.get_state_sequence_();
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