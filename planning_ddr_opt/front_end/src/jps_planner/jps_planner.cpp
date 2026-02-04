#include "front_end/jps_planner/jps_planner.h"

using namespace JPS;

JPSPlanner::JPSPlanner(std::shared_ptr<SDFmap> map, const ros::NodeHandle &nh): map_util_(map), nh_(nh) {
    // graph_search_ = std::make_shared<GraphSearch>(map_util_);
    path_pub_ = ros::Publisher(nh_.advertise<nav_msgs::Path>("jps_path", 1));
    init_path_pub_ = ros::Publisher(nh_.advertise<nav_msgs::Path>("init_jps_path", 1));
    normal_vector_pub_ = ros::Publisher(nh_.advertise<visualization_msgs::Marker>("normal_vector", 1));

    nh_.getParam(ros::this_node::getName()+ "/jps_safe_dis", safe_dis_);
    nh_.getParam(ros::this_node::getName()+ "/max_jps_dis", max_jps_dis_);
    nh_.getParam(ros::this_node::getName()+ "/jps_distance_weight", distance_weight_);
    nh_.getParam(ros::this_node::getName()+ "/jps_yaw_weight", yaw_weight_);
    nh_.getParam(ros::this_node::getName()+ "/trajCutLength", trajCutLength_);

    nh_.getParam(ros::this_node::getName()+ "/max_vel",max_vel_);
    nh_.getParam(ros::this_node::getName()+ "/max_acc",max_acc_);
    nh_.getParam(ros::this_node::getName()+ "/max_omega",max_omega_);
    nh_.getParam(ros::this_node::getName()+ "/max_domega",max_domega_);

    nh_.getParam(ros::this_node::getName()+ "/timeResolution",sampletime_);
    nh_.getParam(ros::this_node::getName()+ "/mintrajNum", mintrajNum_);

    nh_.getParam(ros::this_node::getName()+ "/jps_truncation_time", jps_truncation_time_);

    graph_search_ = std::make_shared<GraphSearch>(map_util_, safe_dis_);

}

bool JPSPlanner::plan(const Eigen::Vector3d &start, const Eigen::Vector3d &goal){
    
    start_state_ = start;
    end_state_ = goal;
    
    Eigen::Vector2i start_idx = map_util_->coord2gridIndex(start.head(2));
    Eigen::Vector2i goal_idx = map_util_->coord2gridIndex(goal.head(2));
    
    double start_dis = map_util_->getDistanceReal(map_util_->gridIndex2coordd(start_idx)) * 0.8;
    double goal_dis = map_util_->getDistanceReal(map_util_->gridIndex2coordd(goal_idx)) * 0.8;
    double safe_dis = std::max(std::min(safe_dis_, start_dis), 0.0);
    safe_dis = std::max(std::min(safe_dis, goal_dis), 0.0);

    graph_search_ = std::make_shared<GraphSearch>(map_util_, safe_dis);
    graph_search_->plan(start_idx(0), start_idx(1), goal_idx(0), goal_idx(1), true, 1e10);

    const auto path = graph_search_->getPath();
    if (path.size() < 1) {
        std::cout << "Cannot find a path from " << start.transpose() <<" to " << goal.transpose() << " Abort!" << std::endl;
        status_ = -1;
        return false;
    }

    std::vector<Eigen::Vector2d> ps;
    for (const auto &it : path) {
        ps.push_back(map_util_->gridIndex2coordd(Eigen::Vector2i(it->x, it->y)));
    }
    pubPath(ps, init_path_pub_);

    raw_path_ = ps;
    std::reverse(std::begin(raw_path_), std::end(raw_path_));
    
    raw_path_.front() = start.head(2);
    raw_path_.back() = goal.head(2);


    return true;
}

void JPSPlanner::get_small_resolution_path_(){
    small_resolution_path_.clear();

    int path_size = path_.size();
    for(int i = 0; i < path_size - 1; i++){
        Eigen::Vector2i start = map_util_->coord2gridIndex(path_[i]);
        Eigen::Vector2i end = map_util_->coord2gridIndex(path_[i+1]);
        std::vector<Eigen::Vector2i> line = getGridsBetweenPoints2D(start, end);
        small_resolution_path_.insert(small_resolution_path_.end(), line.begin(), line.end());
    }

}

void JPSPlanner::pubPath(const std::vector<Eigen::Vector2d> &path, const ros::Publisher &pub){
    nav_msgs::Path path_msg;
    path_msg.header.stamp = ros::Time::now();
    path_msg.header.frame_id = "world";
    for (const auto &it : path) {
        geometry_msgs::PoseStamped pose;
        pose.pose.position.x = it(0);
        pose.pose.position.y = it(1);
        pose.pose.position.z = 0;
        path_msg.poses.push_back(pose);
    }
    pub.publish(path_msg);
}

std::vector<Eigen::Vector2d> JPSPlanner::removeCornerPts(const std::vector<Eigen::Vector2d> &path) {
    if (path.size() < 2)
        return path;

    // cut zigzag segment
    std::vector<Eigen::Vector2d> optimized_path;
    Eigen::Vector2d pose1 = path[0];
    Eigen::Vector2d pose2 = path[1];
    Eigen::Vector2d prev_pose = pose1;
    optimized_path.push_back(pose1);
    double cost1, cost2, cost3;

    if (!checkLineCollision(pose1, pose2))
        cost1 = (pose1 - pose2).norm();
    else
        cost1 = std::numeric_limits<double>::infinity();

    for (unsigned int i = 1; i < path.size() - 1; i++) {
        pose1 = path[i];
        pose2 = path[i + 1];
        if (!checkLineCollision(pose1, pose2))
            cost2 = (pose1 - pose2).norm();
        else
            cost2 = std::numeric_limits<double>::infinity();

        if (!checkLineCollision(prev_pose, pose2))
            cost3 = (prev_pose - pose2).norm();
        else
            cost3 = std::numeric_limits<double>::infinity();

        if (cost3 < cost1 + cost2)
            cost1 = cost3;
        else {
            optimized_path.push_back(path[i]);
            cost1 = (pose1 - pose2).norm();
            prev_pose = pose1;
        }
    }

    optimized_path.push_back(path.back());
    return optimized_path;
}

bool JPSPlanner::checkLineCollision(const Eigen::Vector2d &start, const Eigen::Vector2d &end){
    std::vector<Eigen::Vector2i> line = getGridsBetweenPoints2D(map_util_->coord2gridIndex(start), map_util_->coord2gridIndex(end));
    for(auto line_pt:line){
        if(map_util_->isOccWithSafeDis(line_pt, graph_search_->GetSafeDis())){
            return true;
        }
    }
    return false;
}

std::vector<Eigen::Vector2i> JPSPlanner::getGridsBetweenPoints2D(const Eigen::Vector2i &start, const Eigen::Vector2i &end){
    std::vector<Eigen::Vector2i> line;
    
    int dx = abs(end.x() - start.x());
    int dy = abs(end.y() - start.y());
    int sx = (start.x() < end.x()) ? 1 : -1;
    int sy = (start.y() < end.y()) ? 1 : -1;
    int err = dx - dy;

    double x0 = start.x();
    double y0 = start.y();

    while (true) {
        line.emplace_back(x0, y0);
        if (x0 == end.x() && y0 == end.y()) break;
        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x0 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y0 += sy;
        }
    }

    return line;
}

// void JPSPlanner::getKinoNode(const Eigen::Vector3d &current_state_VAJ, const Eigen::Vector3d &current_state_OAJ){
//     current_state_VAJ_ = current_state_VAJ;
//     current_state_OAJ_ = current_state_OAJ;

//     get_small_resolution_path_();
//     pubPath(path_, path_pub_);
//     getSampleTraj();
//     getTrajsWithTime();
// }

void JPSPlanner::getKinoNodeWithStartPath(const std::vector<Eigen::Vector3d> &start_path, const bool if_forward, const Eigen::Vector3d &current_state_VAJ, const Eigen::Vector3d &current_state_OAJ){
    current_state_VAJ_ = current_state_VAJ;
    current_state_OAJ_ = current_state_OAJ;

    if(start_path.size() > 0){
      std::vector<Eigen::Vector2d> start_path_2d;
      for(auto pt:start_path){
        start_path_2d.push_back(pt.head(2));
        ROS_INFO_STREAM("start_path_3d: " << pt.transpose());
        start_path_2d.pop_back();
      }
      raw_path_.insert(raw_path_.begin(), start_path_2d.begin(), start_path_2d.end());
      start_state_ = start_path.front();
    }
    
    // get_small_resolution_path_();
    path_ = removeCornerPts(raw_path_);
    Unoccupied_path_ = path_;

    // for(auto pt:Unoccupied_path_){
    //   std::cout<<"Unoccupied_path_: "<<pt.transpose()<<std::endl;
    // }

    pubPath(path_, path_pub_);
    getSampleTraj();
    getTrajsWithTime();
}

void JPSPlanner::getSampleTraj(){
    Unoccupied_sample_trajs_.clear();
    double cur_theta;

    Eigen::VectorXd state5d;// x y theta dtheta ds
    state5d.resize(5);
    state5d << start_state_.x(), start_state_.y(), start_state_.z(), 0, 0;
    Unoccupied_sample_trajs_.push_back(state5d); 
    
    cur_theta = atan2(Unoccupied_path_[1].y() - Unoccupied_path_[0].y(), Unoccupied_path_[1].x() - Unoccupied_path_[0].x());
    normalizeAngle(start_state_.z(), cur_theta);
    state5d << start_state_.x(), start_state_.y(), cur_theta , cur_theta - start_state_.z(), 0;
    Unoccupied_sample_trajs_.push_back(state5d); 
    cur_theta = atan2(Unoccupied_path_[0].y() - Unoccupied_path_[1].y(), Unoccupied_path_[0].x() - Unoccupied_path_[1].x()) + M_PI;
    normalizeAngle(start_state_.z(), cur_theta);
    state5d << start_state_.x(), start_state_.y(), cur_theta , cur_theta - start_state_.z(), 0;
    Unoccupied_sample_trajs_.push_back(state5d); // 2
    
    int path_size = Unoccupied_path_.size();
    Eigen::VectorXd pt;
    for(int i = 1; i<path_size-1; i++){ 
        pt = Unoccupied_path_[i];

        state5d << pt.x(), pt.y(), Unoccupied_sample_trajs_.back()[2], 0, sqrt(pow(pt.x() - Unoccupied_sample_trajs_.back()[0], 2) + pow(pt.y() - Unoccupied_sample_trajs_.back()[1], 2));
        Unoccupied_sample_trajs_.push_back(state5d);
        cur_theta = atan2(Unoccupied_path_[i+1].y() - Unoccupied_path_[i].y(), Unoccupied_path_[i+1].x() - Unoccupied_path_[i].x());
        normalizeAngle(Unoccupied_sample_trajs_.back()[2], cur_theta);
        state5d << pt.x(), pt.y(), cur_theta, cur_theta - Unoccupied_sample_trajs_.back()[2], 0;
        Unoccupied_sample_trajs_.push_back(state5d);

    }
    pt = Unoccupied_path_.back();
    state5d << pt.x(), pt.y(), Unoccupied_sample_trajs_.back()[2], 0, sqrt(pow(pt.x() - Unoccupied_sample_trajs_.back()[0], 2) + pow(pt.y() - Unoccupied_sample_trajs_.back()[1], 2));
    Unoccupied_sample_trajs_.push_back(state5d);

    cur_theta = end_state_.z();
    normalizeAngle(Unoccupied_sample_trajs_.back()[2], cur_theta);
    state5d << pt.x(), pt.y(), cur_theta, cur_theta - Unoccupied_sample_trajs_.back()[2], 0;
    Unoccupied_sample_trajs_.push_back(state5d);
}

void JPSPlanner::getTrajsWithTime(){
    cut_Unoccupied_sample_trajs_.clear();


    std::vector<double> Unoccupied_thetas;
    std::vector<double> Unoccupied_pathlengths; 
    std::vector<double> Unoccupied_Weightpathlengths; 

    double Unoccupied_AllWeightingPathLength_ = 0; 
    double Unoccupied_AllPathLength = 0;

    bool if_cut = false;
    Eigen::Vector3d cut_state = Unoccupied_sample_trajs_.back().head(3);

    int PathNodeNum = Unoccupied_sample_trajs_.size();
    cut_Unoccupied_sample_trajs_.push_back(Unoccupied_sample_trajs_[0]);
    Unoccupied_thetas.push_back(Unoccupied_sample_trajs_[0][2]);
    Unoccupied_pathlengths.push_back(0);
    Unoccupied_Weightpathlengths.push_back(0);

    
    int pathnodeindex = 1;
    for(; pathnodeindex<PathNodeNum&&!if_cut; pathnodeindex++){
        Eigen::VectorXd pathnode = Unoccupied_sample_trajs_[pathnodeindex];
        if(Unoccupied_AllPathLength + fabs(pathnode[4]) >= trajCutLength_ && pathnode[4] != 0){
            if_cut = true;
            
            Eigen::Vector3d former_state = Unoccupied_sample_trajs_[pathnodeindex-1].head(3);
            cut_state = former_state + (pathnode.head(3) - former_state) * (trajCutLength_ - Unoccupied_AllPathLength) / fabs(pathnode[4]);
            Eigen::VectorXd state5d; state5d.resize(5);
            state5d<<cut_state.x(), cut_state.y(), cut_state.z(), (trajCutLength_ - Unoccupied_AllPathLength)/fabs(pathnode[4]) * pathnode[3], trajCutLength_ - Unoccupied_AllPathLength;
            cut_Unoccupied_sample_trajs_.push_back(state5d);
            Unoccupied_thetas.push_back(state5d[2]);
            Unoccupied_AllPathLength += state5d[4];
            Unoccupied_pathlengths.push_back(Unoccupied_AllPathLength); 
            Unoccupied_AllWeightingPathLength_ += yaw_weight_ * abs(state5d[3]) + distance_weight_ * abs(state5d[4]);
            Unoccupied_Weightpathlengths.push_back(Unoccupied_AllWeightingPathLength_);

            PathNodeNum = cut_Unoccupied_sample_trajs_.size();
            break;
        }
        cut_Unoccupied_sample_trajs_.push_back(pathnode);
        Unoccupied_thetas.push_back(pathnode[2]);
        Unoccupied_AllPathLength += pathnode[4];
        Unoccupied_pathlengths.push_back(Unoccupied_AllPathLength); 
        Unoccupied_AllWeightingPathLength_ += yaw_weight_ * abs(pathnode[3]) + distance_weight_ * abs(pathnode[4]);
        Unoccupied_Weightpathlengths.push_back(Unoccupied_AllWeightingPathLength_);
    }

    double totalTrajTime_ = evaluateDuration(Unoccupied_AllWeightingPathLength_, current_state_VAJ_.x(),0.0,max_vel_,max_acc_);
    std::vector<Eigen::Vector3d> Unoccupied_traj_pts; // Store the sampled coordinates yaw, s, t
    std::vector<Eigen::Vector3d> Unoccupied_positions; // Store the sampled coordinates x, y, yaw

    double Unoccupied_totalTrajTime_ = totalTrajTime_;
    double Unoccupied_sampletime;
    int Unoccupied_PathNodeIndex = 1;

    Unoccupied_sampletime = Unoccupied_totalTrajTime_ / std::max(int(Unoccupied_totalTrajTime_ / sampletime_ + 0.5), mintrajNum_);

    PathNodeNum = cut_Unoccupied_sample_trajs_.size();
    double tmparc = 0;

    for(double samplet = Unoccupied_sampletime; samplet<Unoccupied_totalTrajTime_-1e-3; samplet+=Unoccupied_sampletime){
        double arc = evaluateLength(samplet, Unoccupied_AllWeightingPathLength_, Unoccupied_totalTrajTime_, current_state_VAJ_.x(), 0.0, max_vel_, max_acc_);
        for (int k = Unoccupied_PathNodeIndex; k<PathNodeNum; k++){
            Eigen::VectorXd pathnode = cut_Unoccupied_sample_trajs_[k];
            Eigen::VectorXd prepathnode = cut_Unoccupied_sample_trajs_[k-1];
            tmparc = Unoccupied_Weightpathlengths[k];
            if(tmparc >= arc){
                Unoccupied_PathNodeIndex = k; 
                double l1 = tmparc-arc;
                double l = Unoccupied_Weightpathlengths[k]-Unoccupied_Weightpathlengths[k-1];
                double interp_s = Unoccupied_pathlengths[k-1] + (l-l1)/l*(pathnode[4]);
                double interp_yaw = cut_Unoccupied_sample_trajs_[k-1][2] + (l-l1)/l*(pathnode[3]);
                Unoccupied_traj_pts.emplace_back(interp_yaw, interp_s, samplet);

                double interp_x = l1/l*prepathnode[0] + (l-l1)/l*(pathnode[0]);
                double interp_y = l1/l*prepathnode[1] + (l-l1)/l*(pathnode[1]);
                Unoccupied_positions.emplace_back(interp_x, interp_y, interp_yaw);
                break;
            }
        }
    }

    Eigen::MatrixXd startS;
    Eigen::MatrixXd endS;
    startS.resize(2,3);
    endS.resize(2,3);  
    Eigen::Vector2d startP(cut_Unoccupied_sample_trajs_[0][2],0);
    Eigen::Vector2d finalP(cut_Unoccupied_sample_trajs_[PathNodeNum-1][2],Unoccupied_pathlengths[PathNodeNum-1]);
    startS.col(0) = startP;
    startS.block(0,1,1,2) = current_state_OAJ_.transpose().head(2);
    startS.block(1,1,1,2) = current_state_VAJ_.transpose().head(2);
    endS.col(0) = finalP;
    endS.col(1).setZero();
    endS.col(2).setZero();
    flat_traj_.UnOccupied_traj_pts = Unoccupied_traj_pts;
    flat_traj_.UnOccupied_initT = Unoccupied_sampletime;
    flat_traj_.UnOccupied_positions = Unoccupied_positions;
  
    flat_traj_.start_state = startS;
    flat_traj_.final_state = endS;
    flat_traj_.start_state_XYTheta = start_state_;
    // flat_traj_.final_state_XYTheta = end_state_;
    flat_traj_.if_cut = if_cut;
    flat_traj_.final_state_XYTheta = cut_state;

    // flat_traj_.printFlatTrajData();
}

void JPSPlanner::normalizeAngle(const double &ref_angle, double &angle){
    while(ref_angle - angle > M_PI){
        angle += 2*M_PI;
    }
    while(ref_angle - angle < -M_PI){
        angle -= 2*M_PI;
    }
}

// Use trapezoidal velocity profile to get the total time of the path
double JPSPlanner::evaluateDuration(const double &length, const double &startV, const double &endV, const double &maxV, const double &maxA){
  double critical_len; 
  double startv2 = pow(startV,2);
  double endv2 = pow(endV,2);
  double maxv2 = pow(maxV,2);
  if(startV>maxV){
    startv2 = maxv2;
  }
  if(endV>max_vel_){
    endv2 = maxv2;
  }

  critical_len = (maxv2-startv2)/(2*maxA)+(maxv2-endv2)/(2*maxA);
  if(length>=critical_len){
    return (maxV-startV)/maxA+(maxV-endV)/maxA+(length-critical_len)/maxV;
  }
  else{
    double tmpv = sqrt(0.5*(startv2+endv2+2*maxA*length));
    return (tmpv-startV)/maxA + (tmpv-endV)/maxA;
  }
}



// Use trapezoidal velocity profile to get the distance at the curt timestamp
double JPSPlanner::evaluateLength(const double &curt, const double &locallength, const double &localtime, const double &startV, const double &endV, const double &maxV, const double &maxA){
  // std::cout<<"curt: "<<curt<<"  locallength: "<<locallength<<"  localtime: "<<localtime<<"  startV: "<<startV<<"  endV: "<<endV<<"  maxV: "<<maxV<<"  maxA: "<<maxA<<std::endl;
  double critical_len; 
  double startv2 = pow(startV,2);
  double endv2 = pow(endV,2);
  double maxv2 = pow(maxV,2);
  if(startV>maxV){
    startv2 = maxv2;
  }
  if(endV>max_vel_){
    endv2 = maxv2;
  }

  critical_len = (maxv2-startv2)/(2*maxA)+(maxv2-endv2)/(2*maxA);
  // Get time from trapezoidal velocity profile
  if(locallength>=critical_len){// If locallength is greater than critical_len, accelerate to max speed and then decelerate
    double t1 = (maxV-startV)/maxA;
    double t2 = t1+(locallength-critical_len)/maxV;
    if(curt<=t1){
      return startV*curt + 0.5*maxA*pow(curt,2);
    }
    else if(curt<=t2){
      return startV*t1 + 0.5*maxA*pow(t1,2)+(curt-t1)*maxV;
    }
    else{
      return startV*t1 + 0.5*maxA*pow(t1,2) + (t2-t1)*maxV + maxV*(curt-t2)-0.5*maxA*pow(curt-t2,2);
    }
  }
  else{// If locallength is less than critical_len, decelerate without accelerating to max speed
    double tmpv = sqrt(0.5*(startv2+endv2+2*maxA*locallength));
    double tmpt = (tmpv-startV)/maxA;
    if(curt<=tmpt){
      return startV*curt+0.5*maxA*pow(curt,2);
    }
    else{
      return startV*tmpt+0.5*maxA*pow(tmpt,2) + tmpv*(curt-tmpt)-0.5*maxA*pow(curt-tmpt,2);
    }
  }
}

double JPSPlanner::evaluateVel(const double &curt, const double &locallength, const double &localtime, const double &startV, const double &endV, const double &maxV, const double &maxA){
  double critical_len; 
  double startv2 = pow(startV,2);
  double endv2 = pow(endV,2);
  double maxv2 = pow(maxV,2);
  if(startV>maxV){
    startv2 = maxv2;
  }
  if(endV>max_vel_){
    endv2 = maxv2;
  }

  critical_len = (maxv2-startv2)/(2*maxA)+(maxv2-endv2)/(2*maxA);
  // Get time from trapezoidal velocity profile
  if(locallength>=critical_len){// If locallength is greater than critical_len, accelerate to max speed and then decelerate
    double t1 = (maxV-startV)/maxA;
    double t2 = t1+(locallength-critical_len)/maxV;
    if(curt<=t1){
      return startV + maxA*curt;
    }
    else if(curt<=t2){
      return maxV;
    }
    else{
      return maxV - maxA*(curt-t2);
    }
  }
  else{// If locallength is less than critical_len, decelerate without accelerating to max speed
    double tmpv = sqrt(0.5*(startv2+endv2+2*maxA*locallength));
    double tmpt = (tmpv-startV)/maxA;
    if(curt<=tmpt){
      return startV + maxA*curt;
    }
    else{
      return tmpv - maxA * (curt - tmpt);
    }
  }
}

double JPSPlanner::evaluteTimeOfPos(const double &pos, const double &locallength, const double &startV, const double &endV, const double &maxV, const double &maxA){
  double critical_len;
  double startv2 = pow(startV,2);
  double endv2 = pow(endV,2);
  double maxv2 = pow(maxV,2);
  double localpos = pos;
  if(startV>maxV){
    startv2 = maxv2;
  }
  if(endV>max_vel_){
    endv2 = maxv2;
  }
  if(pos>locallength){
    localpos = locallength;
  }

  critical_len = (maxv2-startv2)/(2*maxA)+(maxv2-endv2)/(2*maxA);
  if(locallength>=critical_len){
    double s1 = (maxv2-startv2)/maxA/2.0;
    if(localpos < s1){
      return (sqrt(startV*startV + 2*maxA*localpos)-startV)/maxA;
    }
    double s2 = locallength - (maxv2-endv2)/maxA/2.0;
    if(localpos < s2){
      return (maxV - startV)/maxA + (localpos-s1)/maxV;
    }
    else{
        return (maxV - startV)/maxA + (s2-s1)/maxV + (maxV - sqrt(maxv2-2.0*maxA*(localpos-s2)))/maxA;
    }
  }
  else{
    double v_m = sqrt(0.5*(startv2+endv2+2*maxA*locallength));
    if(localpos < (v_m*v_m - startv2)/2.0/maxA){
      return (sqrt(startV*startV + 2*maxA*localpos)-startV)/maxA;
    }
    else{
      double rest_s = pos - (v_m*v_m - startv2)/2.0/maxA;
      return (v_m - startV)/maxA + (v_m - sqrt(v_m*v_m - 2*maxA*rest_s))/maxA;
    }
  }
}

bool JPSPlanner::JPS_check_if_collision(const Eigen::Vector2d &pos){
  return map_util_->getDistanceReal(pos) > safe_dis_;
}

double JPSPlanner::getPathLength() const {
  // Calculate length from raw_path_ which is directly generated by plan()
  if (raw_path_.size() < 2) {
    return 0.0;
  }
  double length = 0.0;
  for (size_t i = 1; i < raw_path_.size(); ++i) {
    length += (raw_path_[i] - raw_path_[i - 1]).norm();
  }
  return length;
}