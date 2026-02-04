#ifndef _MPC_TRAJ_ANAL_H_
#define _MPC_TRAJ_ANAL_H_

#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

#include "gcopter/minco.hpp"

class TrajAnal{
    private:
        minco::MINCO_S3NU minco_;
        Eigen::Vector3d start_state_;
        Trajectory<5, 2> minco_traj_;

        // State sequence: position (x, y, theta, t). Note: does not include the endpoint
        std::vector<Eigen::Vector4d> state_sequence_;
        double state_seq_res_;

        // The resolution for approximating the position using integration. state_seq_res_ must be an integer multiple of Integral_appr_resInt_
        int Integral_appr_resInt_;

    public:
        TrajAnal(){}

        bool if_get_traj_ = false;

        void setRes(const double &state_seq_res, const double &Integral_appr_resInt){
            state_seq_res_ = state_seq_res;
            Integral_appr_resInt_ = Integral_appr_resInt;
        }

        void setTraj(const Eigen::Vector3d &start_state, 
                     const Eigen::MatrixXd &initstate, const Eigen::MatrixXd &finalstate, 
                     const Eigen::MatrixXd &Innerpoints, const Eigen::VectorXd &pieceTimes){
            int traj_num = pieceTimes.size();
            if(Innerpoints.cols() != traj_num - 1){
                printf("Innerpoints.cols() != pieceTimes.size()-1 !!  Innerpoints.cols(): %ld  pieceTimes.size(): %d", Innerpoints.cols(), traj_num);
                exit(1);
            }
            minco_.setConditions(initstate, finalstate, traj_num);
            minco_.setParameters(Innerpoints, pieceTimes);
            minco_.getTrajectory(minco_traj_);
            start_state_ = start_state;

            getSeq();
        }

        void getSeq(){
            state_sequence_.clear();

            double Integral_appr_res = state_seq_res_ / Integral_appr_resInt_;
            double half_Integral_appr_res = Integral_appr_res / 2.0;
            double Integral_appr_res_1_6 = Integral_appr_res / 6.0;

            Eigen::Vector3d current_state = start_state_;
            state_sequence_.emplace_back(current_state.x(), current_state.y(), current_state.z(), 0.0);

            int sequence_num = floor(minco_traj_.getTotalDuration() / Integral_appr_res);

            Eigen::Vector2d p1, p2, p3, v1, v2, v3;
            p3 = minco_traj_.getPos(0.0);
            v3 = minco_traj_.getVel(0.0);
            for(int i=0; i<sequence_num; i++){
                p1 = p3; v1 = v3;
                p2 = minco_traj_.getPos(i * Integral_appr_res + half_Integral_appr_res);
                v2 = minco_traj_.getVel(i * Integral_appr_res + half_Integral_appr_res);
                p3 = minco_traj_.getPos(i * Integral_appr_res + Integral_appr_res);
                v3 = minco_traj_.getVel(i * Integral_appr_res + Integral_appr_res);

                current_state.x() += Integral_appr_res_1_6 * (v1.y()*cos(p1.x()) + 4.0*v2.y()*cos(p2.x()) + v3.y()*cos(p3.x()));
                current_state.y() += Integral_appr_res_1_6 * (v1.y()*sin(p1.x()) + 4.0*v2.y()*sin(p2.x()) + v3.y()*sin(p3.x()));
                if(i%Integral_appr_resInt_ == Integral_appr_resInt_ - 1){
                    state_sequence_.emplace_back(current_state.x(), current_state.y(), p3.x(), (i+1) * Integral_appr_res);
                }
            }

            
        }

        std::vector<Eigen::Vector4d> get_state_sequence_(){
            return state_sequence_;
        }

        double get_traj_duration(){
            return minco_traj_.getTotalDuration();
        }

        Eigen::Vector3d getPstate(const double& t){
            int index = floor(t / state_seq_res_);
            double floor_t = index * state_seq_res_;
            double diff_t = t - floor_t;
            Eigen::Vector4d state = state_sequence_[index];
            Eigen::Vector2d p1, p2, p3, v1, v2, v3;
            p1 = minco_traj_.getPos(floor_t);
            v1 = minco_traj_.getVel(floor_t);
            p2 = minco_traj_.getPos(floor_t + diff_t/2.0);
            v2 = minco_traj_.getVel(floor_t + diff_t/2.0);
            p3 = minco_traj_.getPos(t);
            v3 = minco_traj_.getVel(t);

            state.x() += diff_t/6.0 * (v1.y()*cos(p1.x()) + 4.0*v2.y()*cos(p2.x()) + v3.y()*cos(p3.x()));
            state.y() += diff_t/6.0 * (v1.y()*sin(p1.x()) + 4.0*v2.y()*sin(p2.x()) + v3.y()*sin(p3.x()));
            state.z() = p3.x();
            return state.head(3);
        }

        Eigen::Vector2d getVstate(const double& t){
            return minco_traj_.getVel(t);
        }

        Eigen::Vector2d getAstate(const double& t){
            return minco_traj_.getAcc(t);
        }
};


#endif