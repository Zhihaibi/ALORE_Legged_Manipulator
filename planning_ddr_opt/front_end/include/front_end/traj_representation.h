#ifndef _TRAJ_REPRESENTATION_H
#define _TRAJ_REPRESENTATION_H

#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

#include <ros/ros.h>
#include <ros/package.h>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <map>

#define IN_CLOSE_SET 'a'
#define IN_OPEN_SET 'b'
#define NOT_EXPAND 'c'

#define inf 1 >> 30

class PathNode {
public:
  /* -------------------- */
  Eigen::Vector2i index;
  int yaw_idx;
  /* --- the state is x y theta(orientation) */
  Eigen::Vector3d state;
  double g_score, f_score;
  double penalty_score;
  /* control input should be steer and arc */
  Eigen::Vector2d input;
  PathNode* parent;
  // Three states: not expanded, in close set, in open set
  char node_state;
  /* -------------------- */
  PathNode() {
    parent = NULL;
    node_state = NOT_EXPAND;
  }
  ~PathNode(){};
};
typedef PathNode* PathNodePtr;

struct FlatTrajData{

  // All initial values after uniform sampling: yaw, s, t
  std::vector<Eigen::Vector3d> UnOccupied_traj_pts;
  double UnOccupied_initT;
  std::vector<Eigen::Vector3d> UnOccupied_positions;

  Eigen::MatrixXd start_state;   //pva
  Eigen::MatrixXd final_state;   // end flat state (2, 3)

  Eigen::Vector3d start_state_XYTheta;
  Eigen::Vector3d final_state_XYTheta;
  bool if_cut;

  void printFlatTrajData() {
    std::cout << "UnOccupied_traj_pts:" << std::endl;
    for (const auto& pt : UnOccupied_traj_pts) {
      std::cout << pt.transpose() << std::endl;
    }
    std::cout << "UnOccupied_initT: " << UnOccupied_initT << std::endl;
    
    std::cout << "start_state:" << std::endl;
    std::cout << start_state << std::endl;
    std::cout << "final_state:" << std::endl;
    std::cout << final_state << std::endl;
    std::cout << "start_state_XYTheta: " << start_state_XYTheta.transpose() << std::endl;
    std::cout << "final_state_XYTheta: " << final_state_XYTheta.transpose() << std::endl;
    std::cout << "if_cut: " << if_cut << std::endl;
  }
};




#endif
