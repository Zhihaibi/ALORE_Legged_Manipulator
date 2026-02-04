#include "plan_manager/plan_manager.hpp"
#include "visualizer/visualizer.hpp"
#include <signal.h>

void MySigintHandler(int sig) {
    ros::shutdown();
    // exit(0);
}

int main(int argc,char **argv){
  ros::init(argc, argv, "planmanager");
  ros::NodeHandle nh("~");

  PlanManager planmanager(nh);
  signal(SIGINT,MySigintHandler);  
  
  ros::spin();
  return 0;
}