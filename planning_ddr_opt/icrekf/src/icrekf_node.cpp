#include "icrekf/icrekf.h"
#include <signal.h>

void MySigintHandler(int sig) {
    ros::shutdown();
    // exit(0);
}

int main(int argc, char** argv){ 
  ros::init(argc, argv, "mpc_node");
  ros::NodeHandle nh;

  ICREKF icrekf(nh);
  signal(SIGINT,MySigintHandler);  
  ros::spin();

  return 0;
}