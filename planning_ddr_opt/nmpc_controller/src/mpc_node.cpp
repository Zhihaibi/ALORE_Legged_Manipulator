#include "nmpc_controller/mpc.h"
#include <signal.h>

void MySigintHandler(int sig) {
    ros::shutdown();
    // exit(0);
}

int main(int argc, char** argv){ 
  ros::init(argc, argv, "mpc_node");
  ros::NodeHandle nh("~");

  MpcController mpccontroller(nh);
  signal(SIGINT,MySigintHandler);  
  ros::spin();

  return 0;
}