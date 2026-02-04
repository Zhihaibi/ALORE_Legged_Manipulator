#include "simulator/simulator.h"
#include <signal.h>

void MySigintHandler(int sig) {
    ros::shutdown();
    // exit(0);
}

int main(int argc, char **argv){
  ros::init(argc, argv, "simulator");
  ros::NodeHandle nh("~");
  
  Simulation simulation(nh);
  signal(SIGINT,MySigintHandler);  

  ros::spin();

  return 0;
}
