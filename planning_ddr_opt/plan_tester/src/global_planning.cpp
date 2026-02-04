#include "plan_tester/plan_tester.hpp"
#include "visualizer/visualizer.hpp"
#include <signal.h>

void MySigintHandler(int sig) {
    ros::shutdown();
    // exit(0);
}

int main(int argc,char **argv){
  ros::init(argc, argv, "plantester");
  ros::NodeHandle nh("~");

  PlanTester plantester(nh);
  signal(SIGINT,MySigintHandler);

  ros::spin();
  return 0;
}