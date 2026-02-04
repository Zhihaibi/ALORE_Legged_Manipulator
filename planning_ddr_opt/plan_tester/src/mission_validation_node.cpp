#include <ros/ros.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Pose.h>
#include <std_msgs/Int32MultiArray.h>
#include <vector>
#include <sstream>

// 回调函数，用于接收并打印椅子的访问顺序
void chairOrderCallback(const std_msgs::Int32MultiArray::ConstPtr& msg)
{
    std::stringstream ss;
    for (size_t i = 0; i < msg->data.size(); ++i) {
        ss << msg->data[i] << (i == msg->data.size() - 1 ? "" : ", ");
    }
    ROS_INFO("[Validation Node] Received Chair Order: [ %s ]", ss.str().c_str());
}

// 回调函数，用于接收并打印目标的访问顺序
void targetOrderCallback(const std_msgs::Int32MultiArray::ConstPtr& msg)
{
    std::stringstream ss;
    for (size_t i = 0; i < msg->data.size(); ++i) {
        ss << msg->data[i] << (i == msg->data.size() - 1 ? "" : ", ");
    }
    ROS_INFO("[Validation Node] Received Target Order: [ %s ]", ss.str().c_str());
}

int main(int argc, char** argv)
{
    // 初始化ROS节点
    ros::init(argc, argv, "mission_validation_node");
    ros::NodeHandle nh;

    ROS_INFO("Starting Mission Validation Node...");

    // 创建发布者，用于发布物品和目标的位置
    ros::Publisher items_pub = nh.advertise<geometry_msgs::PoseArray>("/mission/items", 1, true); // Latching publisher
    ros::Publisher targets_pub = nh.advertise<geometry_msgs::PoseArray>("/mission/targets", 1, true); // Latching publisher

    // 创建订阅者，用于接收规划结果
    ros::Subscriber chair_order_sub = nh.subscribe("/mission/results/chair_order", 10, chairOrderCallback);
    ros::Subscriber target_order_sub = nh.subscribe("/mission/results/target_order", 10, targetOrderCallback);

    // 等待一秒，确保所有连接都已建立
    ros::Duration(1.0).sleep();


    // --- 创建并发布物品位置 ---
    geometry_msgs::PoseArray items_msg;
    items_msg.header.stamp = ros::Time::now();
    items_msg.header.frame_id = "world";

    // --- 创建并发布目标位置 ---
    geometry_msgs::PoseArray targets_msg;
    targets_msg.header.stamp = ros::Time::now();
    targets_msg.header.frame_id = "world";

    // 保留offset变量，并通过参数读取
    float sim_2_rviz_offset_1 = 0.0;
    float sim_2_rviz_offset_2 = 0.0;
    nh.param<float>("sim_2_rviz_offset_1", sim_2_rviz_offset_1, 0.0);
    nh.param<float>("sim_2_rviz_offset_2", sim_2_rviz_offset_2, 0.0);


    int num_tasks = 0;
    nh.param<int>("num_tasks", num_tasks, 4);

    for (int i = 1; i <= num_tasks; ++i) {
        double item_x, item_y, target_x, target_y;
        
        // 读取Item参数
        if (!nh.getParam("item" + std::to_string(i) + "_x", item_x) || !nh.getParam("item" + std::to_string(i) + "_y", item_y)) {
            ROS_WARN("Parameters for item %d not found. Skipping task %d.", i, i);
            continue;
        }

        // 读取Target参数
        if (!nh.getParam("target" + std::to_string(i) + "_x", target_x) || !nh.getParam("target" + std::to_string(i) + "_y", target_y)) {
            ROS_WARN("Parameters for target %d not found. Skipping task %d.", i, i);
            continue;
        }

        geometry_msgs::Pose item;
        item.position.x = item_x + sim_2_rviz_offset_1;
        item.position.y = item_y + sim_2_rviz_offset_2;
        item.orientation.w = 1.0;
        items_msg.poses.push_back(item);

        geometry_msgs::Pose target;
        target.position.x = target_x + sim_2_rviz_offset_1;
        target.position.y = target_y + sim_2_rviz_offset_2;
        target.orientation.w = 1.0;
        targets_msg.poses.push_back(target);
    }

    if (items_msg.poses.empty() || targets_msg.poses.empty()) {
        ROS_ERROR("No valid tasks found. Please check parameters in launch file.");
        return -1;
    }

    items_pub.publish(items_msg);
    targets_pub.publish(targets_msg);
    ROS_INFO("Published %zu item positions to /mission/items", items_msg.poses.size());
    ROS_INFO("Published %zu target positions to /mission/targets", targets_msg.poses.size());
    
    ROS_INFO("Mission data published. Waiting for results...");

    // 保持节点运行，以接收回调函数的消息
    ros::spin();

    return 0;
}