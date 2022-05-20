#ifndef PCL_OBJECT_DETECTION_HPP
#define PCL_OBJECT_DETECTION_HPP

#include <chrono>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <limits>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/clock.hpp"
#include <rclcpp/logging.hpp>
#include <rclcpp/time.hpp>

#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include "std_msgs/msg/float32_multi_array.hpp"
#include "std_msgs/msg/int32_multi_array.hpp"
#include "std_msgs/msg/string.hpp"

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include "visualization_msgs/msg/marker.hpp"

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <pcl/common/centroid.h>
#include <pcl/common/geometry.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>

#include "opencv2/video/tracking.hpp"

namespace pcl_object_detection
{

// KF init
int state_dim_ = 4; // [x,y,v_x,v_y]//,w,h]
int meas_dim_ = 2;  // [z_x,z_y,z_w,z_h]
int ctrl_dim_ = 0;
cv::KalmanFilter KF0(state_dim_, meas_dim_, ctrl_dim_, CV_32F);
cv::KalmanFilter KF1(state_dim_, meas_dim_, ctrl_dim_, CV_32F);
cv::KalmanFilter KF2(state_dim_, meas_dim_, ctrl_dim_, CV_32F);
cv::KalmanFilter KF3(state_dim_, meas_dim_, ctrl_dim_, CV_32F);
cv::KalmanFilter KF4(state_dim_, meas_dim_, ctrl_dim_, CV_32F);
cv::KalmanFilter KF5(state_dim_, meas_dim_, ctrl_dim_, CV_32F);

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

class PclObjectDetection : public rclcpp::Node{
public:
    PclObjectDetection(
        const rclcpp::NodeOptions& options=rclcpp::NodeOptions()
    );
    PclObjectDetection(
        const std::string& name_space,
        const rclcpp::NodeOptions& options=rclcpp::NodeOptions()
    );
private:
    void publish_cloud(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pub, pcl::PointCloud<pcl::PointXYZ>::Ptr cluster);
    void kft(const std_msgs::msg::Float32MultiArray ccs);
    double euclidean_distance(geometry_msgs::msg::Point &p1, geometry_msgs::msg::Point &p2);
    std::pair<int, int> find_index_of_min(std::vector<std::vector<float>> dist_mat);
    void publish_bbox_marker(std::vector<geometry_msgs::msg::Point> kf_predictions);
    void publish_object_ids(std::vector<int> obj_ids);
    void initialize_kalman_filter();
    void cloud_callback(const sensor_msgs::msg::PointCloud2::ConstPtr &input);
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;


    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cluster0;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cluster1;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cluster2;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cluster3;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cluster4;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cluster5;

    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr bbox_markers_pub_;
    rclcpp::Publisher<std_msgs::msg::Int32MultiArray>::SharedPtr object_ids_pub_;

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;

    bool first_frame = true;
    bool debug_;
    std::string world_frame_;
    rclcpp::Clock::SharedPtr clock_;



    rclcpp::TimerBase::SharedPtr timer_;

    pcl::PointCloud<pcl::PointXYZ> cluster;
    std::vector<geometry_msgs::msg::Point> prev_cluster_centers;

    std::vector<int> obj_ids;
    std::vector<int> pub_ids;
};

}


#endif
