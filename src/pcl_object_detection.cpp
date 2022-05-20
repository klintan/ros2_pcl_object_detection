/*
Copyright (c) 2020 Andreas Klintberg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "pcl_object_detection.hpp"

namespace pcl_object_detection
{

static const rclcpp::Logger LOGGER = rclcpp::get_logger("pcl_object_detection");

PclObjectDetection::PclObjectDetection(
  const rclcpp::NodeOptions& options
): PclObjectDetection("", options)
{}

PclObjectDetection::PclObjectDetection(
  const std::string& name_space,
  const rclcpp::NodeOptions& options
): Node("PclObjectDetection", name_space, options)
{
  RCLCPP_INFO(this->get_logger(),"PclObjectDetection init complete!");

  // Store clock
  clock_ = this->get_clock();

  // use_debug: enable/disable output of a cloud containing object points
  debug_ = this->declare_parameter<bool>("debug_topics", false);

  // frame_id: frame to transform cloud to (should be XY horizontal)
  world_frame_ = this->declare_parameter<std::string>("frame_id", "map");

  // Create a ROS subscriber for the input point cloud
  rclcpp::QoS subscription_qos(1);
  std::function<void(const sensor_msgs::msg::PointCloud2::SharedPtr)> subscription_callback = std::bind(&PclObjectDetection::cloud_callback,
      this, std::placeholders::_1);
  subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "filtered_clouds",
        1,
        subscription_callback);

  // Create a ROS publisher for the output point cloud
  rclcpp::QoS qos(1);
  qos.best_effort();
  pub_cluster0 = this->create_publisher<sensor_msgs::msg::PointCloud2>("cluster_0", 1);
  pub_cluster1 = this->create_publisher<sensor_msgs::msg::PointCloud2>("cluster_1", 1);
  pub_cluster2 = this->create_publisher<sensor_msgs::msg::PointCloud2>("cluster_2", 1);
  pub_cluster3 = this->create_publisher<sensor_msgs::msg::PointCloud2>("cluster_3", 1);
  pub_cluster4 = this->create_publisher<sensor_msgs::msg::PointCloud2>("cluster_4", 1);
  pub_cluster5 = this->create_publisher<sensor_msgs::msg::PointCloud2>("cluster_5", 1);
  // Subscribe to the clustered pointclouds
  // ros::Subscriber c1=nh.subscribe("ccs",100,KFT);
  object_ids_pub_ = this->create_publisher<std_msgs::msg::Int32MultiArray>("obj_id", 1);
  /* Point cloud clustering
   */

  // cc_pos=nh.advertise<std_msgs::Float32MultiArray>("ccs",100);//clusterCenter1
  //bbox_markers_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("viz", qos);

  /* Point cloud clustering
   */
}

void PclObjectDetection::kft(const std_msgs::msg::Float32MultiArray ccs)
{
    // First predict, to update the internal statePre variable
    std::vector<cv::Mat> pred{KF0.predict()};

    // Get measurements
    // Extract the position of the clusters from the multi-array. To check if the data
    // coming in, check the .z (every third) coordinate and that will be 0.0
    std::vector<geometry_msgs::msg::Point> cluster_centers; // cluster centers

    int i = 0;
    for (std::vector<float>::const_iterator it = ccs.data.begin(); it != ccs.data.end(); it += 3)
    {
        geometry_msgs::msg::Point pt;
        pt.x = *it;
        pt.y = *(it + 1);
        pt.z = *(it + 2);

        cluster_centers.push_back(pt);
    }

    std::vector<geometry_msgs::msg::Point> kf_predictions;
    i = 0;
    for (auto it = pred.begin(); it != pred.end(); it++)
    {
        geometry_msgs::msg::Point pt;
        pt.x = (*it).at<float>(0);
        pt.y = (*it).at<float>(1);
        pt.z = (*it).at<float>(2);

        kf_predictions.push_back(pt);
    }

    // Find the cluster that is more probable to be belonging to a given KF.
    obj_ids.clear();   // Clear the obj_id vector
    obj_ids.resize(6); // Allocate default elements so that [i] doesnt segfault. Should be done better
    // Copy cluster centres for modifying it and preventing multiple assignments of the same ID
    std::vector<geometry_msgs::msg::Point> copy_of_cluster_centers(cluster_centers);
    std::vector<std::vector<float>> dist_mat;

    for (int filter_n = 0; filter_n < 6; filter_n++)
    {
        std::vector<float> dist_vec;
        for (int n = 0; n < 6; n++)
        {
            dist_vec.push_back(euclidean_distance(kf_predictions[filter_n], copy_of_cluster_centers[n]));
        }

        dist_mat.push_back(dist_vec);
    }

    for (int cluster_count = 0; cluster_count < 6; cluster_count++)
    {
        // 1. Find min(distMax)==> (i,j);
        std::pair<int, int> min_index(find_index_of_min(dist_mat));
        // 2. obj_ids[i]=clusterCenters[j]; counter++
        obj_ids[min_index.first] = min_index.second;

        // 3. distMat[i,:]=10000; distMat[:,j]=10000
        dist_mat[min_index.first] = std::vector<float>(6, 10000.0); // Set the row to a high number.
        for (int row = 0; row < dist_mat.size(); row++)             //set the column to a high number
        {
            dist_mat[row][min_index.second] = 10000.0;
        }
        // 4. if(counter<6) got to 1.
    }

    publish_bbox_marker(kf_predictions);

    prev_cluster_centers = cluster_centers;

    publish_object_ids(pub_ids);

    // convert clusterCenters from geometry_msgs::msg::Point to floats
    std::vector<std::vector<float>> cc;
    for (int i = 0; i < 6; i++)
    {
        std::vector<float> pt;
        pt.push_back(cluster_centers[obj_ids[i]].x);
        pt.push_back(cluster_centers[obj_ids[i]].y);
        pt.push_back(cluster_centers[obj_ids[i]].z);

        cc.push_back(pt);
    }
    float meas0[2] = {cc[0].at(0), cc[0].at(1)};

    // The update phase
    cv::Mat meas0Mat = cv::Mat(2, 1, CV_32F, meas0);

    if (!(meas0Mat.at<float>(0, 0) == 0.0f || meas0Mat.at<float>(1, 0) == 0.0f))
        cv::Mat estimated0 = KF0.correct(meas0Mat);
}

double PclObjectDetection::euclidean_distance(geometry_msgs::msg::Point &p1, geometry_msgs::msg::Point &p2)
{
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
}

std::pair<int, int> PclObjectDetection::find_index_of_min(std::vector<std::vector<float>> dist_mat)
{
    std::pair<int, int> min_index;
    float min_el = std::numeric_limits<float>::max();
    for (int i = 0; i < dist_mat.size(); i++)
        for (int j = 0; j < dist_mat.at(0).size(); j++)
        {
            if (dist_mat[i][j] < min_el)
            {
                min_el = dist_mat[i][j];
                min_index = std::make_pair(i, j);
            }
        }
    return min_index;
}

void PclObjectDetection::publish_bbox_marker(std::vector<geometry_msgs::msg::Point> kf_predictions)
{
    //visualization_msgs::msg::MarkerArray cluster_markers;

    for (int i = 0; i < 6; i++)
    {
        visualization_msgs::msg::Marker m;

        m.id = i;
        m.type = visualization_msgs::msg::Marker::CUBE;
        m.header.frame_id = "map";
        m.scale.x = 0.3;
        m.scale.y = 0.3;
        m.scale.z = 0.3;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.color.a = 1.0;
        m.color.r = i % 2 ? 1 : 0;
        m.color.g = i % 3 ? 1 : 0;
        m.color.b = i % 4 ? 1 : 0;

        geometry_msgs::msg::Point cluster_c(kf_predictions[i]);
        m.pose.position.x = cluster_c.x;
        m.pose.position.y = cluster_c.y;
        m.pose.position.z = cluster_c.z;

        //cluster_markers.markers.push_back(m);
    }

    //bbox_markers_pub_->publish(cluster_markers);
}

void PclObjectDetection::publish_object_ids(std::vector<int> obj_ids)
{
    std_msgs::msg::Int32MultiArray obj_ids_msg;
    for (auto it = obj_ids.begin(); it != obj_ids.end(); it++)
    {
        obj_ids_msg.data.push_back(*it);
    }
    object_ids_pub_->publish(obj_ids_msg);
}

void PclObjectDetection::publish_cloud(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pub, pcl::PointCloud<pcl::PointXYZ>::Ptr cluster)
{
    auto cluster_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
    pcl::toROSMsg(*cluster, *cluster_msg);
    cluster_msg->header.frame_id = "map";
    //cluster_msg->header.frame_id = cluster->header.frame_id;
    cluster_msg->header.stamp = clock_->now();
    pub->publish(*cluster_msg);
}

void PclObjectDetection::initialize_kalman_filter()
{
    // Initialize 6 Kalman Filters; Assuming 6 max objects in the dataset.
    // Could be made generic by creating a Kalman Filter only when a new object
    // is detected

    float dvx = 0.01f; // 1.0
    float dvy = 0.01f; // 1.0
    float dx = 1.0f;
    float dy = 1.0f;
    KF0.transitionMatrix = (cv::Mat_<float>(4, 4) << dx, 0, 1, 0, 0, dy, 0, 1, 0, 0,
                            dvx, 0, 0, 0, 0, dvy);
    KF1.transitionMatrix = (cv::Mat_<float>(4, 4) << dx, 0, 1, 0, 0, dy, 0, 1, 0, 0,
                            dvx, 0, 0, 0, 0, dvy);
    KF2.transitionMatrix = (cv::Mat_<float>(4, 4) << dx, 0, 1, 0, 0, dy, 0, 1, 0, 0,
                            dvx, 0, 0, 0, 0, dvy);
    KF3.transitionMatrix = (cv::Mat_<float>(4, 4) << dx, 0, 1, 0, 0, dy, 0, 1, 0, 0,
                            dvx, 0, 0, 0, 0, dvy);
    KF4.transitionMatrix = (cv::Mat_<float>(4, 4) << dx, 0, 1, 0, 0, dy, 0, 1, 0, 0,
                            dvx, 0, 0, 0, 0, dvy);
    KF5.transitionMatrix = (cv::Mat_<float>(4, 4) << dx, 0, 1, 0, 0, dy, 0, 1, 0, 0,
                            dvx, 0, 0, 0, 0, dvy);

    cv::setIdentity(KF0.measurementMatrix);
    cv::setIdentity(KF1.measurementMatrix);
    cv::setIdentity(KF2.measurementMatrix);
    cv::setIdentity(KF3.measurementMatrix);
    cv::setIdentity(KF4.measurementMatrix);
    cv::setIdentity(KF5.measurementMatrix);
    // Process Noise Covariance Matrix Q
    // [ Ex 0  0    0 0    0 ]
    // [ 0  Ey 0    0 0    0 ]
    // [ 0  0  Ev_x 0 0    0 ]
    // [ 0  0  0    1 Ev_y 0 ]
    //// [ 0  0  0    0 1    Ew ]
    //// [ 0  0  0    0 0    Eh ]
    float sigmaP = 0.01;
    float sigmaQ = 0.1;
    setIdentity(KF0.processNoiseCov, cv::Scalar::all(sigmaP));
    setIdentity(KF1.processNoiseCov, cv::Scalar::all(sigmaP));
    setIdentity(KF2.processNoiseCov, cv::Scalar::all(sigmaP));
    setIdentity(KF3.processNoiseCov, cv::Scalar::all(sigmaP));
    setIdentity(KF4.processNoiseCov, cv::Scalar::all(sigmaP));
    setIdentity(KF5.processNoiseCov, cv::Scalar::all(sigmaP));
    // Meas noise cov matrix R
    cv::setIdentity(KF0.measurementNoiseCov, cv::Scalar(sigmaQ)); // 1e-1
    cv::setIdentity(KF1.measurementNoiseCov, cv::Scalar(sigmaQ));
    cv::setIdentity(KF2.measurementNoiseCov, cv::Scalar(sigmaQ));
    cv::setIdentity(KF3.measurementNoiseCov, cv::Scalar(sigmaQ));
    cv::setIdentity(KF4.measurementNoiseCov, cv::Scalar(sigmaQ));
    cv::setIdentity(KF5.measurementNoiseCov, cv::Scalar(sigmaQ));
}

void PclObjectDetection::cloud_callback(const sensor_msgs::msg::PointCloud2::ConstPtr &input)
{
    // If this is the first frame, initialize kalman filters for the clustered objects
    if (first_frame)
    {
        /* Initialize Kalman Filters, same as number of clusters for tracking */
        initialize_kalman_filter();

        /* Process the point cloud */
        pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr clustered_cloud(new pcl::PointCloud<pcl::PointXYZ>);

        /* Creating the KdTree from input point cloud*/
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);

        pcl::fromROSMsg(*input, *input_cloud);

        tree->setInputCloud(input_cloud);


        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.08);
        ec.setMinClusterSize(10);
        ec.setMaxClusterSize(600);
        ec.setSearchMethod(tree);
        ec.setInputCloud(input_cloud);
        /* Extract the clusters out of pc and save indices in cluster_indices.*/
        ec.extract(cluster_indices);

        std::vector<pcl::PointIndices>::const_iterator it;
        std::vector<int>::const_iterator pit;
        // Vector of cluster pointclouds
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cluster_vec;
        // Cluster centroids
        std::vector<pcl::PointXYZ> cluster_centroids;

        for (it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
            float x = 0.0;
            float y = 0.0;
            int num_pts = 0;
            for (pit = it->indices.begin(); pit != it->indices.end(); pit++)
            {
                //TODO Make 3D?
                cloud_cluster->points.push_back(input_cloud->points[*pit]);
                x += input_cloud->points[*pit].x;
                y += input_cloud->points[*pit].y;
                num_pts++;
            }

            pcl::PointXYZ centroid;
            centroid.x = x / num_pts;
            centroid.y = y / num_pts;
            centroid.z = 0.0;

            cluster_vec.push_back(cloud_cluster);

            //Get the centroid of the cluster
            cluster_centroids.push_back(centroid);
        }

        // Ensure at least 6 clusters exist to publish (later clusters may be empty)
        while (cluster_vec.size() < 6) {
          pcl::PointCloud<pcl::PointXYZ>::Ptr empty_cluster(
              new pcl::PointCloud<pcl::PointXYZ>);
          empty_cluster->points.push_back(pcl::PointXYZ(0, 0, 0));
          cluster_vec.push_back(empty_cluster);
        }

        while (cluster_centroids.size() < 6) {
          pcl::PointXYZ centroid;
          centroid.x = 0.0;
          centroid.y = 0.0;
          centroid.z = 0.0;

          cluster_centroids.push_back(centroid);
        }

        // Set initial state
        KF0.statePre.at<float>(0) = cluster_centroids.at(0).x;
        KF0.statePre.at<float>(1) = cluster_centroids.at(0).y;
        KF0.statePre.at<float>(2) = 0; // initial v_x
        KF0.statePre.at<float>(3) = 0; // initial v_y

        // Set initial state
        KF1.statePre.at<float>(0) = cluster_centroids.at(1).x;
        KF1.statePre.at<float>(1) = cluster_centroids.at(1).y;
        KF1.statePre.at<float>(2) = 0; // initial v_x
        KF1.statePre.at<float>(3) = 0; // initial v_y

        // Set initial state
        KF2.statePre.at<float>(0) = cluster_centroids.at(2).x;
        KF2.statePre.at<float>(1) = cluster_centroids.at(2).y;
        KF2.statePre.at<float>(2) = 0; // initial v_x
        KF2.statePre.at<float>(3) = 0; // initial v_y

        // Set initial state
        KF3.statePre.at<float>(0) = cluster_centroids.at(3).x;
        KF3.statePre.at<float>(1) = cluster_centroids.at(3).y;
        KF3.statePre.at<float>(2) = 0; // initial v_x
        KF3.statePre.at<float>(3) = 0; // initial v_y

        // Set initial state
        KF4.statePre.at<float>(0) = cluster_centroids.at(4).x;
        KF4.statePre.at<float>(1) = cluster_centroids.at(4).y;
        KF4.statePre.at<float>(2) = 0; // initial v_x
        KF4.statePre.at<float>(3) = 0; // initial v_y

        // Set initial state
        KF5.statePre.at<float>(0) = cluster_centroids.at(5).x;
        KF5.statePre.at<float>(1) = cluster_centroids.at(5).y;
        KF5.statePre.at<float>(2) = 0; // initial v_x
        KF5.statePre.at<float>(3) = 0; // initial v_y

        first_frame = false;

        /*  // Print the initial state of the kalman filter for debugging
        cout<<"KF0.satePre="<<KF0.statePre.at<float>(0)<<","<<KF0.statePre.at<float>(1)<<"\n";
        cout<<"KF1.satePre="<<KF1.statePre.at<float>(0)<<","<<KF1.statePre.at<float>(1)<<"\n";
        cout<<"KF2.satePre="<<KF2.statePre.at<float>(0)<<","<<KF2.statePre.at<float>(1)<<"\n";
        cout<<"KF3.satePre="<<KF3.statePre.at<float>(0)<<","<<KF3.statePre.at<float>(1)<<"\n";
        cout<<"KF4.satePre="<<KF4.statePre.at<float>(0)<<","<<KF4.statePre.at<float>(1)<<"\n";
        cout<<"KF5.satePre="<<KF5.statePre.at<float>(0)<<","<<KF5.statePre.at<float>(1)<<"\n";
        //cin.ignore();// To be able to see the printed initial state of the
        KalmanFilter
        */
        RCLCPP_INFO(this->get_logger(), "First time filters created");
        RCLCPP_INFO(this->get_logger(), "first_frame set to false");
    } else {
        /* Process the point cloud */
        pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr clustered_cloud(new pcl::PointCloud<pcl::PointXYZ>);

        /* Creating the KdTree from input point cloud*/
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);

        pcl::fromROSMsg(*input, *input_cloud);

        tree->setInputCloud(input_cloud);

        /* Here we are creating a vector of PointIndices, which contains the actual
         * index information in a vector<int>. The indices of each detected cluster
         * are saved here. Cluster_indices is a vector containing one instance of
         * PointIndices for each detected cluster. Cluster_indices[0] contain all
         * indices of the first cluster in input point cloud.
         */
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.08);
        ec.setMinClusterSize(10);
        ec.setMaxClusterSize(600);
        ec.setSearchMethod(tree);
        ec.setInputCloud(input_cloud);
        //RCLCPP_INFO(this->get_logger(), "PCL init successfull");
        /* Extract the clusters out of pc and save indices in cluster_indices.*/
        ec.extract(cluster_indices);
        //RCLCPP_INFO(this->get_logger(), "PCL extract successfull");
        /* To separate each cluster out of the vector<PointIndices> we have to
         * iterate through cluster_indices, create a new PointCloud for each
         * entry and write all points of the current cluster in the PointCloud.
         */
        // pcl::PointXYZ origin (0,0,0);
        // float mindist_this_cluster = 1000;
        // float dist_this_point = 1000;

        std::vector<pcl::PointIndices>::const_iterator it;
        std::vector<int>::const_iterator pit;

        /* Vector of cluster pointclouds */
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cluster_vec;

        /* Cluster centroids */
        std::vector<pcl::PointXYZ> cluster_centroids;

        RCLCPP_INFO(this->get_logger(), "cluster_indices = %d %d %d", cluster_indices.end() - cluster_indices.begin(), cluster_indices.begin(), cluster_indices.end());

        for (it = cluster_indices.begin(); it != cluster_indices.end(); ++it) {
            float x = 0.0;
            float y = 0.0;
            int numPts = 0;
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(
              new pcl::PointCloud<pcl::PointXYZ>);
            for (pit = it->indices.begin(); pit != it->indices.end(); pit++) {

            cloud_cluster->points.push_back(input_cloud->points[*pit]);

            x += input_cloud->points[*pit].x;
            y += input_cloud->points[*pit].y;
            numPts++;

            // dist_this_point = pcl::geometry::distance(input_cloud->points[*pit],
            //                                          origin);
            // mindist_this_cluster = std::min(dist_this_point,
            // mindist_this_cluster);
            }

            pcl::PointXYZ centroid;
            centroid.x = x / numPts;
            centroid.y = y / numPts;
            centroid.z = 0.0;

            cluster_vec.push_back(cloud_cluster);

            // Get the centroid of the cluster
            cluster_centroids.push_back(centroid);
        }
        //RCLCPP_INFO(this->get_logger(), "cluster_vec got some clusters");

        // Ensure at least 6 clusters exist to publish (later clusters may be empty)
        while (cluster_vec.size() < 6) {
          pcl::PointCloud<pcl::PointXYZ>::Ptr empty_cluster(
              new pcl::PointCloud<pcl::PointXYZ>);
          empty_cluster->points.push_back(pcl::PointXYZ(0, 0, 0));
          cluster_vec.push_back(empty_cluster);
        }

        while (cluster_centroids.size() < 6) {
          pcl::PointXYZ centroid;
          centroid.x = 0.0;
          centroid.y = 0.0;
          centroid.z = 0.0;

          cluster_centroids.push_back(centroid);
        }

        std_msgs::msg::Float32MultiArray cc;
        for (int i = 0; i < 6; i++) {
          cc.data.push_back(cluster_centroids.at(i).x);
          cc.data.push_back(cluster_centroids.at(i).y);
          cc.data.push_back(cluster_centroids.at(i).z);
        }
        //RCLCPP_INFO(this->get_logger(), "6 clusters initialized");

        // cc_pos.publish(cc);// Publish cluster mid-points.
        kft(cc);
        int i = 0;
        bool publishedCluster[6];
        for (auto it = obj_ids.begin(); it != obj_ids.end(); it++) {
             //RCLCPP_INFO(this->get_logger(), "Inside the for loop");

          switch (i) {
             //RCLCPP_INFO(this->get_logger(), "Inside the switch case");
          case 0: {
            publish_cloud(pub_cluster0, cluster_vec[*it]);
            publishedCluster[i] =
                true; // Use this flag to publish only once for a given obj ID
            i++;
            break;
          }
          case 1: {
            publish_cloud(pub_cluster1, cluster_vec[*it]);
            publishedCluster[i] =
                true; // Use this flag to publish only once for a given obj ID
            i++;
            break;
          }
          case 2: {
            publish_cloud(pub_cluster2, cluster_vec[*it]);
            publishedCluster[i] =
                true; // Use this flag to publish only once for a given obj ID
            i++;
            break;
          }
          case 3: {
            publish_cloud(pub_cluster3, cluster_vec[*it]);
            publishedCluster[i] =
                true; // Use this flag to publish only once for a given obj ID
            i++;
            break;
          }
          case 4: {
            publish_cloud(pub_cluster4, cluster_vec[*it]);
            publishedCluster[i] =
                true; // Use this flag to publish only once for a given obj ID
            i++;
            break;
          }

          case 5: {
            publish_cloud(pub_cluster5, cluster_vec[*it]);
            publishedCluster[i] =
                true; // Use this flag to publish only once for a given obj ID
            i++;
            break;
          }
          default:
            break;
        }
     }
  }
}

}  // namespace pcl_object_detection
