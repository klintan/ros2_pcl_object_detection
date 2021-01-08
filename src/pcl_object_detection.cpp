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

PclObjectDetection::PclObjectDetection(const rclcpp::NodeOptions &node_options) : Node("pcl_object_detection", node_options)
{
}

void kft(const std_msgs::msg::Float32MultiArray ccs)
{
    // First predict, to update the internal statePre variable
    std::vector<cv::Mat> pred{KF0.predict()};

    // Get measurements
    // Extract the position of the clusters from the multi-array. To check if the data
    // coming in, check the .z (every third) coordinate and that will be 0.0
    std::vector<geometry_msgs::Point> cluster_centers; // cluster centers

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
            dist_vec.push_back(euclidean_distance(kf_predictions[filterN], copy_of_cluster_centers[n]));
        }

        dist_mat.push_back(dist_vec);
    }

    for (int cluster_count = 0; cluster_count < 6; cluster_count++)
    {
        // 1. Find min(distMax)==> (i,j);
        std::pair<int, int> min_index(find_index_of_min(dist_mat));
        // 2. objID[i]=clusterCenters[j]; counter++
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

    // convert clusterCenters from geometry_msgs::Point to floats
    std::vector<std::vector<float>> cc;
    for (int i = 0; i < 6; i++)
    {
        vector<float> pt;
        pt.push_back(cluster_centers[obj_ids_[i]].x);
        pt.push_back(cluster_centers[obj_ids_[i]].y);
        pt.push_back(cluster_centers[obj_ids_[i]].z);

        cc.push_back(pt);
    }
    float meas0[2] = {cc[0].at(0), cc[0].at(1)};

    // The update phase
    cv::Mat meas0Mat = cv::Mat(2, 1, CV_32F, meas0);

    if (!(meas0Mat.at<float>(0, 0) == 0.0f || meas0Mat.at<float>(1, 0) == 0.0f))
        Mat estimated0 = KF0.correct(meas0Mat);
}

double euclidean_distance(geometry_msgs::msg::Point &p1, geometry_msgs::msg::Point &p2)
{
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
}

std::pair<int, int> find_index_of_min(std::vector<std::vector<float>> dist_mat)
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

void publish_bbox_marker(std::vector<geometry_msgs::msg::Point> kf_predictions)
{
    visualization_msgs::msg::MarkerArray cluster_markers;

    for (int i = 0; i < 6; i++)
    {
        visualization_msgs::msg::Marker m;

        m.id = i;
        m.type = visualization_msgs::msg::Marker::CUBE;
        m.header.frame_id = "/map";
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

        cluster_markers.markers.push_back(m);
    }

    bbox_markers_pub_.publish(cluster_markers);
}

void publish_object_ids(std::vector<int> obj_ids)
{
    std_msgs::msg::Int32MultiArray obj_ids_msg;
    for (auto it = obj_ids.begin(); it != obj_ids.end(); it++)
    {
        obj_ids_msg.data.push_back(*it);
    }
    object_ids_pub_.publish(obj_ids_msg);
}

void publish_cloud(rclcpp::Publisher &pub, pcl::PointCloud<pcl::PointXYZ>::Ptr cluster)
{
    auto cluster_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
    pcl::toROSMsg(*cluster, *cluster_msg);
    cluster_msg->header.frame_id = "/map";
    cluster_msg->header.stamp = rclcpp::Clock()::now();
    pub.publish(*cluster_msg);
}

void initialize_kalman_filter()
{
    float dvx = 0.01f; //1.0
    float dvy = 0.01f; //1.0
    float dx = 1.0f;
    float dy = 1.0f;
    kf_track_ = std::make_shared<cv::KalmanFilter>(state_dim_, meas_dim_, ctrl_dim_, CV_32F);
    kf_track_->transitionMatrix = (Mat_<float>(4, 4) << dx, 0, 1, 0, 0, dy, 0, 1, 0, 0, dvx, 0, 0, 0, 0, dvy);
    cv::setIdentity(kf_track_->measurementMatrix);

    float sigma_p = 0.01;
    float sigma_q = 0.1;
    cv::setIdentity(kf_track_->processNoiseCov, Scalar::all(sigma_p));
    cv::setIdentity(kf_track_->measurementNoiseCov, cv::Scalar(sigma_q)); //1e-1

    tracks_.push_back(kf_track_)
}

std::vector<pcl::PointIndices> get_cluster_indices(pcl::PointCloud<pcl::PointXYZ> input_cloud)
{
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

    return cluster_indices;
}

void cloud_callback(const sensor_msgs::msg::PointCloud2::ConstPtr &input)
{
    /* Process the point cloud */
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr clustered_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    /* Creating the KdTree from input point cloud*/
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);

    pcl::fromROSMsg(*input, *input_cloud);

    tree->setInputCloud(input_cloud);

    std::vector<pcl::PointIndices> cluster_indices = get_cluster_indices(input_cloud);

    std::vector<pcl::PointIndices>::const_iterator it;
    std::vector<int>::const_iterator pit;

    /* Vector of cluster pointclouds */
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cluster_vec;

    /* Cluster centroids */
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

    // If this is the first frame, initialize kalman filters for the clustered objects
    if (first_frame)
    {
        /* Initialize Kalman Filters, same as number of clusters for tracking */
        initialize_kalman_filter();

        for (it = kf_tracks_.begin(); it != kf_tracks_.end(); ++it)
        {
            kf_tracks_.at(*it).statePre.at<float>(0) = cluster_centroids.at(0).x;
            kf_tracks_.at(*it).statePre.at<float>(1) = cluster_centroids.at(0).y;
            kf_tracks_.at(*it).statePre.at<float>(2) = 0; // initial v_x
            kf_tracks_.at(*it).statePre.at<float>(3) = 0; //initial v_y
        }

        for (i = kf_tracks_.begin(); i != kf_tracks_.end(); ++i)
        {
            geometry_msgs::Point pt;
            pt.x = cluster_centroids.at(i).x;
            pt.y = cluster_centroids.at(i).y;
            prev_cluster_centers.push_back(pt);
        }

        first_frame = false;
    }
    else
    {
        std_msgs::msg::Float32MultiArray cc;
        for (int i = 0; i < 6; i++)
        {
            cc.data.push_back(cluster_centroids.at(i).x);
            cc.data.push_back(cluster_centroids.at(i).y);
            cc.data.push_back(cluster_centroids.at(i).z);
        }

        kft(cc);
        int i = 0;
        bool published_cluster[6];
        for (auto it = obj_ids.begin(); it != obj_ids.end(); it++)
            publish_cloud(pub_cluster0, cluster_vec[*it]);
        published_cluster[i] = true; //Use this flag to publish only once for a given obj ID
        i++;
    }
}
}
