#ifndef __WRAPPER_H_INCLUDED__
#define __WRAPPER_H_INCLUDED__

#include <iostream>
#include <limits>
#include <algorithm>
#include <sstream>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/conversions.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>


#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <Eigen/Core>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "featDetectorWrapper/objDetection.h"
#include "featDetectorWrapper/detectedObjArray.h"
#include "featDetectorWrapper/classifyObjs.h"

class Wrapper{
	private:
		typedef pcl::PointXYZRGB PCLPoint;
		typedef pcl::PointCloud<PCLPoint> PCLPointCloud;
		ros::NodeHandle node_handle;
		//object detection wrapper server
		ros::ServiceServer detect_wrapper_service;
		
		//the service topic
		std::string service_topic;
		std::string kinect_topic;
		std::string table_topic;
		std::string scene_topic;
		std::string clusters_topic;
		std::string detection_topic;
		std::string frame_id;
		std::string obj_detect_topic;
		double clustering_tolerance_;
		int cluster_min_size_;
  		int cluster_max_size_;
  		double maximum_obj_size;
  		
		ros::Subscriber kinect_subscriber;
		ros::Publisher table_pub;
		ros::Publisher scene_pub;
		ros::Publisher clusters_pub;
		ros::Publisher detection_pub;
		ros::ServiceClient obj_detect_client;
		
		void init();
		bool handleDetectionServiceCall(featDetectorWrapper::objDetection::Request& req, featDetectorWrapper::objDetection::Response& res);
		void subscribeToKinect();
		void unsubscribeFromKinect();
		void publishDetection(const sensor_msgs::PointCloud2::ConstPtr &msg);
		void filterPointcloud(const sensor_msgs::PointCloud2::ConstPtr& original_pc, PCLPointCloud::Ptr& objects_pointcloud, PCLPointCloud::Ptr& table_pointcloud);
		void clusterPointcloud(PCLPointCloud::Ptr& cloud, std::vector<pcl::PointIndices>& cluster_indices);
		void computeCentroid(const PCLPointCloud& cloud, Eigen::Vector3f& centroid);
		sensor_msgs::Image cropCloud(sensor_msgs::PointCloud2::ConstPtr original_cloud, PCLPointCloud::Ptr cloud_filtered, std::vector<int> cluster_indices);
		
	public:
		Wrapper(ros::NodeHandle nh);
		~Wrapper();

};

#endif // __WRAPPER_H_INCLUDED__ 
