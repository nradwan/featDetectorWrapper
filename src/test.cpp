#include <iostream>
#include <limits>
#include <algorithm>

#include <pcl_conversions/pcl_conversions.h>
#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <Eigen/Core>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>



int main (int argc, char** argv){
	
	// Read in the cloud data
	pcl::PCDReader reader;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>), cloud_f (new pcl::PointCloud<pcl::PointXYZRGB>);
	reader.read ("/home/noha/Documents/Hiwi/table_scene_mug_stereo_textured.pcd", *cloud);
	std::cout << "PointCloud before filtering has: " << cloud->points.size () << " data points." << std::endl; //*
	sensor_msgs::PointCloud2 cloud_msg;
	pcl::toROSMsg(*cloud, cloud_msg);
	sensor_msgs::Image cloud_image;
	pcl::toROSMsg(cloud_msg, cloud_image);
	cv_bridge::CvImagePtr cv_ptr;
	try{
		cv_ptr = cv_bridge::toCvCopy(cloud_image, sensor_msgs::image_encodings::BGR8);
	}
	catch(cv_bridge::Exception& e){
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return 0;
	}

	// Create the filtering object: downsample the dataset using a leaf size of 1cm
	pcl::VoxelGrid<pcl::PointXYZRGB> vg;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
	vg.setInputCloud (cloud);
	vg.setLeafSize (0.01f, 0.01f, 0.01f);
	vg.filter (*cloud_filtered);
	std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size ()  << " data points." << std::endl; //*

	// Create the segmentation object for the planar model and set all the parameters
	pcl::SACSegmentation<pcl::PointXYZRGB> seg;
	pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
	pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZRGB> ());
	pcl::PCDWriter writer;
	seg.setOptimizeCoefficients (true);
	seg.setModelType (pcl::SACMODEL_PLANE);
	seg.setMethodType (pcl::SAC_RANSAC);
	seg.setMaxIterations (100);
	seg.setDistanceThreshold (0.02);

	int i=0, nr_points = (int) cloud_filtered->points.size ();
	while (cloud_filtered->points.size () > 0.3 * nr_points)
	{
		// Segment the largest planar component from the remaining cloud
		seg.setInputCloud (cloud_filtered);
		seg.segment (*inliers, *coefficients);
		if (inliers->indices.size () == 0)
		{
			std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
			break;
		}

		// Extract the planar inliers from the input cloud
		pcl::ExtractIndices<pcl::PointXYZRGB> extract;
		extract.setInputCloud (cloud_filtered);
		extract.setIndices (inliers);
		extract.setNegative (false);

		// Get the points associated with the planar surface
		extract.filter (*cloud_plane);
		std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;

		// Remove the planar inliers, extract the rest
		extract.setNegative (true);
		extract.filter (*cloud_f);
		*cloud_filtered = *cloud_f;
	}

	// Creating the KdTree object for the search method of the extraction
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
	tree->setInputCloud (cloud_filtered);

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
	ec.setClusterTolerance (0.02); // 2cm
	ec.setMinClusterSize (100);
	ec.setMaxClusterSize (25000);
	ec.setSearchMethod (tree);
	ec.setInputCloud (cloud_filtered);
	ec.extract (cluster_indices);

	int j = 0;
	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it){
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
		for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit){
			cloud_cluster->points.push_back (cloud_filtered->points[*pit]); //*
			//std::cout << cloud_filtered->points[*pit] << std::endl;
		}
		cloud_cluster->width = cloud_cluster->points.size ();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;

		std::cout << "PointCloud representing the Cluster " << j << ": " << cloud_cluster->points.size () << " data points." << std::endl;
		std::stringstream ss;
		ss << "/home/noha/Documents/Hiwi/cloud_cluster_" << j << ".pcd";
		writer.write<pcl::PointXYZRGB> (ss.str (), *cloud_cluster, false); //*
		
		
		//check if the pointcloud we are looking for
		if(j == 3){
			double min_x = std::numeric_limits<double>::infinity();
			pcl::PointXYZRGB min_x_pt;
			double min_y = std::numeric_limits<double>::infinity();
			pcl::PointXYZRGB min_y_pt;
			double max_x = -1 * std::numeric_limits<double>::infinity();
			pcl::PointXYZRGB max_x_pt;
			double max_y = -1 * std::numeric_limits<double>::infinity();
			pcl::PointXYZRGB max_y_pt;
			
			//get the min and max cluster indices
			for(std::vector<int>::const_iterator c_it = it->indices.begin();
					c_it != it->indices.end(); ++c_it){
				double curr_x = cloud_filtered->points[*c_it].x;
				double curr_y = cloud_filtered->points[*c_it].y;
				//std::cout << curr_x << "," << curr_y << std::endl;
				//std::cout << "curr: " << curr_x << ", " << curr_y << std::endl;
				if(curr_x < min_x){
					min_x = curr_x;
					min_x_pt = cloud_filtered->points[*c_it];
				}		
				if(curr_x > max_x){
					max_x = curr_x;
					max_x_pt = cloud_filtered->points[*c_it];
				}
				if(curr_y < min_y){
					min_y = curr_y;
					min_y_pt = cloud_filtered->points[*c_it];
				}
				if(curr_y > max_y){
					max_y = curr_y;
					max_y_pt = cloud_filtered->points[*c_it];
				}
			}
			
			std::cout << "min: " << min_x_pt << " and " << min_y_pt << std::endl;
			std::cout << "max: " << max_x_pt << " and " << max_y_pt << std::endl;
			
			double threshold_ = 0.0001;
			
			//loop over the original pointcloud
			int r = 0, c = 0;
			bool found_min_x = false, found_min_y = false;
			int min_x_r, min_x_c, min_y_r, min_y_c;
			int max_x_r, max_x_c, max_y_r, max_y_c;
			for(pcl::PointCloud<pcl::PointXYZRGB>::iterator o_it = cloud->begin(); 
				o_it != cloud->end(); ++o_it){
				//TODO: fix problem with finding the cloud (PROBABLY NEEDS RANGE TO BE SET)
				//take the first point found and ignore the rest for the min
				if(o_it->x >= (min_x_pt.x - threshold_) && o_it->x <= (min_x_pt.x) && ! found_min_x){
					//std::cout << "case 1: found: " << *o_it << " at " << r << "," << c << std::endl;
					found_min_x = true;
					min_x_r = r;
					min_x_c = c;
				}	
				//same for the min y
				if(o_it->y >= (min_y_pt.y - threshold_) && o_it->y <= (min_y_pt.y) && ! found_min_y){
					//std::cout << "case 2: found: " << *o_it << " at " << r << "," << c << std::endl;
					found_min_y = true;
					min_y_r = r;
					min_y_c = c;
				}
				//for max point take the last one
				if(o_it->x >= (max_x_pt.x) && o_it->x <= (max_x_pt.x + threshold_)){
					//std::cout << "case 3: found: " << *o_it << " at " << r << "," << c << std::endl;
					max_x_r = r;
					max_x_c = c;
				}
				if(o_it->y >= (max_y_pt.y) && o_it->y <= (max_y_pt.y + threshold_)){
					//std::cout << "case 4: found: " << *o_it << " at " << r << "," << c << std::endl;
					max_y_r = r;
					max_y_c = c;
				}
				r++;
				if(r == cloud->width){
					r = 0;
					c++;
				}
			}	
			std::cout << "min_x: " << min_x_r << "," << min_x_c << std::endl;
			std::cout << "min_y: " << min_y_r << "," << min_y_c << std::endl;
			std::cout << "max_x: " << max_x_r << "," << max_x_c << std::endl;
			std::cout << "max_y: " << max_y_r << "," << max_y_c << std::endl;
			int start_r = std::min(min_x_r, min_y_r);
			int start_c = std::min(min_x_c, min_y_c);
			int end_r = std::max(max_x_r, max_y_r);
			int end_c = std::max(max_x_c, max_y_c);
			int width = end_r - start_r;
			int height = end_c - start_c;
		
			//crop the image around these coordinates
			cv::Mat new_img(cv_ptr->image, cv::Rect(start_r, start_c, width, height));
			cv::namedWindow("Display Window", cv::WINDOW_AUTOSIZE);
			cv::imshow("Display Window", new_img);
			cv::namedWindow("Original Window", cv::WINDOW_AUTOSIZE);
			cv::imshow("Original Window", cv_ptr->image);
			cv::waitKey(0);
		}
		
		j++;
	}
	
	

	return (0);
}

