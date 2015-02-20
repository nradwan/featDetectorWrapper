#include "featDetectorWrapper/wrapper.h"

Wrapper::Wrapper(ros::NodeHandle nh):
				node_handle(nh) {
	init();				
}

Wrapper::~Wrapper(){

}

void Wrapper::init(){
	
	//initialize the wrapper service topic
	service_topic = "ObjDetectWrapper";
	kinect_topic = "/camera/depth_registered/points";
	table_topic = "TablePlane";
	scene_topic = "FilteredScene";
	clusters_topic = "Clusters";
	detection_topic = "Detection";
	frame_id = "/camera_depth_optical_frame";
	obj_detect_topic = "classifyObjs";
	clustering_tolerance_ = 0.025;//0.025;
	cluster_min_size_ =  50;//50;
  	cluster_max_size_ = 15000;
  	maximum_obj_size = 50;//100; //(50 cm)
	
	//announce the object detection service
	detect_wrapper_service = node_handle.advertiseService(service_topic, &Wrapper::handleDetectionServiceCall, this);
	ROS_INFO("Announced service: %s", service_topic.c_str());
	//announce the table publisher
	table_pub = node_handle.advertise<sensor_msgs::PointCloud2>(table_topic, 1);
	ROS_INFO("Table point cloud published on topic: %s", table_topic.c_str());
	//announce the scene publisher
	scene_pub = node_handle.advertise<sensor_msgs::PointCloud2>(scene_topic, 1);
	ROS_INFO("Filtered scene point cloud published on topic: %s", scene_topic.c_str());
	//announce the clusters publisher
	clusters_pub = node_handle.advertise<visualization_msgs::MarkerArray>(clusters_topic, 1);
	ROS_INFO("Clusters published on topic: %s", clusters_topic.c_str());
	//announce the detection publisher
	detection_pub = node_handle.advertise<featDetectorWrapper::detectedObjArray>(detection_topic, 1);
	ROS_INFO("Detected objects published on topic: %s", detection_topic.c_str());
	//initialize the service client
	obj_detect_client = node_handle.serviceClient<featDetectorWrapper::classifyObjs>(obj_detect_topic);
	ROS_INFO("Object detection client created for topic: %s", obj_detect_topic.c_str());
}

bool Wrapper::handleDetectionServiceCall(featDetectorWrapper::objDetection::Request& req, 
											featDetectorWrapper::objDetection::Response& res){
	if(req.command == featDetectorWrapper::objDetection::Request::OBJDET_SUBSCRIBE){
		//subscribe to kinect
		subscribeToKinect();
	}
	else{
		if(req.command == featDetectorWrapper::objDetection::Request::OBJDET_UNSUBSCRIBE){
			//unsubscribe from kinect
			unsubscribeFromKinect();
		}
		else{
			ROS_ERROR("Cannot recognize command in handleClassificationServiceCall.");
			res.result = featDetectorWrapper::objDetectionResponse::FAILURE;
			return false;
		}
	}
	res.result = featDetectorWrapper::objDetectionResponse::SUCCESS;
	return true;							
}

void Wrapper::subscribeToKinect(){
	kinect_subscriber = node_handle.subscribe(kinect_topic, 1, &Wrapper::publishDetection, this);
	ROS_INFO("Subscribed to: %s", kinect_topic.c_str());
}

void Wrapper::unsubscribeFromKinect(){
	kinect_subscriber.shutdown();
	ROS_WARN("UNsubscribed from: %s", kinect_topic.c_str());
}

void Wrapper::publishDetection(const sensor_msgs::PointCloud2::ConstPtr &msg){
	// Segment out the table plane. Return a point cloud of the table and one of the objects.
	PCLPointCloud::Ptr objs_cloud(new PCLPointCloud);
	PCLPointCloud::Ptr table_cloud(new PCLPointCloud);
	filterPointcloud(msg, objs_cloud, table_cloud);
	if(objs_cloud->points.empty()){
		std::cout << "no objs. leaving" << std::endl;
		return;
	}
	
	//publish the table cloud
	sensor_msgs::PointCloud2 table_cloud_msg;
	pcl::toROSMsg(*table_cloud, table_cloud_msg);
	table_cloud_msg.header.frame_id = frame_id;
	table_pub.publish(table_cloud_msg);
	
	//publish the objs cloud
	sensor_msgs::PointCloud2 objs_cloud_msg;
	pcl::toROSMsg(*objs_cloud, objs_cloud_msg);
	objs_cloud_msg.header.frame_id = frame_id;
	scene_pub.publish(objs_cloud_msg);
	
	// Cluster the objects in the objs_cloud
	visualization_msgs::MarkerArray detected_centroids;
	//Cluster the objects
	std::vector<pcl::PointIndices> cluster_indices;
	std::vector<pcl::PointIndices>::const_iterator it;
	clusterPointcloud(objs_cloud, cluster_indices);
	std::cout << "found " << cluster_indices.size() << " clusters" << std::endl;
	std::vector<PCLPointCloud::Ptr> new_clusters; // collect new clusters
	std::vector<Eigen::Vector3f> new_cluster_centroids; // collect new cluster centroids
	int id = 0;
	
	std::vector<sensor_msgs::Image> cluster_images;
	
	for (it = cluster_indices.begin(); it != cluster_indices.end(); ++it) {
		PCLPointCloud::Ptr single_cluster(new PCLPointCloud);
		single_cluster->width = it->indices.size();
		single_cluster->height = 1;
		single_cluster->points.reserve(it->indices.size());
		std::vector<int>::const_iterator pit;
		for (pit = it->indices.begin(); pit != it->indices.end(); ++pit) {
			PCLPoint& pnt = objs_cloud->points[*pit];
			single_cluster->points.push_back(pnt);
		}
    	new_clusters.push_back(single_cluster);
		// Compute centroid
 		Eigen::Vector3f new_centr;
		computeCentroid(*single_cluster, new_centr);
		new_cluster_centroids.push_back(new_centr);
		//add object centroid to marker array
		visualization_msgs::Marker curr_obj;
		curr_obj.header.frame_id = frame_id;
		curr_obj.header.stamp = ros::Time();
		curr_obj.id = id;
		curr_obj.type = visualization_msgs::Marker::SPHERE;
		curr_obj.action = visualization_msgs::Marker::ADD;
		curr_obj.pose.position.x = new_centr(0);
		curr_obj.pose.position.y = new_centr(1);
		curr_obj.pose.position.z = new_centr(2);
		curr_obj.scale.x = 0.1;
		curr_obj.scale.y = 0.1;
		curr_obj.scale.z = 0.1;
		curr_obj.color.a = 1.0;
		curr_obj.color.r = 1.0;
		curr_obj.color.g = 0.0;
		curr_obj.color.b = 0.0;
		detected_centroids.markers.push_back(curr_obj);
		//crop the pointcloud around the cluster
		//cv::Point circle_ctr(new_centr(0), new_centr(1));
		//sensor_msgs::Image cropped_cluster = cropCloud(msg, objs_cloud, it->indices, circle_ctr);
		sensor_msgs::Image cropped_cluster = cropCloud2(*single_cluster, msg);
		cluster_images.push_back(cropped_cluster);
		
		id++;
	}
	
	//publish the clustered objects
	clusters_pub.publish(detected_centroids);
	
	//prepare the clusters for service call
	std::vector<sensor_msgs::PointCloud2> cluster_clouds;
	for(std::vector<PCLPointCloud::Ptr>::iterator ix = new_clusters.begin();
			ix != new_clusters.end(); ++ix){
		sensor_msgs::PointCloud2 curr_cluster;
		//sensor_msgs::Image curr_img_cluster;
		pcl::toROSMsg(**ix, curr_cluster);
		//pcl::toROSMsg(curr_cluster, curr_img_cluster);
		cluster_clouds.push_back(curr_cluster);
		//cluster_images.push_back(curr_img_cluster);		
	}
	
	//create the service call
	/*featDetectorWrapper::classifyObjs srv;
	srv.request.cloud_clusters = cluster_clouds;
	srv.request.image_clusters = cluster_images;
	if(obj_detect_client.call(srv)){
		ROS_INFO("Object Detection service called");
		detection_pub.publish(srv.response.detected_classes);
	}
	else{
		ROS_ERROR("Failure to call the object detection service");
		return;
	}*/
	return;
}

void Wrapper::filterPointcloud(const sensor_msgs::PointCloud2::ConstPtr& original_pc, PCLPointCloud::Ptr& objects_pointcloud, PCLPointCloud::Ptr& table_pointcloud){
// Filtering purposes. Remove points with z > max_z
	double max_z = 2.5;//0.9 + 0.5;
	double min_x = 0.01;// Aachen
	double max_x = 0.65;//1.0;//0.65;  
	double min_y = -0.05;//0.01;// Aachen
	double max_y = 2.0;//1.5;
	//remove all points with z < table_height
	double table_height = -0.5;//0.05;//-0.03; // landmark
	double plane_thresh = 0.01;//0.03;


	//////////////////////////// Filter to reduce the resolution of the cloud ////////////////////////////

	pcl::PCLPointCloud2 filtered_cloud_msg;
	pcl::PCLPointCloud2::Ptr n_original_pc(new pcl::PCLPointCloud2());
	pcl_conversions::toPCL(*original_pc, *n_original_pc);
	pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
	sor.setInputCloud(n_original_pc);

	sor.setLeafSize(0.01, 0.01, 0.01);
	sor.filter(filtered_cloud_msg);



	//////////////////////////// msg -> pcl ////////////////////////////
	PCLPointCloud::Ptr scene_pcl(new PCLPointCloud);
	pcl::fromPCLPointCloud2(filtered_cloud_msg, *scene_pcl);	
	//cout << "Size before = " << scene_pcl->points.size() << endl;


	//////////////////////////// Remove Outliers ////////////////////////////
	sensor_msgs::PointCloud2::Ptr scene_msg(new sensor_msgs::PointCloud2);
	pcl::toROSMsg(*scene_pcl, *scene_msg);


	pcl::PCLPointCloud2 clean_pcl_msg;
  	pcl::StatisticalOutlierRemoval<pcl::PCLPointCloud2> outlier_rem;
	pcl::PCLPointCloud2::Ptr out_rem_pcl(new pcl::PCLPointCloud2());
	pcl_conversions::toPCL(*scene_msg, *out_rem_pcl);
	outlier_rem.setInputCloud(out_rem_pcl);

	outlier_rem.setMeanK(50);
	outlier_rem.setStddevMulThresh(1.0);
	outlier_rem.filter(clean_pcl_msg);

	pcl::fromPCLPointCloud2(clean_pcl_msg, *scene_pcl);

	//////////////////////////// Filter out far away points ////////////////////////////
	// TODO Use table plane normal as the z axis and filter out everything below it.
	PCLPointCloud::Ptr scene_filtered(new PCLPointCloud);
	pcl::PassThrough<pcl::PointXYZRGB> pass;
	pass.setInputCloud(boost::make_shared < pcl::PointCloud<pcl::PointXYZRGB> > (*scene_pcl));
	pass.setFilterFieldName("z");
	pass.setFilterLimits(table_height, max_z);	
	//pass.setFilterLimits(-0.2, max_z);
	pass.filter(*scene_filtered);

	//ROS_WARN("size after z clipping = %u", scene_filtered->points.size());
	
  	/*pass.setInputCloud(boost::make_shared < pcl::PointCloud<pcl::PointXYZRGB> > (*scene_filtered));
	pass.setFilterFieldName("x");
	pass.setFilterLimits(min_x, max_x);	
	pass.filter(*scene_filtered);

	//ROS_WARN("size after x clipping = %u", scene_filtered->points.size());

  	pass.setInputCloud(boost::make_shared < pcl::PointCloud<pcl::PointXYZRGB> > (*scene_filtered));
	pass.setFilterFieldName("y");
	pass.setFilterLimits(min_y, max_y);	
	pass.filter(*scene_filtered);
	*/


	//ROS_WARN("size after y clipping = %u", scene_filtered->points.size());

	if (scene_filtered->empty()) {
		ROS_WARN("No points left after filtering by distance.");
		return;
	}

//	sensor_msgs::PointCloud2 fml;
//	pcl::toROSMsg(*scene_filtered, fml);
 // point_cloud_publisher_.publish(fml);

	//////////////////////////// Segment out table plane and extract the objects point cloud ////////////////////////////
	pcl::ModelCoefficients coefficients;
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	// Create the segmentation object
	pcl::SACSegmentation<pcl::PointXYZRGB> seg;
	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setDistanceThreshold(plane_thresh);
	seg.setInputCloud(scene_filtered);
	seg.segment(*inliers, coefficients);

//  cout << "PLANE: " << coefficients.values[0] << " " << coefficients.values[1] << " " << coefficients.values[2] << " " << coefficients.values[3] << endl;

	pcl::ExtractIndices<pcl::PointXYZRGB> extract;
	// Extract table
	extract.setInputCloud(scene_filtered);
	extract.setIndices(inliers);
	extract.setNegative(false);
	table_pointcloud->clear();
	extract.filter(*table_pointcloud);

	// Extract objects
	extract.setInputCloud(scene_filtered);
	extract.setIndices(inliers);
	extract.setNegative(true);
	objects_pointcloud->clear();
	extract.filter(*objects_pointcloud);

	// Filter z again to make sure we don't get the table edges as points
	pass.setInputCloud(boost::make_shared < pcl::PointCloud<pcl::PointXYZRGB> > (*objects_pointcloud));
	pass.setFilterFieldName("z");
	pass.setFilterLimits(table_height, max_z);	
	//pass.setFilterLimits(-0.2, max_z);
	pass.filter(*objects_pointcloud);

//	cout << "Size after = " << objects_pointcloud->points.size() << endl;
	
	std::cout << "original cloud has size " << scene_filtered->points.size() << std::endl;
	std::cout << "passing on objs cloud of size " << objects_pointcloud->points.size() << std::endl;
	
	/*objects_pointcloud->height = 1;
	objects_pointcloud->width = objects_pointcloud->points.size();*/
}

void Wrapper::clusterPointcloud(PCLPointCloud::Ptr& cloud, std::vector<pcl::PointIndices>& cluster_indices){
	// KdTree object for the search method of the extraction
	pcl::search::KdTree<PCLPoint>::Ptr kdtree_cl(new pcl::search::KdTree<PCLPoint>());
	kdtree_cl->setInputCloud(cloud);
	pcl::EuclideanClusterExtraction<PCLPoint> ec;
	ec.setClusterTolerance(clustering_tolerance_);
	ec.setMinClusterSize(cluster_min_size_);
	ec.setMaxClusterSize(cluster_max_size_);
	ec.setSearchMethod(kdtree_cl);
	ec.setInputCloud(cloud);
	ec.extract(cluster_indices);
}

// Computes the object centroid based on the pointcloud
void Wrapper::computeCentroid(const PCLPointCloud& cloud, Eigen::Vector3f& centroid){
	Eigen::Vector4f centr;
	pcl::compute3DCentroid(cloud, centr);
	centroid[0] = centr[0];
	centroid[1] = centr[1];
	centroid[2] = centr[2];
}
//crops the original pointcloud around the cluster
sensor_msgs::Image Wrapper::cropCloud(sensor_msgs::PointCloud2::ConstPtr original_cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered, std::vector<int> cluster_indices, cv::Point circle_centr){

	static int counter = 0;

	//convert the original cloud to pcl
	pcl::PCLPointCloud2 tmp_cloud;
	pcl_conversions::toPCL(*original_cloud, tmp_cloud);
	pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud;
	pcl::fromPCLPointCloud2(tmp_cloud, pcl_cloud);
	
	//initialize bounds
	double min_x = std::numeric_limits<double>::infinity();
	pcl::PointXYZRGB min_x_pt;
	double min_y = std::numeric_limits<double>::infinity();
	pcl::PointXYZRGB min_y_pt;
	double max_x = -1 * std::numeric_limits<double>::infinity();
	pcl::PointXYZRGB max_x_pt;
	double max_y = -1 * std::numeric_limits<double>::infinity();
	pcl::PointXYZRGB max_y_pt;
	//get the min max cluster indices
	for(std::vector<int>::const_iterator c_it = cluster_indices.begin(); c_it != cluster_indices.end(); ++c_it){
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
	
	
	
	//loop over the original pointcloud
	double cropping_threshold = 0.001;
	min_x = 1 * std::numeric_limits<double>::infinity();
	min_y = 1 * std::numeric_limits<double>::infinity();
	max_x = -1 * std::numeric_limits<double>::infinity();
	max_y = -1 * std::numeric_limits<double>::infinity();
	int min_x_r, min_x_c, max_x_r, max_x_c;
	int min_y_r, min_y_c, max_y_r, max_y_c; 
	for(int row = 0; row < pcl_cloud.height; row++){
		for(int col = 0; col < pcl_cloud.width; col++){
			
			pcl::PointXYZRGB curr_pt = pcl_cloud.at(col, row);
			
			//check for min_x
			if(curr_pt.x >= (min_x_pt.x - cropping_threshold) && curr_pt.x <= (min_x_pt.x)){
				if(curr_pt.x <= min_x){
					min_x = curr_pt.x;
					min_x_r = row;
					min_x_c = col;
				}
			}
			//check for min_y
			if(curr_pt.y >= (min_y_pt.y - cropping_threshold) && curr_pt.y <= (min_y_pt.y)){
				if(curr_pt.y <= min_y){
					min_y = curr_pt.y;
					min_y_r = row;
					min_y_c = col;
				}
			}
			//check for max_x
			if(curr_pt.x >= (max_x_pt.x) && curr_pt.x <= (max_x_pt.x + cropping_threshold)){
				if(curr_pt.x >= max_x){
					max_x = curr_pt.x;
					max_x_r = row;
					max_x_c = col;
				}
			}
			//check for max_y
			if(curr_pt.y >= (max_y_pt.y) && curr_pt.y <= (max_y_pt.y + cropping_threshold)){
				if(curr_pt.y >= max_y){
					max_y = curr_pt.y;
					max_y_r = row;
					max_y_c = col;
				}
			}
		}
	}
	
	/*std::cout << "min_x: " << min_x_r << "," << min_x_c << std::endl;
	std::cout << "min_y: " << min_y_r << "," << min_y_c << std::endl;
	std::cout << "max_x: " << max_x_r << "," << max_x_c << std::endl;
	std::cout << "max_y: " << max_y_r << "," << max_y_c << std::endl;*/
	
	int start_r = min_y_r;
	int start_c = min_x_c;
	int end_r = max_y_r;
	int end_c = max_x_c;
	//std::cout << "start pose: " << start_r << "," << start_c << std::endl;
	//std::cout << "end pose: " << end_r << "," << end_c << std::endl;
	int width = end_r - start_r + 1;
	int height = end_c - start_c + 1;
	
	sensor_msgs::Image original_image;
	pcl::toROSMsg(*original_cloud, original_image);
	cv_bridge::CvImagePtr cv_ptr;
	try{
		cv_ptr = cv_bridge::toCvCopy(original_image, sensor_msgs::image_encodings::BGR8);
	}
	catch(cv_bridge::Exception& e){
		ROS_ERROR("cv_bridge exception: %s", e.what());
	}
	
	if(width > 0 && height > 0){
		//draw a rectangle around the image for the found cluster
		cv::Mat edited_image(cv_ptr->image);
		//cv::circle(edited_image, circle_centr, 5, cv::Scalar(0, 255, 0), 2);
		cv::rectangle(edited_image, cv::Point(start_c, start_r), cv::Point(end_c, end_r), cv::Scalar(255, 0, 0), 2);
		//cv::rectangle(edited_image, cv::Point(min_x_c, min_x_r), cv::Point(max_x_c, max_x_r), cv::Scalar(0, 0, 255), 2);
		//cv::rectangle(edited_image, cv::Point(min_y_c, min_y_r), cv::Point(max_y_c, max_y_r), cv::Scalar(0, 255, 0), 2);
		std::stringstream ss;
		ss << "/home/noha/Documents/Hiwi/foundClusters/im_";
		ss << counter;
		ss << ".jpeg";
		cv::imwrite(ss.str(), edited_image);
	}
	
	//cv::Mat cv_im(cv_ptr->image, cv::Rect(start_r, start_c, width, height));
	//cv::namedWindow("Display Window", cv::WINDOW_AUTOSIZE);
	//cv::imshow("Display Window", cv_im);
	//cv::waitKey(0);
	//sensor_msgs::ImagePtr result_image = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_im).toImageMsg();
	//create new point cloud for the resulting image
	/*pcl::PointCloud<pcl::PointXYZRGB> cropped_cloud;
	cropped_cloud.width = width;
	cropped_cloud.height = height;
	for(int ic = start_c; ic <= end_c; ic++){
		for(int ir = start_r; ir <= end_r; ir++){
			//std::cout << pcl_cloud.points[ir + ic * pcl_cloud.height] << std::endl;
			cropped_cloud.points.push_back(pcl_cloud.points[ir + ic * pcl_cloud.height]);
		}
	}
	//convert the pointcloud to sensor_msgs Image
	sensor_msgs::Image result_image;
	sensor_msgs::PointCloud2 tmp_res;
	pcl::toROSMsg(cropped_cloud, tmp_res);
	pcl::toROSMsg(tmp_res, result_image);
	
	//debugging: visualization
	cv_bridge::CvImagePtr cv_ptr;
	try{
		cv_ptr = cv_bridge::toCvCopy(result_image, sensor_msgs::image_encodings::BGR8);
		cv::namedWindow("Original Window", cv::WINDOW_AUTOSIZE);
		cv::imshow("Original Window", cv_ptr->image);
		cv::waitKey(0);
	}
	catch(cv_bridge::Exception& e){
		ROS_ERROR("cv_bridge exception: %s", e.what());
	}*/
	
	counter++;
	return original_image;
	//return *result_image;
}

sensor_msgs::Image Wrapper::cropCloud2(pcl::PointCloud<pcl::PointXYZRGB> cluster_cloud, sensor_msgs::PointCloud2::ConstPtr original_msg){
	static int counter = 0;
	//compute the bounding box of the cluster 
	pcl::PointXYZRGB min_pt, max_pt;
	pcl::getMinMax3D(cluster_cloud, min_pt, max_pt);
	
	//convert the original msg to pcl
	pcl::PCLPointCloud2 tmp_cloud;
	pcl_conversions::toPCL(*original_msg, tmp_cloud);
	PCLPointCloud original_cloud;
	pcl::fromPCLPointCloud2(tmp_cloud, original_cloud);
	
	//use the P matrix: P: [[525.0, 0.0, 319.5, 0.0], [0.0, 525.0, 239.5, 0.0], [0.0, 0.0, 1.0, 0.0]]
	// and the projection formula to compute the pixel coordinates of the min max points
	// then take the row and col before them
	
	double fx = 525.0;
	double fy = 525.0;
	double cx = 319.5;
	double cy = 239.5;
	
	sensor_msgs::Image original_image;
	pcl::toROSMsg(*original_msg, original_image);
	
	if(min_pt.z == 0.0 || pcl_isnan(min_pt.z) || max_pt.z == 0.0 || pcl_isnan(max_pt.z)){
		ROS_ERROR("min/max point has bad coordinates");
		return original_image;
	}
	int min_pt_r = (min_pt.x / min_pt.z) * fx + cx;
	int min_pt_c = (min_pt.y / min_pt.z) * fy + cy;
	int max_pt_r = (max_pt.x / max_pt.z) * fx + cx;
	int max_pt_c = (max_pt.y / max_pt.z) * fx + cx;
	
	/*cv::Point start_pt (min_pt_c, max_pt_r);
	cv::Point end_pt (max_pt_c, min_pt_r);
	std::cout << "start pt coordinates: " << min_pt_c << "," << max_pt_r << std::endl;
	std::cout << "end pt coordinates: " << max_pt_c << "," << min_pt_r << std::endl;*/
	
	
	cv_bridge::CvImagePtr cv_ptr;
	try{
		cv_ptr = cv_bridge::toCvCopy(original_image, sensor_msgs::image_encodings::BGR8);
	}
	catch(cv_bridge::Exception& e){
		ROS_ERROR("cv_bridge exception: %s", e.what());
	}
	cv::Mat edited_image (cv_ptr->image);
	cv::Point start_pt (std::max(std::min(min_pt_r, max_pt_r) - 1, 0), std::max(std::min(min_pt_c, max_pt_c) - 1, 0));
	cv::Point end_pt (std::min(std::max(max_pt_r, min_pt_r) + 1, (int)original_cloud.height), std::min(std::max(max_pt_c, min_pt_c) + 1, (int)original_cloud.width));
	cv::rectangle(edited_image, start_pt, end_pt, cv::Scalar(255, 0, 0), 2);
	
	if(start_pt.x > end_pt.x || start_pt.y > end_pt.x){
		ROS_ERROR("zero width/height error");
		return original_image;
	}
	
	//crop image and convert to sensor_msgs::Image
	cv::Mat cv_im(cv_ptr->image, cv::Rect(start_pt, end_pt));
	//write image to file
	std::stringstream ss;
	ss << "/home/noha/Documents/Hiwi/foundClusters/im_";
	ss << counter;
	ss << ".jpeg";
	cv::imwrite(ss.str(), cv_im);
	counter++;
	sensor_msgs::ImagePtr result_image = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_im).toImageMsg();

	return *result_image;
}


int main(int argc, char** argv){
	ros::init(argc, argv, "overFeatWrapper");
	ros::NodeHandle nh;
	Wrapper overfeat_wrapper (nh);
	while (ros::ok()) {
		ros::Duration(0.07).sleep();
		ros::spinOnce();
	}
	
	ros::spin();

	return 0;
}
