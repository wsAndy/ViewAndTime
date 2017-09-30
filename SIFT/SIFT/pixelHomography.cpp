#include <iostream>  
#include<highgui.h>  
#include<opencv2/nonfree/features2d.hpp>   
#include<opencv2/legacy/legacy.hpp>   
#include<opencv2/features2d/features2d.hpp>   
using namespace cv;
using namespace std;

int main(int argc, char** argv) {

	argv[1] = "color-cam0-f000.jpg";      // 去校正  
	argv[2] = "color-cam1-f000.jpg";              // 被校正  

	//Mat img_object = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	//Mat img_scene = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
	Mat img_object = imread(argv[1],CV_32FC3);
	Mat img_scene = imread(argv[2], CV_32FC3);
	if (!img_object.data || !img_scene.data) {
		std::cout << " --(!) Error reading images " << std::endl; return -1;
	}

	//-- Step 1: Detect the keypoints using SURF Detector    
	int minHessian = 400;
	SurfFeatureDetector detector(minHessian);
	std::vector<KeyPoint> keypoints_object, keypoints_scene;
	detector.detect(img_object, keypoints_object);
	detector.detect(img_scene, keypoints_scene);

	//-- Step 2: Calculate descriptors (feature vectors)    
	SurfDescriptorExtractor extractor;
	Mat descriptors_object, descriptors_scene;
	extractor.compute(img_object, keypoints_object, descriptors_object);
	extractor.compute(img_scene, keypoints_scene, descriptors_scene);

	//-- Step 3: Matching descriptor vectors using FLANN matcher    
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match(descriptors_object, descriptors_scene, matches);
	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints    
	for (int i = 0; i < descriptors_object.rows; i++) {
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )    
	std::vector< DMatch > good_matches;

	for (int i = 0; i < descriptors_object.rows; i++) {
		if (matches[i].distance < 3 * min_dist) {
			good_matches.push_back(matches[i]);
		}
	}

	Mat img_matches;
	drawMatches(img_object, keypoints_object, img_scene, keypoints_scene,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//-- Localize the object from img_1 in img_2    
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (size_t i = 0; i < good_matches.size(); i++) {
		//-- Get the keypoints from the good matches    
		obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
	}

	Mat H = findHomography(obj, scene, RANSAC);

	Mat derforMat(img_object.cols, img_object.rows, CV_64FC3, Scalar(0, 0, 0));

	Mat xSrcTrans = Mat(1, 1, CV_64FC3, Scalar(0, 0, 0));

	for (size_t i = 0; i < img_object.cols; i++)
	{
		for (size_t j = 0; j < img_object.rows; j++)
		{
			double temp[3][1] = { (double)i, (double)j, (double)1 };
			Mat xSrc = Mat(3, 1, CV_64FC1, temp);
			xSrc = H*xSrc - xSrc;
			xSrcTrans.at<Vec3d>(0, 0)[0] = xSrc.at<Vec3d>(0, 0)[0];
			xSrcTrans.at<Vec3d>(0, 0)[1] = xSrc.at<Vec3d>(1, 0)[0];
			xSrcTrans.at<Vec3d>(0, 0)[2] = xSrc.at<Vec3d>(2, 0)[0];

			derforMat.at<Vec3d>(i, j)[0] = xSrcTrans.at<Vec3d>(0, 0)[0];
			derforMat.at<Vec3d>(i, j)[1] = xSrcTrans.at<Vec3d>(0, 0)[1];
			derforMat.at<Vec3d>(i, j)[2] = xSrcTrans.at<Vec3d>(0, 0)[2];
		}
	}

	imshow("derforMat", derforMat);


	//-- Get the corners from the image_1 ( the object to be "detected" )    
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = Point(0, 0); obj_corners[1] = Point(img_object.cols, 0);
	obj_corners[2] = Point(img_object.cols, img_object.rows); obj_corners[3] = Point(0, img_object.rows);
	std::vector<Point2f> scene_corners(4);
	perspectiveTransform(obj_corners, scene_corners, H);
	//-- Draw lines between the corners (the mapped object in the scene - image_2 )    
	Point2f offset((float)img_object.cols, 0);
	line(img_matches, scene_corners[0] + offset, scene_corners[1] + offset, Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[1] + offset, scene_corners[2] + offset, Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[2] + offset, scene_corners[3] + offset, Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[3] + offset, scene_corners[0] + offset, Scalar(0, 255, 0), 4);

	//-- Show detected matches    
	imshow("Good Matches & Object detection", img_matches);
	waitKey(0);
	return 0;
}

