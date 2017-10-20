#ifndef _SLIC_TOOL_H_
#define _SLIC_TOOL_H_


#include <iostream>
#include "vector"
#include "stdio.h"
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "string"
#include "set"
#include "map"
#include <algorithm>
// #include "SLIC/slic.h"

using namespace std;
using namespace cv;


#define vec2dd vector< vector<double> >
#define vec2di vector< vector<int> >
#define vec2db vector< vector<bool> >

#define FEATURE_NUMBER_REGION 5
#define HOMO_MIN_NUM  5
#define ITERATOR_TIMES_FOR_HOMO 15
#define HOMO_DISTANCE 3
#define CLUSTER_MIN_NUM 3
#define OUTLIER_MAX_NUM 1000

//namespace slic_tool{

// for matching
void matchTwoImageSURF(Mat &img1_col, Mat &img2_col, vector< cv::Point2f> & ,vector< cv::Point2f> & );
void matchTwoImageORB(Mat &img1_col, Mat &img2_col, vector< cv::Point2f> & ,vector< cv::Point2f> & );
int findmaxclu(vec2di& vec);
void addTwoImageToOne(Mat& ,Mat&, Mat&);
void displayMergeImage(Mat & , vector<  cv::Point2f > & , Mat &, vector<  cv::Point2f > &);
void displayShapeMatchImg(Mat& , vector<Point2f>&, Mat&, vector<Point2f>& );



// for superpixels
// show superpixels cluster
Mat getColorCluster(vec2di & cluster);
void displayColorCluster(vec2di & cluster);
Mat getSingleCluster(vec2di & cluster);
vec2di getSuperPixels(Mat& img1_col);
// for  cv::Point3f, I define [x,y,counter]
void displayCenterWithCluster(vector< cv::Point3f > &center, vec2di & cluster);
int getClusterSize(vec2di& cluster);
vector<cv::Point3f> getCenterFromCluster(vec2di & cluster);

// calculate distance between two points in image coordinate
double distance(cv::Point3f& , cv::Point3f& );

vec2dd createDistMat(vector<cv::Point3f> & center);

vec2dd createDistMat(vector<cv::Point3f> & center, vector< set<int> >& neib);


void updateDistMat(vec2dd& , vector<int>& );

// find nearest surperpixels' ID
int findNearestId(vec2dd& ,int& );

// draw surperpixels with special ID
void drawSpecID(vec2di& , int&, int&);

// find surperpixels' neighbors
vector< set<int> > findNeighborSurperpixels(vec2di&);

// show neighbors
void showNeighbor(vector< set<int> >&);

vector<int> findNofeatureId(vector<  cv::Point2f >& ,vec2di& cluster);

// have not complete
void updateNofeatureRegion(vec2di&, vector< set<int>>& );

// get corresponsed maps
Mat getCorresponseMaps( vec2di& ,vector<  cv::Point2f >& ,vector<  cv::Point2f >& );

// warp image and save the result
void warp( Mat& img1_col, Mat& derformMat);
void warp( Mat& img1_col, Mat& derformMat, int index);
void warp( Mat& img1_col, Mat& derformMat1,  Mat& img2_col, Mat& derformMat2, int Vir_num);

// judge homography by calculate distance between x2 and H*x1
bool judgeHomoDistance( Mat& H,vector<cv::Point2f>& obj,vector<cv::Point2f>& scene );

void iteratorGetHomo(vec2di& , std::vector<cv::Mat>&, std::map<int,int>&, std::vector<cv::Point2f>&, std::vector<cv::Point2f>&, int& );



/// new function
void getSuperpixelHomo(vec2di& , std::vector<cv::Mat>&, std::map<int,int>&, std::vector<cv::Point2f>&, std::vector<cv::Point2f>&, int& );

set<int> getClusterID(vec2di& cluster);

vector<int> findNotenoughFeatureID(vec2di&, vector<Point2f>&);

void findNearestIdAndUpdateCluster(vec2di& cluster, vec2dd& dist_table, vector<cv::Point3f>& center,vector<int>& id_not_enough);

void calHomo(vec2di& cluster, vector<Point2f>& mat_point1, vector<Point2f>& mat_point2, std::vector<cv::Mat>& Homo, std::map<int,int>& homo_link, vec2dd& dist_table, vector<Point3f>& center, int count);

bool iteHomo(vector<Point2f>& obj, vector<Point2f>& scene, std::vector<cv::Mat>& Homo, std::map<int,int>& homo_link, int i);

// neighbor
vector<int> getFeatureNumberInCluster(vector<  cv::Point2f >& mat_point1,vec2di& cluster);

void updateCluster(vector< vector<int> >& cluster, map<int,int>& link);

void iterClusterWithNeighbor(vector< vector<int> >& cluster,vector<  cv::Point2f >& mat_point, vector<int>& id_NotEnough);

void calHomo(vec2di& cluster, vector<Point2f>& mat_point1, vector<Point2f>& mat_point2, std::vector<cv::Mat>& Homo, std::map<int,int>& homo_link ,int count);

#endif
