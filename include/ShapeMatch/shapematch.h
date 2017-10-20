#ifndef _SHAPEMATCH_H_
#define _SHAPEMATCH_H_

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>

#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#include <math.h>

#include "lap.h"

using namespace std;
using namespace cv;

#define OUTLIER_THRESHOLD 1.2

const Scalar RED = Scalar(0,0,255);
const Scalar PINK = Scalar(230,130,255);
const Scalar BLUE = Scalar(255,0,0);
const Scalar LIGHTBLUE = Scalar(255,255,160);
const Scalar GREEN = Scalar(0,255,0);
const Scalar WHITE = Scalar(255, 255, 255);


//namespace  shapematch {

double dist(Point2f& p1, Point2f& p2);

double angle(Point2f& p1, Point2f& p2);

vector< vector<int> > getHistogramFromContourPts(vector<Point2f>& contourPts);

void cleanup(int **hist1, int size1, int **hist2, int size2);

void getChiStatistic(vector< vector<double> >& stats,
                     vector< vector<int> >& histogram1, int size1,
                     vector< vector<int> >& histogram2, int size2);

pair<Point2f,Point2f> getMinMax(vector<Point2f>& cpts1, vector<Point2f>& cpts2);
vector <Point2f> getSampledPoints(vector<Point2f>& v, int sr);
void getMatchShape(Mat& img1_,  vector<Point2f>& contourPts1, Mat& img2_,  vector<Point2f>& contourPts2);


void addTwoImageToOne(Mat &img1_col,Mat &img2_col,Mat &match_img);
void displayMergeImage(Mat & img1_col, vector<  cv::Point2f > &mat_point1, Mat & img2_col, vector<  cv::Point2f > &mat_point2 );


//}

#endif
