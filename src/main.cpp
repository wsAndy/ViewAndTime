#include <iostream>
#include "stdio.h"
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "string"
#include "set"
#include "SLIC/slic.h"


using namespace std;
using namespace  cv;


void matchTwoImageSURF(Mat &img1_col, Mat &img2_col, vector< cv::Point2f> & ,vector< cv::Point2f> & );
void matchTwoImageORB(Mat &img1_col, Mat &img2_col, vector< cv::Point2f> & ,vector< cv::Point2f> & );
int findmaxclu(vec2di& vec);

Mat getColorCluster(vec2di & cluster);
Mat getSingleCluster(vec2di & cluster);
void testSLIC(Mat& img1_col);

void addTwoImageToOne(Mat& ,Mat&, Mat&);
void display(Mat & , vector<  cv::Point2f > & , Mat &, vector<  cv::Point2f > &);


int main()
{
    std::string path = "/home/arvr/Desktop/3DVideos-distrib/MSR3DVideo-Ballet/";
    Mat img1_col = imread( path + "cam0/color-cam0-f000.jpg");
    Mat img2_col = imread( path + "cam2/color-cam2-f000.jpg");

//    testSLIC(img1_col);

    vector<  cv::Point2f > mat_point1, mat_point2;

    matchTwoImageSURF(img1_col,img2_col, mat_point1, mat_point2);
    matchTwoImageORB(img1_col,img2_col, mat_point1, mat_point2);

    if(mat_point1.size() == mat_point2.size())
    {

        display(img1_col, mat_point1,img2_col, mat_point2);

    }else{
        cout << "point1 size != point2 size" << endl;
    }






    return 0;
}

void display(Mat & img1_col, vector<  cv::Point2f > &mat_point1, Mat & img2_col, vector<  cv::Point2f > &mat_point2 )
{
    Mat match_img;

    addTwoImageToOne(img1_col,img2_col,match_img);

    for(int i =0 ;i < mat_point1.size(); ++i)
    {
        circle(match_img,mat_point1[i],3,Scalar(255,0,0),1,8);
        circle(match_img,Point2f(mat_point2[i].x + match_img.cols/2,mat_point2[i].y),3,Scalar(0,0,255),3,8);
        line(match_img,mat_point1[i],Point2f(mat_point2[i].x + match_img.cols/2,mat_point2[i].y),Scalar(0,255,0),1,8);
    }

    imshow("mat",match_img);
    waitKey(0);

}

void addTwoImageToOne(Mat &img1_col,Mat &img2_col,Mat &match_img)
{
     match_img = Mat::zeros(img1_col.rows,img1_col.cols + img2_col.cols,CV_8UC3);
     for(int i = 0; i < match_img.rows; ++i)
     {
         for(int j = 0; j < match_img.cols/2; ++j)
         {
             match_img.at<Vec3b>(i,j)[0] = img1_col.at<Vec3b>(i,j)[0];
             match_img.at<Vec3b>(i,j)[1] = img1_col.at<Vec3b>(i,j)[1];
             match_img.at<Vec3b>(i,j)[2] = img1_col.at<Vec3b>(i,j)[2];
         }

         for(int j = match_img.cols/2 ; j < match_img.cols; ++j)
         {
             match_img.at<Vec3b>(i,j)[0] = img2_col.at<Vec3b>(i,j-match_img.cols/2)[0];
             match_img.at<Vec3b>(i,j)[1] = img2_col.at<Vec3b>(i,j-match_img.cols/2)[1];
             match_img.at<Vec3b>(i,j)[2] = img2_col.at<Vec3b>(i,j-match_img.cols/2)[2];
         }
     }

}

void testSLIC(Mat& img1_col)
{
    Mat *lab_img1,lab1;

    cvtColor(img1_col,lab1,CV_BGR2Lab);

    lab_img1 = &lab1;

    int w = img1_col.cols, h = img1_col.rows;
    int nr_superpixels = 300;

    int nc = 80;

    double step = sqrt((w*h)/(double)(nr_superpixels)  );

    Slic sl;

    sl.generate_superpixels(lab_img1,step,nc);
    sl.create_connectivity(lab_img1);

    vec2di new_cluster = sl.get_new_cluster();
    vec2di cluster = sl.get_cluster();


    Mat sup = getSingleCluster(cluster);
    Mat new_sup = getColorCluster(new_cluster);

    //vec2dd center = sl.get_center();
    //vec2dd distance = sl.get_distance();

    imshow("new_cluster",new_sup);
    imshow("cluster",sup);
    waitKey(0);

}

Mat getSingleCluster(vec2di& cluster)
{

    int max_clu = findmaxclu(cluster);
    int h = cluster[0].size();
    int w = cluster.size();

    Mat sup = Mat::zeros(h,w,CV_8UC1);

    for(int i = 0; i < cluster.size(); ++i)
    {// get one column
        for(int j = 0; j < cluster[i].size(); ++j)
        {
            if(cluster[i][j] == -1)
            {
                  sup.at<uchar>(j,i) = 0;

            }else{

                  sup.at<uchar>(j,i) = int(cluster[i][j]*255/max_clu);

            }

        }

    }

    return sup;

}

Mat getColorCluster(vec2di & cluster)
{

    set<int> last_clus;
    last_clus.insert(-1);

    int max_clu = findmaxclu(cluster);

    vector<Scalar> sup_col;
    for(int i =0; i < max_clu; ++i)
    {
        sup_col.push_back(Scalar(rand()%255+1, rand()%255+1, rand()%255+1 ));
    }

    int h = cluster[0].size();
    int w = cluster.size();

    Mat sup = Mat::zeros(h,w,CV_8UC3);

    for(int i = 0; i < cluster.size(); ++i)
    {// get one column
        for(int j = 0; j < cluster[i].size(); ++j)
        {
            if(cluster[i][j] == -1)
            {
                ;
            }else{
                sup.at<Vec3b>(j,i)[0] = sup_col[cluster[i][j]][0];
                sup.at<Vec3b>(j,i)[1] = sup_col[cluster[i][j]][1];
                sup.at<Vec3b>(j,i)[2] = sup_col[cluster[i][j]][2];

            }
        }

    }

    return sup;
}

int findmaxclu(vector<vector<int> > &vec)
{
    int max = vec[0][0];
    for(int i = 0; i < vec.size(); ++i)
    {
        for(int j = 0; j < vec[i].size(); ++j)
        {
            if(max < vec[i][j])
            {
                max = vec[i][j];
            }
        }
    }
    return max;

}



void matchTwoImageORB(Mat &img1_col, Mat& img2_col, vector< cv::Point2f > & mat_point1, vector< cv::Point2f > & mat_point2)
{
    Mat img1_gray,img2_gray;
    cvtColor(img1_col,img1_gray,CV_BGR2GRAY);
    cvtColor(img2_col,img2_gray,CV_BGR2GRAY);

    ORB orb;
    vector<KeyPoint> keyPoints_1, keyPoints_2;
    Mat descriptors_1, descriptors_2;

    orb(img1_gray, Mat(), keyPoints_1, descriptors_1);
    orb(img2_gray, Mat(), keyPoints_2, descriptors_2);

    BruteForceMatcher<HammingLUT> matcher;
    vector<DMatch> matches;
    matcher.match(descriptors_1, descriptors_2, matches);

    double max_dist = 0; double min_dist = 100;
    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }
    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );
    //-- Draw only "good" matches (i.e. whose distance is less than 0.6*max_dist )
    //-- PS.- radiusMatch can also be used here.
    std::vector< DMatch > good_matches;
    for( int i = 0; i < descriptors_1.rows; i++ )
    {
        if( matches[i].distance < 0.8*max_dist ) // max_dist*0.6
        {
            good_matches.push_back( matches[i]);
        }
    }

    Mat img_matches;
    drawMatches(img1_col, keyPoints_1, img2_col, keyPoints_2,
        good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
        vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//    imshow( "Match before ransac", img_matches);



    // start ransac
    vector<KeyPoint> R_keypoint01,R_keypoint02;
    for (size_t i=0;i<good_matches.size();i++)
    {
        R_keypoint01.push_back(keyPoints_1[good_matches[i].queryIdx]);
        R_keypoint02.push_back(keyPoints_2[good_matches[i].trainIdx]);
    }

    vector<Point2f>p01,p02;
    for (size_t i=0;i<good_matches.size();i++)
    {
        p01.push_back(R_keypoint01[i].pt);
        p02.push_back(R_keypoint02[i].pt);
    }

    vector<uchar> RansacStatus;
    Mat Fundamental= findFundamentalMat(p01,p02,RansacStatus,FM_RANSAC);

    vector<KeyPoint> RR_keypoint01,RR_keypoint02;
    vector<DMatch> RR_matches;            //重新定义RR_keypoint 和RR_matches来存储新的关键点和匹配矩阵
    int index=0;

    for (size_t i=0;i<good_matches.size();i++)
    {
        if (RansacStatus[i]!=0)
        {
            RR_keypoint01.push_back(R_keypoint01[i]);
            RR_keypoint02.push_back(R_keypoint02[i]);
            good_matches[i].queryIdx=index;
            good_matches[i].trainIdx=index;
            RR_matches.push_back(good_matches[i]);
            index++;


            mat_point1.push_back(R_keypoint01[i].pt);
            mat_point2.push_back(R_keypoint02[i].pt);
        }
    }
//    Mat img_RR_matches;
//    drawMatches(img1_col,RR_keypoint01,img2_col,RR_keypoint02,RR_matches,img_RR_matches);
//    imshow("after ransac2",img_RR_matches);

//    waitKey(0);
}

// after test, it cannot work for our object
void  matchTwoImageSURF(Mat& img1_col, Mat& img2_col, vector< cv::Point2f > & mat_point1, vector< cv::Point2f > & mat_point2)
{
        Mat img1_gray,img2_gray;

        int cols = img1_col.cols;
        int rows = img1_col.rows;


        cvtColor(img1_col,img1_gray,CV_BGR2GRAY);
        cvtColor(img2_col,img2_gray,CV_BGR2GRAY);

        //-- Step 1: Detect the keypoints using SURF Detector
        int minHessian = 3;

        SurfFeatureDetector detector( minHessian );

        std::vector<KeyPoint> keypoints_1, keypoints_2;

        detector.detect( img1_gray, keypoints_1 );
        detector.detect( img2_gray, keypoints_2 );

        //-- Step 2: Calculate descriptors (feature vectors)
        SurfDescriptorExtractor extractor;

        Mat descriptors_1, descriptors_2;

        extractor.compute( img1_gray, keypoints_1, descriptors_1 );
        extractor.compute( img2_gray, keypoints_2, descriptors_2 );


        //-- Step 3: Matching descriptor vectors using FLANN matcher
        FlannBasedMatcher matcher;
        std::vector< DMatch > matches;
        matcher.match( descriptors_1, descriptors_2, matches );

        double max_dist = 0; double min_dist = 100;

        //-- Quick calculation of max and min distances between keypoints
        for( int i = 0; i < descriptors_1.rows; i++ )
        { double dist = matches[i].distance;
          if( dist < min_dist ) min_dist = dist;
          if( dist > max_dist ) max_dist = dist;
        }

        cout << "max dist = " << max_dist << endl;
        cout << "min dist = " << min_dist << endl;
         std::vector< DMatch > good_matches;

         for( int i = 0; i < descriptors_1.rows; i++ )
         {
             if( matches[i].distance <= max(15*min_dist, 0.002) )
           { good_matches.push_back( matches[i]); }
         }

    //     //-- Draw only "good" matches
         Mat img_matches;
         drawMatches( img1_col, keypoints_1, img2_col, keypoints_2,
                      good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                      vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

//         imshow( "Good Matches before ransac", img_matches );


         // start ransac
         vector<KeyPoint> R_keypoint01,R_keypoint02;
         for (size_t i=0;i<good_matches.size();i++)
         {
             R_keypoint01.push_back(keypoints_1[good_matches[i].queryIdx]);
             R_keypoint02.push_back(keypoints_2[good_matches[i].trainIdx]);
         }

         vector<Point2f>p01,p02;
         for (size_t i=0;i<good_matches.size();i++)
         {
             p01.push_back(R_keypoint01[i].pt);
             p02.push_back(R_keypoint02[i].pt);
         }

         vector<uchar> RansacStatus;
         Mat Fundamental= findFundamentalMat(p01,p02,RansacStatus,FM_RANSAC);

         vector<KeyPoint> RR_keypoint01,RR_keypoint02;
         vector<DMatch> RR_matches;            //重新定义RR_keypoint 和RR_matches来存储新的关键点和匹配矩阵
         int index=0;

         for (size_t i=0;i<good_matches.size();i++)
         {
             if (RansacStatus[i]!=0)
             {
                 RR_keypoint01.push_back(R_keypoint01[i]);
                 RR_keypoint02.push_back(R_keypoint02[i]);
                 good_matches[i].queryIdx=index;
                 good_matches[i].trainIdx=index;
                 RR_matches.push_back(good_matches[i]);


                 index++;


                 mat_point1.push_back(R_keypoint01[i].pt);
                 mat_point2.push_back(R_keypoint02[i].pt);

             }
         }

//         Mat img_RR_matches;
//         drawMatches(img1_col,RR_keypoint01,img2_col,RR_keypoint02,RR_matches,img_RR_matches);
//         imshow("after ransac",img_RR_matches);

//         waitKey(0);

}
