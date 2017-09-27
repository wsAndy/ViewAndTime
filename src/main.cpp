#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "string"
#include "set"
#include "SLIC/slic.h"

using namespace std;
using namespace  cv;


void matchTwoImage(Mat img1_col, Mat img2_col);
int findmaxclu(vec2di& vec);

Mat getColorCluster(vec2di & cluster);
Mat getSingleCluster(vec2di & cluster);
void testSLIC(Mat& img1_col);

int main()
{
    std::string path = "/home/arvr/Desktop/3DVideos-distrib/MSR3DVideo-Ballet/";
    Mat img1_col = imread( path + "cam0/color-cam0-f000.jpg");
    Mat img2_col = imread( path + "cam1/color-cam1-f000.jpg");

//    testSLIC(img1_col);

    matchTwoImage(img1_col,img2_col);




    return 0;
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




// after test, it cannot work for our object
void matchTwoImage(Mat img1_col, Mat img2_col)
{
        Mat img1_gray,img2_gray;

        int cols = img1_col.cols;
        int rows = img1_col.rows;


        cvtColor(img1_col,img1_gray,CV_BGR2GRAY);
        cvtColor(img2_col,img2_gray,CV_BGR2GRAY);

        //-- Step 1: Detect the keypoints using SURF Detector
        int minHessian = 40;

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
    // 能否画一个dist的分布图
        cout << "max dist = " << max_dist << endl;
        cout << "min dist = " << min_dist << endl;
        //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
         //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
         //-- small)
         //-- PS.- radiusMatch can also be used here.
         std::vector< DMatch > good_matches;

         for( int i = 0; i < descriptors_1.rows; i++ )
         {
             if( matches[i].distance <= max(2*min_dist, 0.02) )
           { good_matches.push_back( matches[i]); }
         }

    //     //-- Draw only "good" matches
         Mat img_matches;
         drawMatches( img1_col, keypoints_1, img2_col, keypoints_2,
                      good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                      vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

         //-- Show detected matches
//         resize(img_matches,img_matches,Size(int(cols/2),int(rows/4)));
         imshow( "Good Matches", img_matches );

         waitKey(0);

}
