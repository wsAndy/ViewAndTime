
/****
 *
 *
 *
 *   writed by sheng wang
 *   2017-9-15
 *
 */

#include "AF/anisometric.hpp"
#include "SLIC/slic_tool.h"
#include "SLIC/slic.h"
#include "ShapeMatch/shapematch.h"

using namespace std;
using namespace  cv;




int main()
{
//    std::string path = "/Users/sheng/Desktop/MSR3DVideo-Breakdancers/";
//    Mat img1_col = imread( path + "cam0/color-cam0-f000.jpg");
//    Mat img2_col = imread( path + "cam2/color-cam2-f000.jpg");
    // Mat img1_col = imread("/Users/sheng/Desktop/museum/2.png");
    // Mat img2_col = imread("/Users/sheng/Desktop/museum/3.png");

    // resize(img1_col,img1_col,cv::Size(int(img1_col.cols/2),int(img1_col.rows/2)));
    // resize(img2_col,img2_col,cv::Size(int(img2_col.cols/2),int(img2_col.rows/2)));

   Mat img1_col = imread("/Users/sheng/Downloads/templeRing/templeR0001.png");
   Mat img2_col = imread("/Users/sheng/Downloads/templeRing/templeR0002.png");

    vector<  cv::Point2f > mat_point1, mat_point2;

/***
 *    使用边缘进行匹配
 *    实验发现效果还不如直接用特征点....
 *    一定有什么 trick
 */
//    getMatchShape(img1_col,mat_point1,img2_col,mat_point2);

/***
 *
 *  show matching shape.
 *
 */
//    displayShapeMatchImg(img1_col,mat_point1,img2_col,mat_point2);
//    return 0;

/***
 *
 *    use ORB and SURF features to match pixels
 *
 *
 */
    matchTwoImageSURF(img1_col,img2_col, mat_point1, mat_point2);
    matchTwoImageORB(img1_col,img2_col, mat_point1, mat_point2);

//    cout << "feature1 size = " << mat_point1.size() <<endl;


/****
 *
 *    show matching pixels in two image
 *
 */
//    if(mat_point1.size() == mat_point2.size())
//    {

//        displayMergeImage(img1_col, mat_point1,img2_col, mat_point2);

//    }else{
//        cout << "point1 size != point2 size" << endl;
//    }

    vec2di cluster1 = getSuperPixels(img1_col);
    vec2di cluster2 = getSuperPixels(img2_col);

    cout << "mat point1 = " << mat_point1.size() <<endl;
    cout << "mat point2 = " << mat_point2.size() <<endl;


    Mat derformMat1 = getCorresponseMaps(cluster1, mat_point1, mat_point2);

    Mat derformMat2 = getCorresponseMaps(cluster2, mat_point2, mat_point1);

//    vector<Mat> vec_sp;
//    split(derformMat,vec_sp);

//    Mat out = vec_sp[0].clone();
//    out.convertTo(out, CV_32FC1);

//    PM_Diffusion pm(out);

//    out = pm.diffusion();
//    double min;
//    double max;
//    minMaxIdx(out, &min, &max);

//    out.convertTo(out, CV_8UC1, 255 / (max - min), -min);


//    imshow("def",derformMat1);
//    waitKey(0);


//    Mat Hab = findHomography(mat_point1,mat_point2,RANSAC);
//    Mat Hba = findHomography(mat_point2,mat_point1,RANSAC);

//    Mat dst1,dst2;
////    warpAffine(img1_col,dst1,Hab,img1_col.size());
////    warpAffine(img2_col,dst2,Hba,img2_col.size());

//    vector<cv::Point2f> im1,im2,tar1,tar2;
//    for(int i = 0; i < img1_col.rows; ++i)
//    {
//        for(int j = 0; j < img1_col.cols; ++j)
//        {
//            im1.push_back(Point2f(j,i));
//            im2.push_back(Point2f(j,i));
//        }
//    }

//    perspectiveTransform(im1,tar1,Hab);
//    perspectiveTransform(im2,tar2,Hba);

//    Mat to1,to2;
//    to1 = cv::Mat::zeros(img1_col.rows,img1_col.cols,CV_8UC3);
//    to2 = cv::Mat::zeros(img1_col.rows,img1_col.cols,CV_8UC3);

//    for(int i = 0; i < tar1.size() ; ++i)
//    {
////            im1.push_back(Point2f(j,i));
////            im2.push_back(Point2f(j,i));
//        if(tar1[i].x < 0 || tar1[i].x >= img1_col.cols || tar1[i].y < 0 || tar1[i].y >= img1_col.rows )
//            continue;
//            to1.at<Vec3b>(tar1[i])[0] = 255;
//            to1.at<Vec3b>(tar1[i])[1] = 255;
//            to1.at<Vec3b>(tar1[i])[2] = 255;

//    }
//    for(int i = 0; i < tar2.size() ; ++i)
//    {
////            im1.push_back(Point2f(j,i));
////            im2.push_back(Point2f(j,i));
//        if(tar2[i].x < 0 || tar2[i].sx >= img1_col.cols || tar2[i].y < 0 || tar2[i].y >= img1_col.rows )
//            continue;
//            to2.at<Vec3b>(tar2[i])[0] = 255;
//            to2.at<Vec3b>(tar2[i])[1] = 255;
//            to2.at<Vec3b>(tar2[i])[2] = 255;
//    }

//    imshow("ab",to1);
//    imshow("ba",to2);
//    waitKey(0);

    warp(img1_col,derformMat1,1);
    warp(img2_col,derformMat2,2);
//    warp(img1_col, derformMat1, img2_col,derformMat2,10);

    // AF, but how AF change the result ???
//    imshow("Diffuesed Image", out);
//    waitKey(0);

    return 0;
}
