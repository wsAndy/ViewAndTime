
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
    std::string path = "/Users/sheng/Desktop/MSR3DVideo-Breakdancers/";
    Mat img1_col = imread( path + "cam0/color-cam0-f000.jpg");
    Mat img2_col = imread( path + "cam2/color-cam2-f000.jpg");

//    resize(img1_col,img1_col,cv::Size(int(img1_col.cols/2),int(img1_col.rows/2)));
//    resize(img2_col,img2_col,cv::Size(int(img2_col.cols/2),int(img2_col.rows/2)));

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
//    vec2di cluster2 = getSuperPixels(img2_col);

    cout << "mat point1 = " << mat_point1.size() <<endl;
    cout << "mat point2 = " << mat_point2.size() <<endl;

    Mat derformMat1 = getCorresponseMaps(cluster1, mat_point1, mat_point2);
//    Mat derformMat2 = getCorresponseMaps(cluster2, mat_point2, mat_point1);

    warp(img1_col,derformMat1);
//    warp(img1_col, derformMat1, img2_col,derformMat2,10);

    // AF, but how AF change the result ???
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
//    imshow("Diffuesed Image", out);
//    waitKey(0);

    return 0;
}

