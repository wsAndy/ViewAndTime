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
#include "map"
#include "SLIC/slic.h"
#include "AF/anisometric.hpp"



using namespace std;
using namespace  cv;

// for matching
void matchTwoImageSURF(Mat &img1_col, Mat &img2_col, vector< cv::Point2f> & ,vector< cv::Point2f> & );
void matchTwoImageORB(Mat &img1_col, Mat &img2_col, vector< cv::Point2f> & ,vector< cv::Point2f> & );
int findmaxclu(vec2di& vec);
void addTwoImageToOne(Mat& ,Mat&, Mat&);
void displayMergeImage(Mat & , vector<  cv::Point2f > & , Mat &, vector<  cv::Point2f > &);


// for superpixels
// show superpixels cluster
Mat getColorCluster(vec2di & cluster);
Mat getSingleCluster(vec2di & cluster);
vec2di getSuperPixels(Mat& img1_col);
// for  cv::Point3f, I define [x,y,counter]
void displayCenterWithCluster(vector< cv::Point3f > &center, vec2di & cluster);

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
Mat getCorresponseMaps(Mat& ,vector<  cv::Point2f >& ,vector<  cv::Point2f >& );

// warp image and save the result
void warp(Mat& img1_col, Mat& derformMat);
void warp( Mat& img1_col, Mat& derformMat1,  Mat& img2_col, Mat& derformMat2, int Vir_num);


int main()
{
    std::string path = "/Users/sheng/Desktop/MSR3DVideo-Breakdancers/";
    Mat img1_col = imread( path + "cam0/color-cam0-f000.jpg");
    Mat img2_col = imread( path + "cam2/color-cam2-f000.jpg");


    vector<  cv::Point2f > mat_point1, mat_point2;

    // use ORB and SURF features to match pixels
    matchTwoImageSURF(img1_col,img2_col, mat_point1, mat_point2);
    matchTwoImageORB(img1_col,img2_col, mat_point1, mat_point2);

//    cout << "feature1 size = " << mat_point1.size() <<endl;

    /*
     *  show matching pixels in two image
     * */
//    if(mat_point1.size() == mat_point2.size())
//    {

//        displayMergeImage(img1_col, mat_point1,img2_col, mat_point2);

//    }else{
//        cout << "point1 size != point2 size" << endl;
//    }

    Mat derformMat1 = getCorresponseMaps(img1_col,mat_point1,mat_point2);
    Mat derformMat2 = getCorresponseMaps(img2_col,mat_point2,mat_point1);

/// a test
/// warp(img1_col,derformMat1);

    warp(img1_col, derformMat1, img2_col,derformMat2,10);

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

void warp( Mat& img1_col, Mat& derformMat1,  Mat& img2_col, Mat& derformMat2, int Vir_num)
{
    for(int ind = 1; ind < Vir_num+1; ++ind)
    {
        Mat vir_image1 = Mat::zeros(img1_col.rows, img1_col.cols, CV_8UC3);
        Mat vir_image2 = Mat::zeros(img2_col.rows, img2_col.cols, CV_8UC3);

        for(int i =0; i < img1_col.rows; ++i)
        {
            for(int j = 0; j < img1_col.cols; ++j)
            {
                int x =  i + int(derformMat1.at<Vec3d>(i,j)[0] * ind / (Vir_num+1) ); // row
                int y =  j + int(derformMat1.at<Vec3d>(i,j)[1] * ind / (Vir_num+1) ); // col

                if( x < 0 || x >= vir_image1.rows || y < 0 || y >= vir_image1.cols)
                {
                    continue;
                }
                vir_image1.at<Vec3b>(x,y)[0] = img1_col.at<Vec3b>(i,j)[0] ;
                vir_image1.at<Vec3b>(x,y)[1] = img1_col.at<Vec3b>(i,j)[1] ;
                vir_image1.at<Vec3b>(x,y)[2] = img1_col.at<Vec3b>(i,j)[2] ;
            }
        }

        for(int i =0; i < img2_col.rows; ++i)
        {
            for(int j = 0; j < img2_col.cols; ++j)
            {
                int x =  i + int(derformMat2.at<Vec3d>(i,j)[0] * (1 - double(ind) / (Vir_num+1)) ); // row
                int y =  j + int(derformMat2.at<Vec3d>(i,j)[1] * (1 - double(ind) / (Vir_num+1)) ); // col

                if( x < 0 || x >= vir_image2.rows || y < 0 || y >= vir_image2.cols)
                {
                    continue;
                }
                vir_image2.at<Vec3b>(x,y)[0] = img2_col.at<Vec3b>(i,j)[0] ;
                vir_image2.at<Vec3b>(x,y)[1] = img2_col.at<Vec3b>(i,j)[1] ;
                vir_image2.at<Vec3b>(x,y)[2] = img2_col.at<Vec3b>(i,j)[2] ;
            }
        }

        Mat blend_image;
        addWeighted(vir_image1, double(ind)/(Vir_num+1) ,vir_image2, 1 - double(ind)/(Vir_num+1) ,0 ,blend_image);

        stringstream ss;
        string str_name;
        ss << "vir";
        ss << ind;
        ss << ".jpg";
        ss >> str_name;
        ss.clear();
//        imshow( str_name.c_str() ,vir_image);
        imwrite("/Users/sheng/Desktop/save/"+str_name,blend_image);


    }
}

void warp( Mat& img1_col, Mat& derformMat)
{

    for(int ind = 0; ind <11; ++ind)
    {
        // show interpolation virtual image
        Mat vir_image = Mat::zeros(img1_col.rows,img1_col.cols,CV_8UC3);
        for(int i =0; i < img1_col.rows; ++i)
        {
            for(int j = 0; j < img1_col.cols; ++j)
            {
                int x =  i + int(derformMat.at<Vec3d>(i,j)[0] * ind *0.1); // row
                int y =  j + int(derformMat.at<Vec3d>(i,j)[1] * ind *0.1); // col

                if( x < 0 || x >= vir_image.rows || y < 0 || y >= vir_image.cols)
                {
                    continue;
                }

                vir_image.at<Vec3b>(x,y)[0] = img1_col.at<Vec3b>(i,j)[0] ;

                vir_image.at<Vec3b>(x,y)[1] = img1_col.at<Vec3b>(i,j)[1] ;

                vir_image.at<Vec3b>(x,y)[2] = img1_col.at<Vec3b>(i,j)[2] ;

            }
        }

        stringstream ss;
        string str_name;
        ss << "vir";
        ss << ind;
        ss << ".jpg";
        ss >> str_name;
        ss.clear();
//        imshow( str_name.c_str() ,vir_image);
        imwrite("/Users/sheng/Desktop/save/"+str_name,vir_image);
//        waitKey(0);

    }

}

Mat getCorresponseMaps(Mat& img1_col,vector<  cv::Point2f >& mat_point1,vector<  cv::Point2f >& mat_point2)
{
    vec2di cluster = getSuperPixels(img1_col); // not follow sequences

    Mat new_sup = getColorCluster(cluster);
//    imshow("new_cluster",new_sup);

//    waitKey(0);
//    return 0;


    vector<int> id_without = findNofeatureId(mat_point1,cluster);

    cerr << "************************************************" <<endl
         << "HERE, I need to use neighbor information, and " << endl
         << "complete the function updateNofeatureRegion" << endl
         << "************************************************" <<endl;
/***
 *
 *    TODO
    vector< set<int> > neib = findNeighborSurperpixels(cluster);
    update cluster, cluster changes after this step
    updateNofeatureRegion(cluster,neib);
*/
    vector<cv::Point3f> center = getCenterFromCluster(cluster);

    vec2dd dist_mat = createDistMat(center);
/***
 *  TOADD
 *
 * if find neighbor is right,
 * then dist mat only need to calculate the distances between neighbor region
 *
 *
    vec2dd dist_mat = createDistMat(center,neib);
*/

    /// set the distance of those no features included surperpixels to -1
    updateDistMat(dist_mat,id_without);

    /// start to merge the neighbor superpixels
    std::map<int,int> id_link;

    for(int i = 0; i < id_without.size(); ++i)
    {
        /// i-th surperpixels ID donnot have feature
        /// id_without[i] means the superpixels' ID ,also the column number in dist_mat

        int target_id = findNearestId(dist_mat,id_without[i] );
        if(target_id == -1)
        {
            cerr << "Error: not find nearest points" << endl;
            return Mat();
        }
///        cout <<  id_without[i] << "\'s target = " << target_id << endl;
///        drawSpecID(cluster,id_without[i],target_id);

        id_link[ id_without[i] ] = target_id;

    }


/// then fix change the cluster again
    for(int i = 0 ; i < cluster.size(); ++i)
    {
        for(int j = 0; j < cluster[i].size(); ++j)
        {
            if(std::find(id_without.begin(), id_without.end(),cluster[i][j])!=id_without.end())
            {
                cluster[i][j] = id_link[ cluster[i][j] ];
            }
        }
    }

/// clustering features according to superpixel regions
    vector< set<int> > region_featureID;
    vector< cv::Mat> Homo;
    std::map<int,int> homo_link;

    for(int i = 0; i < findmaxclu(cluster); ++i)
    {
        set<int> tmp;
        tmp.insert(-1);
        region_featureID.push_back(tmp);
    }
    for(int i  =0; i < mat_point1.size(); ++i)
    {
        int id = cluster[int(mat_point1[i].x)][int(mat_point1[i].y)];
        region_featureID[id].insert(i);
    }
    for(int i = 0; i < region_featureID.size(); ++i)
    {
        std::vector<cv::Point2f> obj;
        std::vector<cv::Point2f> scene;
        for(std::set<int>::iterator j = region_featureID[i].begin();
            j!=region_featureID[i].end(); ++j)
        {
            if(*j != -1)
            {
                obj.push_back(mat_point1[*j]);
                scene.push_back(mat_point2[*j]);
            }
        }

        if(!obj.empty() && obj.size() >= 4) // obj.size() >= 4 is the lowest request,
                                            // in function findNofeatureId, there also has a limitation
        {
            // calculate Homography for small region
            Mat H = findHomography(obj,scene,RANSAC);
            cerr << "TODO: use distance H to filter those outilers" <<endl;
            /// todo
            /// update H again.
            Homo.push_back(H);
            homo_link[i] = Homo.size()-1;
        }

    }

    cerr << " /// if update Homography again, then merge regions without Homography" <<endl
         << "/// and update Homography again." << endl;

    cerr << "Incorrect Homographies lead to wrong results." <<endl;
    cout << "Homography size = " << Homo.size() << endl;
//    for(std::map<int,int>::iterator it = homo_link.begin(); it!=homo_link.end(); ++it)
//    {
//        cout << it->first << " -> " << it->second <<endl;
//    }

    // 这边还是断了呀，h这个的计算，还是...有些乱，算了先试试看吧，使用H，来假设是线性变化的，来开始投影吧

    Mat derformMat(cluster[0].size(), cluster.size(), CV_64FC3, Scalar(0,0,0));

    Mat xSrcTrans = Mat(1,1,CV_64FC3,Scalar(0,0,0));


    double max_r = 0, max_g = 0;

    for(int i = 0; i < cluster.size(); ++i)
    {
        for(int j = 0; j < cluster[0].size(); ++j)
        {
            double temp[3][1] = { (double)j,(double)i,1 };
            Mat xSrc = Mat(3,1,CV_64FC1,temp);
            Mat temp_h;
            temp_h = Homo[homo_link[cluster[i][j]]]; // should ensure each pixel maps to an Homography Mat
            xSrc = temp_h*xSrc - xSrc;

            xSrcTrans.at<Vec3d>(0,0)[0] = xSrc.at<Vec3d>(0,0)[0];
            xSrcTrans.at<Vec3d>(0,0)[1] = xSrc.at<Vec3d>(1,0)[0];
            xSrcTrans.at<Vec3d>(0,0)[2] = xSrc.at<Vec3d>(2,0)[0];

            derformMat.at<Vec3d>(j,i)[0] = ((xSrcTrans.at<Vec3d>(0,0)[0]) );
            derformMat.at<Vec3d>(j,i)[1] = ((xSrcTrans.at<Vec3d>(0,0)[1]) );
            derformMat.at<Vec3d>(j,i)[2] = ((xSrcTrans.at<Vec3d>(0,0)[2]) );

            if(max_r < (abs(xSrcTrans.at<Vec3d>(0,0)[0]) ))
            {
                max_r = (abs(xSrcTrans.at<Vec3d>(0,0)[0]) );
            }

            if(max_g < (abs(xSrcTrans.at<Vec3d>(0,0)[1]) ))
            {
                max_g = (abs(xSrcTrans.at<Vec3d>(0,0)[1]) );
            }
        }
    }

    // if derformation vector only use color information,
    // or set the transformation as RGB values.

//    imshow("vector",derformMat );

    return  derformMat;

}


vector<int> findNofeatureId(vector<  cv::Point2f >& mat_point1,vec2di& cluster)
{
    vector<int> vec_count_cluster;
    int max_clu = findmaxclu(cluster);

    for(int i = 0; i < max_clu; ++i)
    {
        vec_count_cluster.push_back(0); // vec_count_cluster -> ID
    }
    for(int i = 0; i < mat_point1.size(); ++i)
    {
        int ID = cluster[int(mat_point1[i].x)][int(mat_point1[i].y)];
        vec_count_cluster[ID]++; // count the number of features in region
    }

    vector<int> id_without;
    for(int i = 0; i < vec_count_cluster.size(); ++i)
    {
        if(vec_count_cluster[i] < 25) // for homography ... 4 is the lowest request
                                      // 30 is a trick....
                                      // small number leads to error matching and error warp result without update homography again
        {
            id_without.push_back(i);
        }
    }
    return id_without;
}

void showNeighbor(vector< set<int> >& neib)
{

    for(int i  =0; i < neib.size(); ++i)
    {
        if(neib[i].size() > 1)
        {
            cout << i << " -> ";
            for(std::set<int>::iterator j = neib[i].begin(); j != neib[i].end(); ++j)
            {
                cout << *j << " , ";
            }
            cout << endl;
        }
    }
}


vector< set<int> > findNeighborSurperpixels(vec2di& cluster)
{
    vector< set<int> > neighbor;
    int max_clu = findmaxclu(cluster);

    for(int i = 0; i < max_clu; ++i)
    {
        set<int> in;
        in.insert(-1);
        neighbor.push_back(in);
    }


    for(int i =0; i < cluster.size()-1; ++i)
    {
        for(int j =0 ;j < cluster[i].size()-1; ++j)
        {
            // for column
            if(cluster[i][j+1] != cluster[i][j])
            {
                neighbor[cluster[i][j]].insert(cluster[i][j+1]);
                neighbor[cluster[i][j+1]].insert(cluster[i][j]);
            }

            // for row

            if(cluster[i+1][j] != cluster[i][j])
            {
                neighbor[cluster[i][j]].insert(cluster[i+1][j]);
                neighbor[cluster[i+1][j]].insert(cluster[i][j]);
            }

        }
    }

    return neighbor;

}



void drawSpecID(vec2di& cluster, int& ori_id, int& tar_id)
{
    Mat img = Mat::zeros(cluster[0].size(),cluster.size(),CV_8UC3);
    for(int i = 0; i < cluster.size(); ++i)
    {
        for(int j =0; j < cluster[i].size(); ++j)
        {
            if(cluster[i][j] == ori_id)
            {
                img.at<Vec3b>(j,i)[0] = 255;
                img.at<Vec3b>(j,i)[1] = 0;
                img.at<Vec3b>(j,i)[2] = 0;
            }
            if(cluster[i][j] == tar_id)
            {
                img.at<Vec3b>(j,i)[0] = 0;
                img.at<Vec3b>(j,i)[1] = 0;
                img.at<Vec3b>(j,i)[2] = 255;
            }
        }
    }

    imshow("ori blue -> tar reg",img);
    waitKey(0);

}


// return superpixels ID
int findNearestId(vec2dd& dist, int& id)
{
    int min = 1000000;
    int targ_id = -1;

    for(int i = 0; i < dist[id].size(); ++i )
    {// column
        if(dist[id][i] > 0 && min > dist[id][i])
        {
            min = dist[id][i];
            targ_id = i;
        }
    }

    for(int i = id; i < dist.size(); ++i)
    {//  row
        if(dist[i][id] > 0 && min > dist[i][id])
        {
            min = dist[id][i];
            targ_id = i;
        }

    }

    return targ_id;


}

void updateDistMat(vec2dd & dist_mat, vector<int>& id)
{

    for(int i = 0 ; i < id.size()-1; ++i)
    {
        for(int j = i+1; j < id.size(); ++j)
        {
            if(id[i] > id[j]) // very important! since the structure of dist_mat like a triangle
            {
                dist_mat[ id[i] ][ id[j] ] = -1;
            }else{
                dist_mat[ id[j] ][ id[i] ] = -1;
            }
        }
    }

}

vec2dd createDistMat(vector<cv::Point3f> & center)
{
    vec2dd dist_mat;

        for(int i = 0; i < center.size();++i)
        {
            vector<double> dist_ab;
            for(int j = 0; j <=i ; ++j)
            {
                if(j == i)
                {
                    dist_ab.push_back(-1); // beside itself
                }else{
                    dist_ab.push_back(distance(center[i],center[j]));
                }
            }
            dist_mat.push_back(dist_ab);
            dist_ab.clear();

        }
        return dist_mat;
}

vec2dd createDistMat(vector<cv::Point3f> & center, vector< set<int> >& neib)
{
    vec2dd dist_mat;

    for(int i = 0; i < center.size(); ++i)
    {
        vector<double> dist_ab;
        for(int j  =0; j <= i ; ++j)
        {
            dist_ab.push_back(-1);
        }
        dist_mat.push_back(dist_ab);
    }

    for(int i =0; i < neib.size(); ++i)
    {
        for(std::set<int>::iterator j = neib[i].begin(); j!=neib[i].end(); ++j)
        {
            if(*j != -1)
            {
                if(*j <= i)
                {
                    dist_mat[i][*j] = distance(center[i],center[*j]);
                }else{
                    dist_mat[*j][i] = distance(center[i],center[*j]);
                }
            }
        }
    }



    return dist_mat;
}


double distance(cv::Point3f& point1 , cv::Point3f& point2)
{
    return sqrt(  pow(point1.x - point2.x, 2) + pow(point1.y - point2.y, 2)  );
}

vector<cv::Point3f> getCenterFromCluster(vec2di & cluster)
{
    // merge cluster
    vector< cv::Point3f > center;
    // init center
    int max_clus = findmaxclu(cluster);

    for(int i=0; i < max_clus; ++i)
    {
        center.push_back(Point3f(0,0,0));
    }


    for(int i = 0; i < cluster.size(); ++i)
    {
        for(int j = 0; j < cluster[0].size(); ++j)
        {
            center[cluster[i][j]].x += i;
            center[cluster[i][j]].y += j;
            center[cluster[i][j]].z += 1; // as counter
        }
    }// the sequence of center is not like the really Matrix

    for(int i = 0; i < center.size(); ++i)
    {
        if(abs(center[i].z)>1e-2 )
        {
            center[i].x = center[i].x/center[i].z;
            center[i].y = center[i].y/center[i].z;
        }
    }

    return center; // x,y,counter

}



void displayCenterWithCluster(vector< cv::Point3f > &center, vec2di & cluster)
{
    Mat sin = getColorCluster(cluster);
    for(int i = 0 ; i < center.size(); ++i)
    {
        if( abs(center[i].z)>1e-2 )
        {
            circle(sin,Point2f(center[i].x, center[i].y), 3,Scalar(255,0,255),2);
            stringstream ss;
            ss << i;
            string str = ss.str();
            putText(sin,str,Point2f(center[i].x, center[i].y),FONT_HERSHEY_COMPLEX,0.4,cv::Scalar(0, 0, 0),1,8,0);
        }
    }
    imshow("sin",sin);
    waitKey(0);
}

void displayMergeImage(Mat & img1_col, vector<  cv::Point2f > &mat_point1, Mat & img2_col, vector<  cv::Point2f > &mat_point2 )
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
    imwrite("/home/arvr/Desktop/match.jpg",match_img);
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


// vec2di cluster
// 2d: each column
// i: surperpixel ID

vec2di getSuperPixels(Mat& img1_col)
{
    Mat *lab_img1,lab1;

    cvtColor(img1_col,lab1,CV_BGR2Lab);

    lab_img1 = &lab1;

    int w = img1_col.cols, h = img1_col.rows;
    int nr_superpixels = 300;   // 300 is a trick， bigger nr_superpixel

    int nc = 150;                // 150 is a trick, bigger nc lead to

    double step = sqrt((w*h)/(double)(nr_superpixels)  );

    Slic sl;

    sl.generate_superpixels(lab_img1,step,nc);
    sl.create_connectivity(lab_img1);

    vec2di new_cluster = sl.get_new_cluster();
//    vec2di cluster = sl.get_cluster();

    return new_cluster;


    //vec2dd center = sl.get_center();
    //vec2dd distance = sl.get_distance();

//    Mat sup = getSingleCluster(cluster);
//    Mat new_sup = getColorCluster(new_cluster);
//    imshow("new_cluster",new_sup);
//    imshow("cluster",sup);
//    waitKey(0);

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
    return max+1; // since the vector donnot locate max at [max]

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
