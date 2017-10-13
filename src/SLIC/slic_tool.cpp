
/****
 *
 *
 *
 *   writed by sheng wang
 *   2017-9-15
 *
 */


#include "SLIC/slic_tool.h"
#include "SLIC/slic.h"

//using namespace superpixel;
//using namespace slic_tool;


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
                int x =  i + int(derformMat1.at<Vec3d>(i,j)[1] * ind / (Vir_num+1) ); // row
                int y =  j + int(derformMat1.at<Vec3d>(i,j)[0] * ind / (Vir_num+1) ); // col

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
                int x =  i + int(derformMat2.at<Vec3d>(i,j)[1] * (1 - double(ind) / (Vir_num+1)) ); // row
                int y =  j + int(derformMat2.at<Vec3d>(i,j)[0] * (1 - double(ind) / (Vir_num+1)) ); // col

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

        Mat mask = cv::Mat::zeros(blend_image.rows, blend_image.cols, CV_8UC1);
        for(int i = 0; i < blend_image.rows; ++i)
        {
            for(int j = 0; j < blend_image.cols; ++j)
            {
                if(blend_image.at<Vec3b>(i,j)[0] ==0 &&
                   blend_image.at<Vec3b>(i,j)[1] ==0 &&
                   blend_image.at<Vec3b>(i,j)[2] ==0     )
                {
                    mask.at<uchar>(i,j) = 255;
                }
            }
        }

//        imshow("mask",mask);
//        waitKey(0);

//        cv::inpaint(blend_image,mask,blend_image,3,CV_INPAINT_TELEA);

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
                // derformMat这边存储的是位置的偏移，所以在逻辑上也不成问题
                int x =  i + int(derformMat.at<Vec3d>(i,j)[1] * ind *0.1); // row
                int y =  j + int(derformMat.at<Vec3d>(i,j)[0] * ind *0.1); // col

                if( x < 0 || x >= vir_image.rows || y < 0 || y >= vir_image.cols)
                {
                    continue;
                }

                vir_image.at<Vec3b>(x,y)[0] = img1_col.at<Vec3b>(i,j)[0] ;

                vir_image.at<Vec3b>(x,y)[1] = img1_col.at<Vec3b>(i,j)[1] ;

                vir_image.at<Vec3b>(x,y)[2] = img1_col.at<Vec3b>(i,j)[2] ;

            }
        }


        Mat mask = cv::Mat::zeros(vir_image.rows, vir_image.cols, CV_8UC1);
        for(int i = 0; i < vir_image.rows; ++i)
        {
            for(int j = 0; j < vir_image.cols; ++j)
            {
                if(vir_image.at<Vec3b>(i,j)[0] ==0 &&
                   vir_image.at<Vec3b>(i,j)[1] ==0 &&
                   vir_image.at<Vec3b>(i,j)[2] ==0     )
                {
                    mask.at<uchar>(i,j) = 255;
                }
            }
        }

//        imshow("mask",mask);
//        waitKey(0);


// inpaint cost too much time !!!!
        cout << "inpaint cost too much time" <<endl;
//        cv::inpaint(vir_image,mask,vir_image,3,CV_INPAINT_TELEA);

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

set<int> getClusterID(vector<vector<int> > & cluster)
{
    set<int > id;
    for(int i = 0; i < cluster.size(); ++i)
    {// cols
        for(int j = 0; j < cluster[0].size(); +j)
        {//rows
            id.insert(cluster[i][j]);
        }
    }
    return id;
}

vector<int> findNotenoughFeatureID(vector<vector<int> > & cluster, vector<Point2f> & mat_point)
{
    return findNofeatureId(mat_point,cluster);
}


void findNearestIdAndUpdateCluster(vector<vector<int> > & cluster,
                                   vec2dd& dist_table,
                                   vector<Point3f>& center,
                                   vector<int>& id_no)
{
    map<int, int> link;

    for(int i = 0; i < id_no.size(); ++i)
    {

        int target_id = findNearestId(dist_table,id_no[i]);
        if(target_id == -1)
        {
            cerr << "ERROR: not find nearest id in findNearestIdAndUpdateCluster" << endl;
        }
//        cout <<id_no[i] << " --> " << target_id << endl;
        link[ id_no[i] ] = target_id;
    }

    for(int i = 0 ; i < cluster.size(); ++i)
    {
        for(int j = 0; j < cluster[i].size(); ++j)
        {
            if(std::find(id_no.begin(), id_no.end(),cluster[i][j])!=id_no.end())
            {
                cluster[i][j] = link[ cluster[i][j] ];
            }
        }
    }

    // cluster has been update
    // then update dist_table and center

    // update center
    std::vector<cv::Point3f> center_new = getCenterFromCluster(cluster);

    int mmax = findmaxclu(cluster);

    vec2dd dist_table_new = createDistMat(center_new);

    // now all the superpixels has enough features and just create a new dist_table
    dist_table.clear();

    for(int i = 0; i < dist_table_new.size(); ++i)
    {
        dist_table.push_back(dist_table_new[i]);
    }

    center.clear();

    for(int i = 0; i < center_new.size(); ++i)
    {

        center.push_back(center_new[i]);

    }

}

void calHomo(vec2di& cluster, vector<Point2f>& mat_point1, vector<Point2f>& mat_point2, std::vector<cv::Mat>& Homo, std::map<int,int>& homo_link, vec2dd& dist_table,vector<Point3f>& center ,int count)
{
    count ++;
    Homo.clear();
    homo_link.clear();

    vector< set<int> > region_featureID;
    std::vector<int> merge_again_id;

    displayCenterWithCluster(center,cluster);

    for(int i = 0; i < findmaxclu(cluster); ++i)
    {
        set<int> tmp;

        tmp.insert(-1);
        region_featureID.push_back(tmp);
    }
    for(int i = 0; i < mat_point1.size(); ++i)
    {
        int id = cluster[int(mat_point1[i].x)][int(mat_point1[i].y)];
        region_featureID[id].insert(i); // means id-th superpixel save feature i-th feature
    }

    cout << "region_featureID.size() = " << region_featureID.size() <<endl;
    for(int i = 0; i < region_featureID.size(); ++i)
    {
        if(region_featureID[i].size() == 1)    continue;

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

        if ( iteHomo(obj,scene,Homo,homo_link,i) )
        {
            // get homography with enough feature
            ;
        }
        else{
            // need to merge again
            cout << "SUPERPIXLE ID: " << i << "   need to merge again."<< endl;
            merge_again_id.push_back(i); // i is cluster ID
        }
    }

    if(merge_again_id.size() > 0 && count <= ITERATOR_TIMES_FOR_HOMO )
    {
        cout << "merge again " <<endl;
        findNearestIdAndUpdateCluster(cluster, dist_table, center,merge_again_id);

        calHomo(cluster,mat_point1,mat_point2,Homo,homo_link, dist_table,center, count);
    }



}

bool iteHomo(vector<Point2f>& obj, vector<Point2f>& scene, std::vector<cv::Mat>& Homo, std::map<int,int>& homo_link, int i)
{
    if(!obj.empty() && obj.size() >= HOMO_MIN_NUM && scene.size() >= HOMO_MIN_NUM)
    {
        Mat H = findHomography(obj,scene,CV_RANSAC);

        if(judgeHomoDistance(H,obj,scene))
        {
            Homo.push_back(H);
            homo_link[i] = Homo.size() - 1;
            return true;
        }
        else{
            // iterate
            // update obj and scene first???
            iteHomo(obj,scene, Homo,homo_link,i);
        }
    }else{
        // not have enough feature
        return false;
    }
}


void getSuperpixelHomo(vec2di& cluster, std::vector<cv::Mat>& Homo, std::map<int,int>& homo_link,
                       std::vector<cv::Point2f>& mat_point1, std::vector<cv::Point2f>& mat_point2, int& count)
{
    count = 0;

//    set<int> id = getClusterID(cluster);
    vector<cv::Point3f> center = getCenterFromCluster(cluster);

    vector<int> id_NotEnough = findNotenoughFeatureID(cluster,mat_point1);

    for(int i = 0; i < id_NotEnough.size() ; ++i)
    {
        cout << id_NotEnough[i] << "   ";
    }
    cout << endl;

    vec2dd dist_table = createDistMat(center);
    updateDistMat(dist_table,id_NotEnough);

    /// cluster , dist_table and center all has been changed
    /// id_NotEnough is useless now
    /// dince the whole cluster has enough feature.
    findNearestIdAndUpdateCluster(cluster, dist_table, center, id_NotEnough);

    cout << "into calHomo" <<endl;
    calHomo(cluster,mat_point1,mat_point2,Homo,homo_link, dist_table, center, count);
}

void iteratorGetHomo(vec2di& cluster, std::vector<cv::Mat>& Homo, std::map<int,int>& homo_link, std::vector<cv::Point2f>& mat_point1, std::vector<cv::Point2f>& mat_point2, int& count)
{


    count = count + 1;

    Homo.clear();
    homo_link.clear();

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

    *   a problem is that if few feature in the image,
    *   many superpixels will not have features and
    *   it's hard to merge all of them

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
    vector<int> id_without = findNofeatureId(mat_point1,cluster);

    cout << "id_without size = " << id_without.size() << endl;

//    displayCenterWithCluster(center,cluster);

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
            cerr << "Error: not find nearest points from ID [" <<  id_without[i] << "]"<< endl;
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
///
    vector< set<int> > region_featureID;
    std::vector<int> merge_again_id;
    bool bool_merge_again = false; // for special cases when some region have too much outliers and need to merge with other region again

    for(int i = 0; i < findmaxclu(cluster); ++i)
    {
        set<int> tmp;
        tmp.insert(-1);
        region_featureID.push_back(tmp);
    }
    for(int i = 0; i < mat_point1.size(); ++i)
    {
        int id = cluster[int(mat_point1[i].x)][int(mat_point1[i].y)];
        region_featureID[id].insert(i); // means id-th superpixel save feature i-th feature
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
            if( judgeHomoDistance(H,obj,scene) )
            {
                Homo.push_back(H);
                homo_link[i] = Homo.size()-1;
            }else{
                // obj and scene have been updated in judgeHomoDistance(H,obj,scene)
                // again and update Homography

                if(!obj.empty() && obj.size() >= 4 && scene.size() >= 4)
                {
                    H = findHomography(obj,scene,RANSAC);
                    Homo.push_back(H);
                    homo_link[i] = Homo.size()-1;
                }else{
                    // merge region again
                    cout << endl<<endl<< "need merge region, since this region donnot have enough feature" <<endl<<endl;
                    merge_again_id.push_back( i );
                    bool_merge_again = true;
//                    break; // don't break, save those

                    // break from for(int i = 0; i < region_featureID.size(); ++i)
                    // then update `superpixel`,`center`, `region`

                }

            }
        }

    } // for(int i = 0; i < region_featureID.size(); ++i)

    if(bool_merge_again)
    {
        // here, write update again, maybe its a big work, but since this case occurs by chance and the update coda is too chaos
        // write it additionally

        /// ********************************************************
        /// input: merge_again_id
        ///        cluster
        /// ********************************************************

        // update cluster first
        std::map<int,int> id_link_again;
        for(int i = 0; i < merge_again_id.size(); ++i)
        {
            // superpixel id : merge_again_id[i]
            int target_id = findNearestId(dist_mat, merge_again_id[i]);
            if(target_id == -1)
            {
                cerr << "Error: not find nearest points for ID [" << merge_again_id[i] << "]" << endl;
            }
            id_link_again[ merge_again_id[i] ] = target_id;
        }

        for(int i = 0 ; i < cluster.size(); ++i)
        {
            for(int j = 0; j < cluster[i].size(); ++j)
            {
                if(std::find(merge_again_id.begin(), merge_again_id.end(),cluster[i][j])!=merge_again_id.end())
                {
                    cluster[i][j] = id_link_again[ cluster[i][j] ];
                }
            }
        }

        // now cluster update again
        // run again

        cout << "mat_point = " <<  mat_point1.size() <<endl;

        if(count < ITERATOR_TIMES_FOR_HOMO)
        {
            iteratorGetHomo(cluster, Homo, homo_link, mat_point1, mat_point2, count);
        }

    }


}

Mat getCorresponseMaps(vec2di& cluster,vector<  cv::Point2f >& mat_point1,vector<  cv::Point2f >& mat_point2)
{

    vector< cv::Mat> Homo;
    std::map<int,int> homo_link;
    int counter_gethomo = 0;

////    TEST !!!!
///
///

    getSuperpixelHomo(cluster,Homo,homo_link, mat_point1, mat_point2, counter_gethomo);

//    iteratorGetHomo(cluster,Homo,homo_link, mat_point1, mat_point2, counter_gethomo);


    cerr << "Incorrect Homographies lead to wrong results." <<endl;
    cout << "Homography size = " << Homo.size() << endl;
//    for(std::map<int,int>::iterator it = homo_link.begin(); it!=homo_link.end(); ++it)
//    {
//        cout << it->first << " -> " << it->second <<endl;
//    }

    // 这边还是断了呀，h这个的计算，还是...有些乱，算了先试试看吧，使用H，来假设是线性变化的，来开始投影吧

    Mat derformMat(cluster[0].size(), cluster.size(), CV_64FC3, Scalar(0,0,0));


//    Mat H = findHomography(mat_point1, mat_point2, CV_RANSAC);

    double max_r = 0, max_g = 0;

    for(int i = 0; i < cluster.size(); ++i) // col  -----
    {
        for(int j = 0; j < cluster[0].size(); ++j) // row |
        {
            Mat temp_h;
//            temp_h = H;
            temp_h = Homo[homo_link[cluster[i][j]]]; // should ensure each pixel maps to an Homography Mat

            vector<Point2f> pix,target_pix;
            pix.push_back(Point2f(i,j));

            perspectiveTransform(pix,target_pix,temp_h);

            derformMat.at<Vec3d>(j,i)[0] = (target_pix[0].x - i); // col
            derformMat.at<Vec3d>(j,i)[1] = (target_pix[0].y - j); // row
            derformMat.at<Vec3d>(j,i)[2] = 1;

            if(max_r < (abs(target_pix[0].x - i) ))
            {
                max_r = (abs(target_pix[0].x - i) );
            }

            if(max_g < (abs(target_pix[0].y - j) ))
            {
                max_g = (abs(target_pix[0].y - j) );
            }
        }
    }

    // if derformation vector only use color information,
    // or set the transformation as RGB values.

//    imshow("vector",derformMat );

    return  derformMat;

}

bool judgeHomoDistance( Mat& H,vector<cv::Point2f>& obj,vector<cv::Point2f>& scene )
{
    vector<int> outlier_id;
    for(int i =0 ;i < obj.size(); ++i)
    {
        double x1_[3][1] = { obj[i].x,obj[i].y,1 };
        Mat x1 = Mat(3,1,CV_64FC1,x1_);
        double x2_[3][1] = { scene[i].x,scene[i].y,1 };
        Mat x2 = Mat(3,1,CV_64FC1,x2_);

        Mat dil = H*x1 - x2;
        if(sqrt( pow(dil.at<Vec3d>(0,0)[0],2) + pow(dil.at<Vec3d>(1,0)[0],2) ) <= 3 ) // within 3 pixel regions
        {
            continue;
        }else{
            outlier_id.push_back(i);
        }

    }

    if(outlier_id.size() == 0)
    {
        return true;
    }
    else{
        // update obj and scene first
        int count = 0;

        for(int i = 0; i < outlier_id.size(); ++i)
        {
            obj[ outlier_id[i] ].x = -10;
            scene[ outlier_id[i] ].x = -10;
        }
        for(std::vector<cv::Point2f>::iterator it = obj.begin(); it != obj.end();)
        {
            if( abs((*it).x+10) < 1e-4 )
            {
                it = obj.erase(it);
                count++;
            }else{
                ++it;
            }
            if(count >= outlier_id.size())
            {
                break;
            }
        }

        count = 0;
        for(std::vector<cv::Point2f>::iterator it = scene.begin(); it != scene.end();)
        {
            if( abs((*it).x+10) < 1e-4 )
            {
                it = scene.erase(it);
                count++;
            }else{
                ++it;
            }
            if(count >= outlier_id.size())
            {
                break;
            }
        }
        // update obj and scene end.

        return false;
    }
}


vector<int> findNofeatureId(vector<  cv::Point2f >& mat_point1,vec2di& cluster)
{
    vector<int> vec_count_cluster;
    int max_clu = findmaxclu(cluster);
    cout << "findNofeatureId: cluster size = " << max_clu << endl;

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
//        cout << vec_count_cluster[i] << endl;
        if(vec_count_cluster[i] < FEATURE_NUMBER_REGION) // for homography ... 4 is the lowest request
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

    cout << "dist[id].size = " << dist[id].size() << endl;
    cout << "dist.size = " << dist.size() << endl;

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
            for(int j = 0; j <= i ; ++j)
            {
                if(j == i)
                {
                    dist_ab.push_back(-1); // beside itself
                }else{
                  if(abs(center[i].z) < 1e-4 || abs(center[j].z) < 1e-4)
                  {
                    dist_ab.push_back(-1);
                  }else{ // fix bugs....
                    dist_ab.push_back(distance(center[i],center[j]));
                  }
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
//    imwrite("/Users/sheng/Desktop/superpixel_center.jpg",sin);
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
//    imwrite("/home/arvr/Desktop/match.jpg",match_img);
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
    int nr_superpixels = 150;   // 300 is a trick， bigger nr_superpixel

    int nc = 100;                // 150 is a trick, bigger nc lead to

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

//    vector<KeyPoint> RR_keypoint01,RR_keypoint02;
//    vector<DMatch> RR_matches;            //重新定义RR_keypoint 和RR_matches来存储新的关键点和匹配矩阵
//    int index=0;

    for (size_t i=0;i<good_matches.size();i++)
    {
        if (RansacStatus[i]!=0)
        {
//            RR_keypoint01.push_back(R_keypoint01[i]);
//            RR_keypoint02.push_back(R_keypoint02[i]);
//            good_matches[i].queryIdx=index;
//            good_matches[i].trainIdx=index;
//            RR_matches.push_back(good_matches[i]);
//            index++;


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
