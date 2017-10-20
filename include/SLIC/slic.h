/*
 *  This file is copied from:
 *   `https://github.com/PSMM/SLIC-Superpixels`
 *
 *    since it use an old opencv version, I convert it to opencv2.4.13
 *
 *
 * */

#ifndef _SLIC_H_
#define _SLIC_H_

#include "opencv2/opencv.hpp"

#include "stdio.h"
#include "math.h"
#include "float.h"
#include "vector"

using namespace std;
using namespace cv;

#define vec2dd vector< vector<double> >
#define vec2di vector< vector<int> >
#define vec2db vector< vector<bool> >

#define NR_ITERATIONS 10

//namespace superpixel{
/*
 * class Slic.
 *
 * In this class, an over-segmentation is created of an image, provided by the
 * step-size (distance between initial cluster locations) and the colour
 * distance parameter.
 */
class Slic {
private:
    /* The cluster assignments and distance values for each pixel. */
    vec2di clusters;
    vec2dd distances;

    /* The LAB and xy values of the centers. */
    vec2dd centers;
    /* The number of occurences of each center. */
    vector<int> center_counts;

    vec2di new_clusters;

    /* The step size per cluster, and the colour (nc) and distance (ns)
     * parameters. */
    int step, nc, ns;

    /* Compute the distance between a center and an individual pixel. */
    double compute_dist(int ci, Point2d pixel, Scalar colour);
    /* Find the pixel with the lowest gradient in a 3x3 surrounding. */
    Point2d find_local_minimum(Mat *image, Point2d center);

    /* Remove and initialize the 2d vectors. */
    void clear_data();
    void init_data(Mat *image);

public:
    /* Class constructors and deconstructors. */
    Slic();
    ~Slic();

    /* Generate an over-segmentation for an image. */
    void generate_superpixels(Mat *image, int step, int nc);
    /* Enforce connectivity for an image. */
    void create_connectivity(Mat *image);

    /* Draw functions. Resp. displayal of the centers and the contours. */
    void display_center_grid(Mat *image, Scalar colour);
    void display_contours(Mat *image, Scalar colour);
    void colour_with_cluster_means(Mat *image);

    vec2di get_cluster();

    vec2dd get_center();

    vec2dd get_distance();

    vec2di get_new_cluster();

};


//}

#endif
