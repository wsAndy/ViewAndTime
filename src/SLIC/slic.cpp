#include "SLIC/slic.h"

//using namespace superpixel;

//using namespace slic;

/*
 * Constructor. Nothing is done here.
 */
Slic::Slic() {

    cout << "init" << endl;

}

/*
 * Destructor. Clear any present data.
 */
Slic::~Slic() {
    clear_data();
}

/*
 * Clear the data as saved by the algorithm.
 *
 * Input : -
 * Output: -
 */
void Slic::clear_data() {
    clusters.clear();
    new_clusters.clear();
    distances.clear();
    centers.clear();
    center_counts.clear();
}

/*
 * Initialize the cluster centers and initial values of the pixel-wise cluster
 * assignment and distance values.
 *
 * Input : The image (Mat*).
 * Output: -
 */
void Slic::init_data(Mat *image) {
    /* Initialize the cluster and distance matrices. */
    for (int i = 0; i < image->cols; i++) {
        vector<int> cr;
        vector<double> dr;
        for (int j = 0; j < image->rows; j++) {
            cr.push_back(-1);
            dr.push_back(FLT_MAX);
        }
        clusters.push_back(cr);

        distances.push_back(dr);
    }

    /* Initialize the centers and counters. */
    for (int i = step; i < image->cols - step/2; i += step) {
        for (int j = step; j < image->rows - step/2; j += step) {
            vector<double> center;
            /* Find the local minimum (gradient-wise). */
            Point2d nc = find_local_minimum(image, Point2d(i,j));

            center.push_back(image->at<cv::Vec3b>(nc)[0]);
            center.push_back(image->at<cv::Vec3b>(nc)[1]);
            center.push_back(image->at<cv::Vec3b>(nc)[2]);


            /* Generate the center vector. */
            center.push_back(nc.x);
            center.push_back(nc.y);

            /* Append to vector of centers. */
            centers.push_back(center);
            center_counts.push_back(0);
        }
    }
}

/*
 * Compute the distance between a cluster center and an individual pixel.
 *
 * Input : The cluster index (int), the pixel (Point2d), and the Lab values of
 *         the pixel (Scalar).
 * Output: The distance (double).
 */
double Slic::compute_dist(int ci, Point2d pixel, Scalar colour) {

    double dc = sqrt(pow(centers[ci][0] - colour.val[0], 2) + pow(centers[ci][1]
                                                                  - colour.val[1], 2) + pow(centers[ci][2] - colour.val[2], 2));
    double ds = sqrt(pow(centers[ci][3] - pixel.x, 2) + pow(centers[ci][4] - pixel.y, 2));
    return sqrt(pow(dc / nc, 2) + pow(ds / ns, 2));
}

/*
 * Find a local gradient minimum of a pixel in a 3x3 neighbourhood. This
 * method is called upon initialization of the cluster centers.
 *
 * Input : The image (Mat*) and the pixel center (Point2d).
 * Output: The local gradient minimum (Point2d).
 */

Point2d Slic::find_local_minimum(Mat *image, Point2d center) {
    double min_grad = FLT_MAX;
    Point2d loc_min = Point2d(center.x, center.y);

    for (int i = center.x-1; i < center.x+2; i++) {
        for (int j = center.y-1; j < center.y+2; j++) {

            Scalar c1 = image->at<uchar>(j+1,i);
            Scalar c2 = image->at<uchar>(j,i+1);
            Scalar c3 = image->at<uchar>(j,i);

            /* Convert colour values to grayscale values. */
            double i1 = c1.val[0];
            double i2 = c2.val[0];
            double i3 = c3.val[0];
            /*double i1 = c1.val[0] * 0.11 + c1.val[1] * 0.59 + c1.val[2] * 0.3;
            double i2 = c2.val[0] * 0.11 + c2.val[1] * 0.59 + c2.val[2] * 0.3;
            double i3 = c3.val[0] * 0.11 + c3.val[1] * 0.59 + c3.val[2] * 0.3;*/

            /* Compute horizontal and vertical gradients and keep track of the
               minimum. */
            if (sqrt(pow(i1 - i3, 2)) + sqrt(pow(i2 - i3,2)) < min_grad) {
                min_grad = fabs(i1 - i3) + fabs(i2 - i3);
                loc_min.x = i;
                loc_min.y = j;
            }
        }
    }

    return loc_min;
}

/*
 * Compute the over-segmentation based on the step-size and relative weighting
 * of the pixel and colour values.
 *
 * Input : The Lab image (Mat*), the stepsize (int), and the weight (int).
 * Output: -
 */
void Slic::generate_superpixels(Mat *image, int step, int nc) {
    this->step = step;
    this->nc = nc;
    this->ns = step;

    /* Clear previous data (if any), and re-initialize it. */
    clear_data();
    init_data(image);

    /* Run EM for 10 iterations (as prescribed by the algorithm). */
    for (int i = 0; i < NR_ITERATIONS; i++) {
        /* Reset distance values. */
        for (int j = 0; j < image->cols; j++) {
            for (int k = 0;k < image->rows; k++) {
                distances[j][k] = FLT_MAX;
            }
        }

        for (int j = 0; j < (int) centers.size(); j++) {
            /* Only compare to pixels in a 2 x step by 2 x step region. */
            for (int k = centers[j][3] - step; k < centers[j][3] + step; k++) {
                for (int l = centers[j][4] - step; l < centers[j][4] + step; l++) {

                    if (k >= 0 && k < image->cols && l >= 0 && l < image->rows) {

//                        Scalar colour = Scalar(0,0,0);
//                        Vec3b inten = image->at<Vec3b>(l,k);
//                        for(int ch = 0; ch < image->channels(); ++ch)
//                        {
//                            colour.val[0] = inten.val[0];
//                            colour.val[1] = inten.val[1];
//                            colour.val[2] = inten.val[2];
//                        }

                        Scalar colour = (Scalar)image->at<Vec3b>(l,k);


                        double d = compute_dist(j, Point2d(k,l), colour);


                        /* Update cluster allocation if the cluster minimizes the
                           distance. */
                        if (d < distances[k][l]) {
                            distances[k][l] = d;
                            clusters[k][l] = j;
                        }
                    } // if (k >= 0 && k < image




                }
            }
        }

        /* Clear the center values. */
        for (int j = 0; j < (int) centers.size(); j++) {
            centers[j][0] = centers[j][1] = centers[j][2] = centers[j][3] = centers[j][4] = 0;
            center_counts[j] = 0;
        }

        /* Compute the new cluster centers. */
        for (int j = 0; j < image->cols; j++) {
            for (int k = 0; k < image->rows; k++) {
                int c_id = clusters[j][k];

                if (c_id != -1) {

//                    Scalar colour = Scalar(0,0,0);
//                    Vec3b inten = image->at<Vec3b>(k,j);
//                    for(int ch = 0; ch < image->channels(); ++ch)
//                    {
//                        colour.val[0] = inten.val[0];
//                        colour.val[1] = inten.val[1];
//                        colour.val[2] = inten.val[2];
//                    }

                    Scalar colour = (Scalar)image->at<Vec3b>(k,j);

                    centers[c_id][0] += colour.val[0];
                    centers[c_id][1] += colour.val[1];
                    centers[c_id][2] += colour.val[2];
                    centers[c_id][3] += j;
                    centers[c_id][4] += k;

                    center_counts[c_id] += 1;
                }
            }
        }

        /* Normalize the clusters. */
        for (int j = 0; j < (int) centers.size(); j++) {
            centers[j][0] /= center_counts[j];
            centers[j][1] /= center_counts[j];
            centers[j][2] /= center_counts[j];
            centers[j][3] /= center_counts[j];
            centers[j][4] /= center_counts[j];
        }
    }

}

/*
 * Enforce connectivity of the superpixels. This part is not actively discussed
 * in the paper, but forms an active part of the implementation of the authors
 * of the paper.
 *
 * Input : The image (Mat*).
 * Output: -
 */
void Slic::create_connectivity(Mat *image) {
    int label = 0, adjlabel = 0;
    const int lims = (image->cols * image->rows) / ((int)centers.size());

    const int dx4[4] = {-1,  0,  1,  0};
    const int dy4[4] = { 0, -1,  0,  1};

    /* Initialize the new cluster matrix. */

    for (int i = 0; i < image->cols; i++) {
        vector<int> nc;
        for (int j = 0; j < image->rows; j++) {
            nc.push_back(-1);
        }
        new_clusters.push_back(nc);
    }

    for (int i = 0; i < image->cols; i++) {
        for (int j = 0; j < image->rows; j++) {
            if (new_clusters[i][j] == -1) {
                vector<Point2d> elements;
                elements.push_back(Point2d(i, j));

                /* Find an adjacent label, for possible use later. */
                for (int k = 0; k < 4; k++) {
                    int x = elements[0].x + dx4[k], y = elements[0].y + dy4[k];

                    if (x >= 0 && x < image->cols && y >= 0 && y < image->rows) {
                        if (new_clusters[x][y] >= 0) {
                            adjlabel = new_clusters[x][y];
                        }
                    }
                }

                int count = 1;
                for (int c = 0; c < count; c++) {
                    for (int k = 0; k < 4; k++) {
                        int x = elements[c].x + dx4[k], y = elements[c].y + dy4[k];

                        if (x >= 0 && x < image->cols && y >= 0 && y < image->rows) {
                            if (new_clusters[x][y] == -1 && clusters[i][j] == clusters[x][y]) {
                                elements.push_back(Point2d(x, y));
                                new_clusters[x][y] = label;
                                count += 1;
                            }
                        }
                    }
                }

                /* Use the earlier found adjacent label if a segment size is
                   smaller than a limit. */
                if (count <= lims >> 2) {
                    for (int c = 0; c < count; c++) {
                        new_clusters[elements[c].x][elements[c].y] = adjlabel;
                    }
                    label -= 1;
                }
                label += 1;
            }
        }
    } // for


}

/*
 * Display the cluster centers.
 *
 * Input : The image to display upon (Mat*) and the colour (Scalar).
 * Output: -
 */
void Slic::display_center_grid(Mat *image, Scalar colour) {

    for (int i = 0; i < (int) centers.size(); i++) {
        circle(*image, Point2d(centers[i][3], centers[i][4]), 2, colour, 2);
    }

}

/*
 * Display a single pixel wide contour around the clusters.
 *
 * Input : The target image (Mat*) and contour colour (Scalar).
 * Output: -
 */
void Slic::display_contours(Mat *image, Scalar colour) {
    const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
    const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

    /* Initialize the contour vector and the matrix detailing whether a pixel
     * is already taken to be a contour. */
    vector<Point2d> contours;
    vec2db istaken;
    for (int i = 0; i < image->cols; i++) {
        vector<bool> nb;
        for (int j = 0; j < image->rows; j++) {
            nb.push_back(false);
        }
        istaken.push_back(nb);
    }
    /* Go through all the pixels. */
    for (int i = 0; i < image->cols; i++) {
        for (int j = 0; j < image->rows; j++) {
            int nr_p = 0;

            /* Compare the pixel to its 8 neighbours. */
            for (int k = 0; k < 8; k++) {
                int x = i + dx8[k], y = j + dy8[k];

                if (x >= 0 && x < image->cols && y >= 0 && y < image->rows) {
                    if (istaken[x][y] == false && clusters[i][j] != clusters[x][y]) {
                        nr_p += 1;
                    }
                }
            }

            /* Add the pixel to the contour list if desired. */
            if (nr_p >= 2) {
                contours.push_back(Point2d(i,j));
                istaken[i][j] = true;
            }
        }
    }

    /* Draw the contour pixels. */
    for (int i = 0; i < (int)contours.size(); i++) {
        image->at<Vec3b>(contours[i].y,contours[i].x)[0] = colour.val[0];
        image->at<Vec3b>(contours[i].y,contours[i].x)[1] = colour.val[1];
        image->at<Vec3b>(contours[i].y,contours[i].x)[2] = colour.val[2];
    }
}

/*
 * Give the pixels of each cluster the same colour values. The specified colour
 * is the mean RGB colour per cluster.
 *
 * Input : The target image (Mat*).
 * Output: -
 */
void Slic::colour_with_cluster_means(Mat *image) {

    vector<Scalar> colours(centers.size());

    /* Gather the colour values per cluster. */
    for (int i = 0; i < image->cols; i++) {
        for (int j = 0; j < image->rows; j++) {
            int index = clusters[i][j];

            Scalar colour = (Scalar)image->at<Vec3b>(j, i);

            colours[index].val[0] += colour.val[0];
            colours[index].val[1] += colour.val[1];
            colours[index].val[2] += colour.val[2];
        }
    }

    /* Divide by the number of pixels per cluster to get the mean colour. */
    for (int i = 0; i < (int)colours.size(); i++) {
        colours[i].val[0] /= center_counts[i];
        colours[i].val[1] /= center_counts[i];
        colours[i].val[2] /= center_counts[i];
    }

    /* Fill in. */
    for (int i = 0; i < image->cols; i++) {
        for (int j = 0; j < image->rows; j++) {
            Scalar ncolour = colours[clusters[i][j]];
//            cvSet2D(image, j, i, ncolour);

            image->at<Vec3b>(j,i)[0] = ncolour.val[0];
            image->at<Vec3b>(j,i)[1] = ncolour.val[1];
            image->at<Vec3b>(j,i)[2] = ncolour.val[2];


        }
    }
}


vec2di Slic::get_cluster()
{
    return this->clusters;

}

vec2di Slic::get_new_cluster()
{
    return this->new_clusters;

}

vec2dd Slic::get_center()
{
    return this->centers;
}

vec2dd Slic::get_distance()
{
    return this->distances;
}

