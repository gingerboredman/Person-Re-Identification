#include <opencv4/opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;



const char* keys =
    "{ help  h| | run with path to images to be compared (localised images) }"
    "{ @input1 | | Path to input image 1. }"
    "{ @input2 | | Path to input image 2. }";


Mat1b grabCutWrapper(Mat img, int iterations)
{
    Mat1b markers(img.rows, img.cols);
    markers.setTo(GC_PR_BGD);

// cut out a small area in the middle of the image
    int m_rows = 0.1 * img.rows;
    int m_cols = 0.1 * img.cols;
// of course here you could also use Rect() instead of Range to select 
// the region of interest
    Mat1b fg_seed = markers(Range(img.rows/2 - m_rows/2, img.rows/2 + m_rows/2), 
                            Range(img.cols/2 - m_cols/2, img.cols/2 + m_cols/2));
// mark it as foreground
    fg_seed.setTo(GC_FGD);

// select first 5 rows of the image as background
    Mat1b bg_seed = markers(Range(0, 5),Range::all());
    bg_seed.setTo(GC_BGD);

    Mat bgd, fgd;

    grabCut(img, markers, Rect(), bgd, fgd, iterations, GC_INIT_WITH_MASK);

    // let's get all foreground and possible foreground pixels
    Mat1b mask_fgpf = ( markers == GC_FGD) | ( markers == GC_PR_FGD);
    // and copy all the foreground-pixels to a temporary image
    cv::Mat3b tmp = cv::Mat3b::zeros(img.rows, img.cols);
img.copyTo(tmp, mask_fgpf);
// show it


    return mask_fgpf;
}

int main(int argc, char** argv)
{

    CommandLineParser parser( argc, argv, keys );

    Mat src1 = imread("/home/aditya/Code/Re-Id/t1.jpg");
    // cout<<src1;
    Mat src2 = imread("/home/aditya/Code/Re-Id/test2.jpg");
    // cout<<src2;
    if (!src1.data || !src2.data)
    { 
        return 1; }

    // src1 = grabCutWrapper(src1, 1);
    // src2 = grabCutWrapper(src2, 1);
    Mat hsv1 ,hsv2;
    cvtColor( src1, hsv1, COLOR_BGR2HSV );
    cvtColor( src2, hsv2, COLOR_BGR2HSV );
    
    int h_bins = 50, s_bins = 60;
    int histSize[] = { h_bins, s_bins };
    // hue varies from 0 to 179, saturation from 0 to 255
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };
    const float* ranges[] = { h_ranges, s_ranges };
    // Use the 0-th and 1-st channels
    int channels[] = { 0, 1 };

    Mat hist1, hist2;

    calcHist( &hsv1, 1, channels, Mat(), hist1, 2, histSize, ranges, true, false );
    normalize( hist1, hist1, 0, 1, NORM_MINMAX, -1, Mat() );
    calcHist( &hsv2, 1, channels, Mat(), hist2, 2, histSize, ranges, true, false );
    normalize( hist2, hist2, 0, 1, NORM_MINMAX, -1, Mat() );

    for( int compare_method = 0; compare_method < 4; compare_method++ )
    {
        double metric = compareHist( hist1, hist2, compare_method );
        cout<< "Method " << compare_method << ":" << metric << "\n";
    }
    
    return 0;
}