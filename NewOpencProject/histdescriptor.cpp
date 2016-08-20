#include "histdescriptor.h"


using namespace cv;
using namespace std;

Mat DescriptorY::ImageHistCompute(Mat image){
    
    int width, height;
    vector<Mat>Hists;
    width = image.cols; //get image width
    height = image.rows;
    
    binWidth = width / binCountX; // calculate bin width
    binHeight = height / binCountY;
    
    for (int i = 0; i<(binCountX*binCountY); i++){
        binLocx = i%binCountX; //the x coordinate of a certain bin with order i
        binLocy = floor(i / binCountX);// the y coordinate of a certain bin
        binMask = image(Range(binCountY*binLocy + 1, binCountY*(binLocy + 1)),
                        Range(binCountX*binLocx + 1, binCountX*(binLocx + 1)));
        cvtColor(binMask, binMask, CV_BGR2Lab); //change color space
        split(binMask, bgr_planes);
        bool accumulate = false;
        bool uniform = true;
        // Compute the histograms:
        calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
        calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
        calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);
        
        /// Normalize the result to [ 0, histImage.rows ]
        normalize(b_hist, b_hist, 0, hist_h, NORM_MINMAX, -1, Mat());
        normalize(g_hist, g_hist, 0, hist_h, NORM_MINMAX, -1, Mat());
        normalize(r_hist, r_hist, 0, hist_h, NORM_MINMAX, -1, Mat());
        
        /*a = sqrt(norm(b_hist));
         b = sqrt(norm(g_hist));
         c = sqrt(norm(r_hist));
         b_hist = b_hist / a;
         g_hist = g_hist / b;
         r_hist = r_hist / c;
         */
        
        vconcat(b_hist, g_hist, combo1);
        vconcat(combo1, r_hist, binHist);
        binHist = binHist.t(); //transpose the matrix
        Hists.push_back(binHist);
    }
    
    combo2 = Hists[0];
    for (vector<Mat>::size_type i = 1; i < binCountX*binCountY; i++)
        vconcat(combo2, Hists[i], combo2);
    return combo2;
    
}