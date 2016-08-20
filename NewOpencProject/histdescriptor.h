#ifndef _HISTDESCRIPTOR_H
#define _HISTDESCRIPTOR_H

#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/core/core.hpp>
#include<iostream>
#include<algorithm>
#include<math.h>
#include <iostream>
#include "descriptor.h"

using namespace cv;
using namespace std;


class DescriptorY : public Descriptor {
    
public:
    Mat image, binMask, binHist, binsHistCombo;
    Mat src, combo1,combo2,HistCombo;
    Mat b_hist, g_hist, r_hist;
    string probeID, testID; //testID is the certain one image's ID
    int binCountX = 5; //# of small bins in each row of image
    int binCountY = 8; //# of small bins in each col of image
    double a, b, c;
    float SimScore;
    
    int width, height, binWidth, binHeight,binLocx,binLocy;
    
    /// Establish the number of bins
    int histSize = 256;
    
    /// Set the ranges ( for B,G,R) )
    float range[2] = {0, 256};
    const float* histRange = { range };
    
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);
    
    vector<string> file_list;
    vector<Mat> bgr_planes;
    vector <Mat> descriptor; // a vector of computed image hist
    Mat histImage;
    
    Mat ImageHistCompute(Mat );
    
  //  DescriptorY(): histImage(hist_h, hist_w, CV_8UC3, Scalar(0,0,0))
 //   {};

    virtual double distance(const Mat &probeImage, const Mat &galleryImage)
    {
        Mat desc1, desc2;
        desc1 = ImageHistCompute(probeImage);
        desc2 = ImageHistCompute(galleryImage);
        return bhattacharyya(desc1, desc2);
    }
};




#endif