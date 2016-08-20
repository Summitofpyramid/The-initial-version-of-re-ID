#ifndef _REIDDESCRIPTOR_H
#define _REIDDESCRIPTOR_H

#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/core/core.hpp>
#include<iostream>
#include<algorithm>
#include<math.h>
#include<limits>

#include "descriptor.h"


using namespace std;
using namespace cv;



class ReID : public Descriptor {
    
private:
    
    Mat HSVVector(Mat img);
    cv::Mat getWhsvFeature(cv::Mat img);
    
public:
    //-----------------------------
    virtual double distance(const Mat &probeImage, const Mat &galleryImage)
    {
        Mat desc1 = getWhsvFeature(probeImage);
        Mat desc2 = getWhsvFeature(galleryImage);
        return bhattacharyya(desc1, desc2);
    }
};


class ReidDescriptor{
private:
	cv::Mat Whsv;
public:
	ReidDescriptor();
	ReidDescriptor(cv::Mat person);
	float compare(ReidDescriptor reid);
	//Get the feature of the person from the image.
	static cv::Mat getWhsvFeature(cv::Mat img,cv::Mat MSK=cv::Mat());
	//Compare the similarity between two original image of people, return the score(0-1,0 means the most similar)
	static float compareImg(cv::Mat img1, cv::Mat img2);
	//Compare the similarity between two ReidDescirptor of people, return the score(0-1,0 means the most similar)
	static float compareDiscriptor(ReidDescriptor f1,ReidDescriptor f2);
	static float bhattacharyya(cv::Mat k, cv::Mat q);
private:

	static int fminbnd(float(*func)(int, cv::Mat, cv::Mat, int, float), cv::Mat img, cv::Mat MSK, int loc, float alpha, int range1, int range2);
	static float dissym_div(int x, cv::Mat img, cv::Mat MSK, int loc, float alpha);
	static float sym_div(int x, cv::Mat img, cv::Mat MSK, int loc, float alpha);
	static float sym_dissimilar_MSKH(int x, cv::Mat img, cv::Mat MSK, int loc, float alpha);
	static cv::Mat gau_kernel(int x, float varW, int H, int W);
	static cv::Mat padarray(cv::Mat m, int num);
	static float max_value(cv::Mat m);
	static cv::Mat map_kernel(cv::Mat img, int delta1, int delta2, float alpha, float varW, int H, int W, int &TLanti, int &HDanti,cv::Mat MSK=cv::Mat());
	static cv::Mat whistcY(cv::Mat raster, cv::Mat Weight, int bins);
	static cv::Mat Whsv_estimation(cv::Mat img, vector<int> NBINs, cv::Mat MAP_KRNL, int HDanti, int TLanti);
	static float onebyone(cv::Mat img1, cv::Mat img2);
	static void drawHist(string str,cv::Mat hist);
    
    
    virtual double distance(const Mat &probeImage, const Mat &galleryImage)
    {
        return onebyone(probeImage, galleryImage);
    }
    
};

#endif
