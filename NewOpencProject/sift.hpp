//
//  sift.hpp
//  NewOpencProject
//
//  Created by Johnson Johnson on 2016-09-08.
//  Copyright © 2016 Johnson Johnson. All rights reserved.
//

#ifndef _SIFT_H
#define _SIFT_H

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "descriptor.h"
#include <opencv2/imgproc/imgproc.hpp>


class sift_: public Descriptor{
public:
    Mat dictionary;
    Mat compute_hist(Mat image){
        
        Mat BOWdescriptor; // the BOW hist of image
        vector<KeyPoint> keypoints;
        //read the dictionary
        FileStorage fs("dictionary.yml", FileStorage::READ);
        fs["vocabulary"] >> dictionary;
        fs.release();
        
        //create a nearest neighbor matcher
        Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
        //create Sift feature point extracter
        Ptr<FeatureDetector> detector(new SiftFeatureDetector());
        //create Sift descriptor extractor
        Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor);
        //create BoF (or BoW) descriptor extractor
        BOWImgDescriptorExtractor bowDE(extractor,matcher);
        //Set the dictionary with the vocabulary we created in the first step
        bowDE.setVocabulary(dictionary);
        detector->detect(image,keypoints);
        bowDE.compute(image, keypoints, BOWdescriptor);
        return BOWdescriptor;
    }
    
     virtual double distance(const Mat& gallery,const Mat& probe){
         Mat h1,h2;
         h1 = compute_hist(gallery);
         h2 = compute_hist(probe);
        return bhattacharyya(h1,h2);
    }
    
    
};

#endif /* sift_hpp */


