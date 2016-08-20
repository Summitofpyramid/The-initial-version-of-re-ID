#ifndef _DESCRIPTOR_H
#define _DESCRIPTOR_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <sstream>
#include <vector>
#include <limits>
#include <fstream>
#include <iomanip>
#include <iostream>


using namespace std;
using namespace cv;

/*
 *  Base class to describe a descriptor
 */
class Descriptor{
public:
    virtual double distance(const Mat &probeImage, const Mat &galleryImage) = 0;
    virtual ~Descriptor(){}
    double bhattacharyya(cv::Mat k, cv::Mat q);
};


/*
 *  This is an example of a Descriptor class for you to implement
 *  the only method to implement is the distance method. 
 *  The method receives two images in Opencv MAT format
 *  and returns a similarity score between the two images.
 *  The similarity score is higher the closer to 0.
 *  i.e., a similarity score of 0 means that both images are the same
 */
class Test : public Descriptor {
    virtual double distance(const Mat &probeImage, const Mat &galleryImage)
    {
        return 1.32343;
    }
};



/*
 *  This class create the Similarity Table between 
 *  a set of n probes and m gallery items.
 *  The output is a matrix of double values with m rows and n columns saved in a .csv file
 *  The class assume that the correct class for probe i is at the
 *  same index i in the gallery vector.
 *  e.g.: 
 *      probe[i] and gallery[i] with i < n
 *      should be the correct match between the probe and the gallery
 *      Note that if you have more items in the gallery that in the probe vector
 *      they should appear after all the correct matches are listed
 *
 *      probe (vector)              gallery (vector)
 *      "01.jpg"                    "01b.jpg"
 *      "02.jpg"                    "02b.jpg"
 *      "03.jpg"                    "03b.jpg"
 *      ...                         ...
 *      "0n.jpg'                    "0nb.jpg"
 *                                  "0n+1b.jpg"
 *                                  ...
 *                                  "0mb.jpg"
 *
 */

class SimilarityTable{
    
private:
    Mat _similarityTable;
    string _folder;
    vector< string> _probes;
    vector< string> _gallery;
    Ptr<Descriptor> _descriptor;
    
public:
    SimilarityTable(const Ptr<Descriptor> &desc,
                    const string &folder,
                    const vector<string> &probes,
                    const vector<string> &gallery);
    
    void createTable();
    void outputFile(const string &filename);
    
    
    
};



class TextInput{
public:
    static void readLines(const string& filename, vector<string> &lines);
};
#endif

