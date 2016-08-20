#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

int main(){
Mat img = imread("/Users/JohnsonJohnson/Documents/MJ/2.jpeg");

MSER mser(5,100,1500);
vector<vector<Point> > point;
mser(img,point);
Mat output(img.size(),CV_8UC3);
output = Scalar(255,255,255);
RNG rng;

for(vector<vector<Point> >::iterator it = point.begin();it!=point.end();it++)
{
Vec3b c(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));
    
for(vector<Point>::iterator itPTS= it->begin();itPTS!=it->end();++itPTS)
 {
	if(output.at<Vec3b>(*itPTS)[0]==255)
		output.at<Vec3b>(*itPTS)=c;
 }
}

    imshow("output",output);
    waitKey();

}
