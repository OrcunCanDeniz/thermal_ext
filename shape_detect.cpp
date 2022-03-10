
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;
int main(int argc, char** argv)
{
	Mat src=imread("cropped_new.jpg",0);
    // cvtColor(src,src,CV_BGR2GRAY);
    cv::adaptiveThreshold(src,src,255,cv::ADAPTIVE_THRESH_GAUSSIAN_C,cv::THRESH_BINARY,15,0);
	std::cout<<"ASDASDASDSAD"<<std::endl;
    vector< vector <Point> > contours; // Vector for storing contour
    vector< Vec4i > hierarchy;

    findContours( src, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
	std::cout<<contours.size()<<std::endl;
    for( int i = 0; i< contours.size(); i=hierarchy[i][0] ) // iterate through each contour.
      {
        Rect r= boundingRect(contours[i]);
        if(hierarchy[i][2]<0) //Check if there is a child contour
          rectangle(src,Point(r.x,r.y), Point(r.x+r.width,r.y+r.height), Scalar(0,255,0),2,8,0);
        else
          rectangle(src,Point(r.x,r.y), Point(r.x+r.width,r.y+r.height), Scalar(0,0,255),2,8,0);

      }
    imshow("src",src);
    waitKey();
}