#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "nms/utils.hpp"
#include "nms/nms.hpp"
#include <iostream>
#include <cmath> 

bool bad_detection_flag;

using namespace cv;
using namespace std;

int main(int argc, char const *argv[])
{
    Rect r1(5,5,5,5);
    Rect r2(5,5,5,5);

    Rect union_box = r1 | r2;
    Rect intersection_box = r1 & r2;

    float iou = intersection_box.area() / (float)union_box.area();

    cout<<iou<<endl;
    return 0;
}
