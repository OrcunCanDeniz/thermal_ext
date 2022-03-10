
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;
Mat src_gray;
int thresh = 100;
RNG rng(12345);


struct initRoi {
  // on off
  bool init;
  bool end;

  //initial coordination based on EVENT_LBUTTONDOWN
  int initX;
  int initY;

  // actual coordination 
  int actualX;
  int actualY;

  //Selected Rect
  cv::Rect roiRect; 

  //Selected Mat roi
  cv::Mat takenRoi;
}SelectedRoi;

static void CallBackF(int event, int x, int y, int flags, void* img) {
//Mouse Right button down
  if (event == cv::EVENT_RBUTTONDOWN) {
    std::cout << "right button " << std::endl;
    return;
  }
//Mouse Left button down
  if (event == cv::EVENT_LBUTTONDOWN) {
    SelectedRoi.initX = x;
    SelectedRoi.initY = y;
    SelectedRoi.init = true;
    std::cout << "left button DOWN, x: "<< x << "y: "<< y << std::endl; 
    return;
  }
//Mouse Left button up
  if (event == cv::EVENT_LBUTTONUP) {
    SelectedRoi.actualX = x;
    SelectedRoi.actualY = y;
    SelectedRoi.end = true;
    std::cout << "left button UP, x: "<< x << "y: "<< y << std::endl;
    return;
  }
}


void thresh_callback(int, void* );
int main( int argc, char** argv )
{
    CommandLineParser parser( argc, argv, "{@input | HappyFish.jpg | input image}" );
    Mat src = imread( "./data/new_board/frames/frame0000.jpg") ;
    if( src.empty() )
    {
      cout << "Could not open or find the image!\n" << endl;
      cout << "Usage: " << argv[0] << " <Input image>" << endl;
      return -1;
    }
    Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::Mat roiSelectionFrame(gray.size(),gray.type());
    gray.copyTo(roiSelectionFrame);
    while (true)
    {
      cv::imshow("Select ROI", roiSelectionFrame);
      cv::putText(roiSelectionFrame, "Click and drag to mark calib. board ROI. ESC to continue after selection.",cv::Point(10,25),cv::FONT_HERSHEY_DUPLEX,0.5,cv::Scalar(0,255,0),1,false);
      cv::putText(roiSelectionFrame, " 'r' to reset ROI, 'c' to continue.",cv::Point(10,50),cv::FONT_HERSHEY_DUPLEX,0.5,cv::Scalar(0,255,0),1,false);
      cv::setMouseCallback("Select ROI", CallBackF, 0);
      if (SelectedRoi.init && SelectedRoi.end)
      { 
          cv::rectangle(roiSelectionFrame, cv::Rect(SelectedRoi.initX, SelectedRoi.initY, 
                      SelectedRoi.actualX - SelectedRoi.initX,  SelectedRoi.actualY - SelectedRoi.initY), 
                      cv::Scalar(0, 255, 255), 1);
          SelectedRoi.init = false;
          SelectedRoi.end = false;
      }
      char key_pressed = (char)(cv::waitKey(1));
      if(key_pressed == 'c')  //complete roi selection
      {
        break;
      } else if (key_pressed == 'r') {
        gray.copyTo(roiSelectionFrame); // Reset roi
      }
    }

    gray = gray(cv::Rect(SelectedRoi.initX, SelectedRoi.initY, 
                      SelectedRoi.actualX - SelectedRoi.initX,  SelectedRoi.actualY - SelectedRoi.initY));


    destroyWindow("Select ROI");


    gray.copyTo(src_gray);
    // blur( src_gray, src_gray, Size(3,3) );
    const char* source_window = "Source";
    namedWindow( source_window );
    imshow( source_window, src );
    const int max_thresh = 255;
    createTrackbar( "Canny thresh:", source_window, &thresh, max_thresh, thresh_callback );
    thresh_callback( 0, 0 );
    waitKey();
    return 0;
}
void thresh_callback(int, void* )
{
    Mat canny_output;
    Canny( src_gray, canny_output, thresh, thresh*2 );
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours( canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE );
    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
    std::cout<< "Num of conts; " << contours.size() <<std::endl; 
    for( size_t i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
        drawContours( drawing, contours, (int)i, color, 0.5, LINE_8, hierarchy, 0 );
    }
    imshow( "Contours", drawing );
}