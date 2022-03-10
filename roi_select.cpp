#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

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
//Mouse move coordinates update
//   if (event == cv::EVENT_MOUSEMOVE) {
  
//     std::cout<< "event mouse move"<< std::endl; 
//     SelectedRoi.actualX = x;
//     SelectedRoi.actualY = y;
//     SelectedRoi.roiRect = cv::Rect(SelectedRoi.initX, SelectedRoi.initY,
//     SelectedRoi.actualX,  SelectedRoi.actualY);
//     return;
//   }
}


int main(int argc, char** argv)
{
    cv::Mat src = cv::imread( "./data/new_board/frames/frame0004.jpg", cv::IMREAD_COLOR );
    // Check if image is loaded fine
    if(src.empty()){
        printf(" Error opening image\n");
        printf(" Program Arguments: [image_name -- default]");
        return EXIT_FAILURE;
    }
    cv::Mat gray;
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

    cv::imwrite("cropped_new.jpg", gray);
    // cv::adaptiveThreshold(gray,gray,255,cv::ADAPTIVE_THRESH_GAUSSIAN_C,cv::THRESH_BINARY,13,0);
    // cv::imshow("threshold", gray);
    // cv::waitKey(0);
    // cv::medianBlur(gray, gray, 5);
    // std::vector<cv::Vec3f> circles;
    // cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1,
    //              gray.rows/16,  // change this value to detect circles with different distances to each other
    //              100, 30, 1, 50 // change the last two parameters
    //         // (min_radius & max_radius) to detect larger circles
    // );
    // for( size_t i = 0; i < circles.size(); i++ )
    // {
    //     cv::Vec3i c = circles[i];
    //     cv::Point center = cv::Point(c[0] + SelectedRoi.initX, c[1]+SelectedRoi.initY);
    //     // circle center
    //     cv::circle(src, center, 1, cv::Scalar(0,100,100), 3, cv::LINE_AA);
    //     // circle outline
    //     int radius = c[2];
    //     cv::circle( src, center, radius, cv::Scalar(255,0,255), 3, cv::LINE_AA);
    // }
    cv::imshow("detected circles", gray);
    cv::waitKey();
    return EXIT_SUCCESS;
}