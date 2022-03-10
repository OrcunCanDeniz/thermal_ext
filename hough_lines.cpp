#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <algorithm>

using namespace cv;
using namespace std;

RNG rng(12345);
Mat src, dst;
int thresh = 100;
vector<vector<Point> > contours;
vector<Vec4i> hierarchy;
std::vector<cv::Vec3f> circles;


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

bool less_by_x(const cv::Point& lhs, const cv::Point& rhs)
{
  return lhs.x < rhs.x;
}

bool less_by_y(const cv::Point& lhs, const cv::Point& rhs)
{
  return lhs.y < rhs.y;
}

static void CallBackF(int event, int x, int y, int flags, void* img) 
{
//Mouse Right button down
  if (event == cv::EVENT_RBUTTONDOWN) 
  {
    std::cout << "right button " << std::endl;
    return;
  }
//Mouse Left button down
  if (event == cv::EVENT_LBUTTONDOWN)
  {
    SelectedRoi.initX = x;
    SelectedRoi.initY = y;
    SelectedRoi.init = true;
    std::cout << "left button DOWN, x: "<< x << "y: "<< y << std::endl; 
    return;
  }
//Mouse Left button up
  if (event == cv::EVENT_LBUTTONUP) 
  {
    SelectedRoi.actualX = x;
    SelectedRoi.actualY = y;
    SelectedRoi.end = true;
    std::cout << "left button UP, x: "<< x << "y: "<< y << std::endl;
    return;
  }
}


static double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

int eval_cont(vector<cv::Point> &approx, std::vector<cv::Point>& contour)
{        
        if (approx.size() >= 4 && approx.size() <= 6)
		{
			// Number of vertices of polygonal curve
			int vtc = approx.size();

			// Get the cosines of all corners
			std::vector<double> cos;
			for (int j = 2; j < vtc+1; j++)
				cos.push_back(angle(approx[j%vtc], approx[j-2], approx[j-1]));

			// Sort ascending the cosine values
			std::sort(cos.begin(), cos.end());

			// Get the lowest and the highest cosine
			double mincos = cos.front();
			double maxcos = cos.back();

			// Use the degrees obtained above and the number of vertices
			// to determine the shape of the contour
			if (vtc == 4 && mincos >= -0.1 && maxcos <= 0.3)
				return 0;
		}
		else
		{
			// Detect and label circles
			double area = cv::contourArea(contour);
			cv::Rect r = cv::boundingRect(contour);
			int radius = r.width / 2;

			if (std::abs(1 - ((double)r.width / r.height)) <= 0.2 &&
			    std::abs(1 - (area / (CV_PI * std::pow(radius, 2)))) <= 0.2)
				return 1;
		}
}

void thresh_callback(int, void* )
{
    // Mat canny_output;
    Canny( src, dst, thresh, thresh*2 );
    for( size_t i = 0; i < circles.size(); i++ ) // draw over canny output
    {
        cv::Vec3i c = circles[i];
        cv::Point center = cv::Point(c[0], c[1]);
        // circle center
        // cv::circle(dst, center, -1, cv::Scalar(0,0,0), 3, cv::LINE_AA);
        // circle outline
        float radius = c[2] * 1.15;
        int rounded_radius = std::round(radius*1.15);
        cv::circle( dst, center, rounded_radius, cv::Scalar(0,0,0), -1, cv::LINE_AA);
    }
    findContours( dst, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE );
    Mat drawing_cb = Mat::zeros( dst.size(), CV_8UC3 );
    std::cout<< "Num of conts; " << contours.size() <<std::endl; 
    for( size_t i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
        drawContours( drawing_cb, contours, (int)i, color, 0.5, LINE_8, hierarchy, 0 );
    }
    imshow( "Contours", drawing_cb );
}

int main(int argc, char** argv)
{
    // Declare the output variables
    Mat dst, gray, out;
    // Loads an image
    Mat raw_frame = imread("./data/new_board/frames/frame0000.jpg", 0);
    if(raw_frame.empty()){
        printf(" Error opening image\n");
        return -1;
    }
    // src = imread("cropped_new.jpg", 0);
    // Check if image is loaded fine
    // if(src.empty()){
    //     printf(" Error opening image\n");
    //     return -1;
    // }
    // cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    cv::Mat roiSelectionFrame(raw_frame.size(),raw_frame.type());
    raw_frame.copyTo(roiSelectionFrame);
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
        raw_frame.copyTo(roiSelectionFrame); // Reset roi
      }
    }
    destroyWindow("Select ROI");

    src = raw_frame(cv::Rect(SelectedRoi.initX, SelectedRoi.initY, 
                      SelectedRoi.actualX - SelectedRoi.initX,  SelectedRoi.actualY - SelectedRoi.initY));

    
    cv::HoughCircles(src, circles, cv::HOUGH_GRADIENT, 1,
                 src.rows/64,  // change this value to detect circles with different distances to each other
                 100, 30, 1, 50); // change the last two parameters
            // (min_radius & max_radius) to detect larger circles


    namedWindow("cropped canny");
    resizeWindow("cropped canny", raw_frame.cols, raw_frame.rows);
    imshow( "cropped canny", src);
    const int max_thresh = 255;
    createTrackbar( "Canny thresh:", "cropped canny", &thresh, max_thresh, thresh_callback );
    thresh_callback( 0, 0 );
    waitKey();

    // Edge detection
    // Canny(src, dst, 50, 200, 3);
    // Copy edges to the images that will display the results in BGR

    // cv::medianBlur(src, gray, 3);
    Mat drawing = Mat::zeros( src.size(), CV_8UC3 ); 
    
    // vector<vector<Point> > contours;
    // vector<vector<Point> > contours_filtered;
    // vector<Vec4i> hierarchy;
    // findContours( dst, contours, hierarchy, RETR_EXTERNAL , CHAIN_APPROX_SIMPLE );

    // cout<< "contours dim 1: " << contours.size() << "contours dim 2: " << contours[0].size() << endl;



    for( size_t i = 0; i< contours.size(); i++ )
    {
        
        RotatedRect min_rect = minAreaRect(contours[i]);
        float aspect_ratio = min_rect.size.width / (float)min_rect.size.height;

        if (aspect_ratio >= 1.2 || aspect_ratio <= 0.8)
        {
            contours.erase(contours.begin() + i);
            continue;
        } 

        // calc centroid of contour

        vector<cv::Point> approx;
        cv::approxPolyDP(cv::Mat(contours[i]), approx, cv::arcLength(cv::Mat(contours[i]), true)*0.02, true);
        
        int ret;
        if (!cv::isContourConvex(approx))
        {
            contours.erase(contours.begin() + i);
            continue;
        } else {
            ret = eval_cont(approx, contours[i]); // ret = 1 if cont is circle 0 if rect
            // cout<<"ret"<<endl;
        }

        if (ret == 1)
        {   
            contours.erase(contours.begin() + i);
            continue;
        }

        // drawing(Rect((int)round(min_rect.center.x), (int)round(min_rect.center.y), min_rect.size.y, min_rect.center.y));
        // cout<<min_rect.size.height<< " " << min_rect.size.width << " AR: "<< aspect_ratio << endl;
        // Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
        // cout<<contours.size()<<endl;
        // fillPoly( drawing, contours_filtered, Scalar(255,0,255));
        // imshow( "Contours", drawing );
    }

    cout<< "FILTERED CONTOURS NUM: " << contours.size() << endl;
    // vector<vector<Point>> regions; //vector to hold points belonging to 4 different regions 
    vector<vector<Point>> region_corners;
    Mat filtered_drawing = Mat::zeros( src.size(), CV_8UC3 );
    for (int i=0; i<contours.size(); i++)
    {
        drawContours( filtered_drawing, contours, i, Scalar(0,0,255), 0.7, LINE_8, 0 );
    }

    imshow("filtered conts", filtered_drawing);

    for (int v=0; v<4; v++)
    {
        region_corners.push_back(vector<Point>());
    }


    for (int i = 0; i<contours.size(); i++)
    {
        vector<vector<Point>> regions;
        for (int v=0; v<4; v++)
        {
            regions.push_back(vector<Point>());
        }

        Moments mu = moments(contours[i]);
        int cX = round(mu.m10 / (mu.m00 + 1e-5));
        int cY = round(mu.m01 / (mu.m00 + 1e-5));
        circle( drawing, Point(cX, cY), 2,Scalar( 0, 0, 255 ),FILLED,LINE_8 );

        for (int pt=0; pt<contours[i].size(); pt++)
        {
            if (contours[i][pt].x < cX && contours[i][pt].y < cY) // region 1
            {    
                regions[0].push_back(contours[i][pt]);
            } else if (contours[i][pt].x > cX && contours[i][pt].y < cY) { // region 2
                regions[1].push_back(contours[i][pt]);
            } else if (contours[i][pt].x > cX && contours[i][pt].y > cY) { //region 3
                regions[2].push_back(contours[i][pt]);
            } else if (contours[i][pt].x < cX && contours[i][pt].y > cY) { // region 4
                regions[3].push_back(contours[i][pt]);
            }
        }       

        for (int reg_idx=0 ; reg_idx<4 ; reg_idx++ )
        {   
            auto mmx = minmax_element(regions[reg_idx].begin(), regions[reg_idx].end(), less_by_x);
            auto mmy = minmax_element(regions[reg_idx].begin(), regions[reg_idx].end(), less_by_y);
            //mm s .first will have an iterator to the minimum element, and mm s .second to the maximum one.
            Point corner(15000,15000);

            switch (reg_idx){
                case 0:
                    corner.x =(*mmx.first).x;
                    corner.y =(*mmy.first).y;
                    break;
                case 1:
                    corner.x =(*mmx.second).x;
                    corner.y =(*mmy.first).y;
                    break;
                case 2:
                    corner.x =(*mmx.second).x; 
                    corner.y =(*mmy.second).y;
                    break;
                case 3:
                    corner.x = (*mmx.first).x; 
                    corner.y = (*mmy.second).y;
                    break;
            }
            if (corner.x != 15000 && corner.y != 15000)
            {
                region_corners[i].push_back(corner);
            }
        }
    }

    cout<<"OUTING"<<endl;
    cvtColor(src, out, COLOR_GRAY2BGR);
    
    for (int reg_idx=0 ; reg_idx<4 ; reg_idx++ )
    { 
        for (int cont_pt=0 ; cont_pt<region_corners[reg_idx].size(); cont_pt++)
        {
            circle(out, region_corners[reg_idx][cont_pt], 2, Scalar(255,0,255), -1);
        imshow( "Contours", out );
        }
        // drawContours( drawing, contours, a, Scalar(0,0,255), 2, LINE_8, hierarchy, 0 );
    }
    waitKey();
    
    return 0;
}