#include "mser.hpp"
#include "../nms/utils.hpp"
#include "../nms/nms.hpp"

bool bad_detection_flag;

// std::vector<cv::Point> clickedCorners;

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

static void CallBackROI(int event, int x, int y, int flags, void* img) {
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


// static void CallBackPoints(int event, int x, int y, int flags, void* img)
// {
//     if (event == cv::EVENT_LBUTTONDOWN)
//     {
//         clickedCorners.push_back(cv::Point(x,y));
//         // cv::circle( img, cv::Point(x,y), 1, cv::Scalar(45,123,77), 3, cv::LINE_AA);
//     }

//     return;
// }


bool doesBoxOverlapCircle(cv::Rect circle, cv::Rect rect)
{
    int cx=0, cy=0;
    cx = (int)round(circle.x+(float)(circle.width/2));
    cy = (int)round(circle.y+(float)(circle.height/2));
    if( (cx<rect.x+rect.width) && (cy<rect.y+rect.height) && (cx>rect.x) && (cy>rect.y) )
    {
        return true;
    } else {
        return false;
    }
}

int tag_squares(cv::Rect thermal_blob, cv::Point image_center)
{
    if(thermal_blob.x < image_center.x && thermal_blob.y < image_center.y)
    {
        return 0;
    } else if (thermal_blob.x > image_center.x && thermal_blob.y < image_center.y)
    {
        return 1;
    } else if(thermal_blob.x > image_center.x && thermal_blob.y > image_center.y)
    {
        return 2;
    } else if(thermal_blob.x < image_center.x && thermal_blob.y > image_center.y)
    {
        return 3;
    }
}

std::vector< std::vector<float>> convert2xyxy(std::vector<cv::Rect> boxes)
{
    std::vector<std::vector<float>> ret;

    for (int i=0; i<boxes.size(); i++)
    {
        ret.push_back({(float)boxes[i].x, (float)boxes[i].y, (float)boxes[i].x+boxes[i].width, (float)boxes[i].y+boxes[i].height});
    }
    return ret;
}

cv::Point calc_centroid(cv::Rect box)
{
    int cx=0, cy=0; 
    cx = (box.x + box.width/2);
    cy = (box.y + box.height/2);

    return cv::Point(cx,cy);
}

float calc_dist_to_center(cv::Point cluster_centroids, cv::Point image_center)
{

    float dist = sqrt(pow((image_center.x - cluster_centroids.x), 2.0) + pow((image_center.y - cluster_centroids.y), 2.0 ));

    return dist;
}

std::vector<cv::Rect> get_circles(cv::Mat &src)
{ 
    cv::Mat gray;
    medianBlur(src, gray, 5);
    std::vector<cv::Vec3f> circles;
    HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1,
                 gray.rows/16,  // change this value to detect circles with different distances to each other
                 100, 30, 1, 30 // change the last two parameters
            // (min_radius & max_radius) to detect larger circles
    );
    std::vector<cv::Rect> ret;
    for (int i=0; i<circles.size(); i++)
    {
        cv::Rect tmp;
        tmp.x = circles[i][0] - circles[i][2];
        tmp.y = circles[i][1] - circles[i][2];
        tmp.height = circles[i][2] *2;
        tmp.width = tmp.height; 
        ret.push_back(tmp);

        cv::Vec3i c = circles[i];
        cv::Point center = cv::Point(c[0], c[1]);
        // circle center
        circle( src, center, 1, cv::Scalar(0,0,0), 3, cv::LINE_AA);
        // circle outline
        int radius = c[2];
        // circle( gray, center, radius, Scalar(0,0,0), 3, LINE_AA);

    }
    
    return ret;
}

bool region_filter(cv::Rect mser_box, cv::Rect filter_rect)
{   
    bool ret;
    float mser_x = mser_box.x + mser_box.width/2;
    float mser_y = mser_box.y + mser_box.height/2;
    // cout<<filter_rect<<endl;
    if(mser_x > filter_rect.x && mser_y > filter_rect.y)
    { 
        if ((mser_x < (filter_rect.x + filter_rect.width)) && (mser_y < (filter_rect.y + filter_rect.height)))
        {
        ret = true;
        }
    } else {
        ret = false;
    }

    // cout<< "Region filter test point: " << mser_x << " , " << mser_y << " RET: " << ret << endl;
    return ret;
}

bool key_point_region_filter(cv::Point2f mser_box, cv::Rect filter_rect)
{   
    bool ret;
    float mser_x = mser_box.x;
    float mser_y = mser_box.y;
    // cout<<filter_rect<<endl;
    if(mser_x > filter_rect.x && mser_y > filter_rect.y)
    { 
        if ((mser_x < (filter_rect.x + filter_rect.width)) && (mser_y < (filter_rect.y + filter_rect.height)))
        {
        ret = true;
        }
    } else {
        ret = false;
    }

    // cout<< "Region filter test point: " << mser_x << " , " << mser_y << " RET: " << ret << endl;
    return ret;
}

// void pointSelector(cv::Mat src_img)
// {
//     for(auto const &pt: clickedCorners)
//     {
//         circle( src_img, cv::Point(pt.x, pt.y), 1, cv::Scalar(200,150,56), 3, cv::LINE_AA);
//     }

//     return;
// }

cv::Rect roiSelector(cv::Mat src_img)
{
    cv::Mat roiSelectionFrame(src_img.size(),src_img.type());
    src_img.copyTo(roiSelectionFrame);
    while (true)
    {
      cv::imshow("Select ROI", roiSelectionFrame);
      cv::putText(roiSelectionFrame, "Click and drag to mark calib. board ROI.",cv::Point(10,25),cv::FONT_HERSHEY_DUPLEX,0.5,cv::Scalar(0,255,0),1,false);
      cv::putText(roiSelectionFrame, " 'r' to reset ROI, 'c' to continue.",cv::Point(10,50),cv::FONT_HERSHEY_DUPLEX,0.5,cv::Scalar(0,255,0),1,false);
      cv::setMouseCallback("Select ROI", CallBackROI, 0);
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
        src_img.copyTo(roiSelectionFrame); // Reset roi
      }
    }

    cv::destroyWindow("Select ROI");

    cv::Rect roi = cv::Rect(SelectedRoi.initX, SelectedRoi.initY, 
                      SelectedRoi.actualX - SelectedRoi.initX,  SelectedRoi.actualY - SelectedRoi.initY);
    return roi;
}

std::vector<cv::Rect> detectSquares(cv::Mat gray, cv::Rect roi, cv::Ptr<cv::MSER> ms, cv::Rect filter_rect)
{
    std::vector<cv::Rect> thermal_blobs(4);
    std::vector<cv::Rect> final_thermal_blobs(4);

    std::vector<std::vector<cv::Point> > regions;
    std::vector<std::vector<cv::Point> > filtered_regions;

    float roi_area = roi.area();
    // equalizeHist(gray, gray);
    imshow("Source", gray);
    cv::Mat img = gray(roi);

    std::vector<cv::Rect> mser_bbox;
    std::vector<cv::Rect> filtered_boxes;

    ms->detectRegions(img, regions, mser_bbox);
    std::vector<cv::Rect> circles_boxes = get_circles(img);
    
    for (int i = 0; i < regions.size(); i++)
    {
        float aspect_ratio = mser_bbox[i].width/(float)mser_bbox[i].height;
        if (aspect_ratio < 0.97 || aspect_ratio > 1.03)
        {
            continue;
        }

        float board2box_ratio = mser_bbox[i].area()/(float)roi_area;
        if (board2box_ratio > (0.028*1.1) || board2box_ratio < (0.028*0.9))
        {
            continue;
        }

        filtered_boxes.push_back(mser_bbox[i]);
        filtered_regions.push_back(regions[i]);
    }

    if (filtered_boxes.size() == 0)
    {
    std::cout<<"NO BOXES DETECTED !! "<<std::endl;
    }

    for (int i = 0; i < filtered_boxes.size(); i++)
    {
        rectangle(img, filtered_boxes[i], CV_RGB(0, 255, 0));  
    }

    rectangle(img, filter_rect, cv::Scalar(255,255,255), 0.3);

    imshow("Boxes", img);

    std::vector<std::vector<float>> boxes_xyxy = convert2xyxy(filtered_boxes);

    // cv::Mat drawing = cv::Mat::zeros( img.size(), CV_8UC3 );

    std::vector<cv::Rect> reducedRectangle = nms(boxes_xyxy, 0.7);

    std::vector< std::pair<float, int> > distances_vec;

    for( int i=0; i<reducedRectangle.size(); i++)
    {
        cv::Point center = calc_centroid(reducedRectangle[i]);
        float dist = calc_dist_to_center(center, cv::Point(img.cols/2,img.rows/2));
        distances_vec.push_back(std::make_pair(dist,i));
    }

    sort(distances_vec.begin(), distances_vec.end());

    for (int i=(distances_vec.size()-4); i<distances_vec.size(); i++)
    {
        int idx = distances_vec[i].second;
        int tag = tag_squares(reducedRectangle[idx], cv::Point(img.cols/2,img.rows/2));
        thermal_blobs[tag] = reducedRectangle[idx];
    }
    
    bad_detection_flag = false;

    for (int a=0; a<thermal_blobs.size(); a++)
    {
        cv::Scalar color = (207,86,234);
        if(thermal_blobs[a].area() == 0)
        {
            continue;
        }

        bool is_overlapping;
        for (int c = 0; c<circles_boxes.size() ; c++)
        {
            is_overlapping = doesBoxOverlapCircle(circles_boxes[c], thermal_blobs[a]); // change check method to if circle center lays in the square
            if(is_overlapping)
            {   
                break;
            }
        }

        if(is_overlapping)
        {
            continue;
        }

        if(region_filter(thermal_blobs[a], filter_rect))
        {
            color = cv::Scalar(255,255,0);
            continue;
        }
        thermal_blobs[a].x += roi.x;
        thermal_blobs[a].y += roi.y;
        final_thermal_blobs[a] = (thermal_blobs[a]);
    }    
    
    for(int i=0; i<circles_boxes.size(); i++) // omittable
    {
        circles_boxes[i].x += roi.x;
        circles_boxes[i].y += roi.y;
    }

    return final_thermal_blobs;
}

// std::vector<cv::Point2f> getCorners(std::vector<cv::Rect> detected_boxes)
// {
//     std::vector<cv::Point2f> ret;

//     for(auto & box : detected_boxes)
//     {

//         if(box.area() == 0)
//         {
//             continue;
//         }
//         // std::vector<cv::Point2f> tmp;
//         ret.push_back(cv::Point2f(box.x, box.y));
//         ret.push_back(cv::Point2f(box.x+box.width, box.y));
//         ret.push_back(cv::Point2f(box.x, box.y + box.height));
//         ret.push_back(cv::Point2f(box.x+box.width, box.y+box.height));

//         // ret.push_back(tl,tr,br,bl);
//     }

//     return ret;
// }

std::vector<std::vector<cv::Point2f>> getCorners(std::vector<cv::Rect> detected_boxes)
{
    std::vector<std::vector<cv::Point2f>> ret;

    for(auto & box : detected_boxes)
    {

        if(box.area() == 0)
        {
            continue;
        }
        // std::vector<cv::Point2f> tmp;
        cv::Point2f tl(box.x, box.y);
        cv::Point2f tr(box.x+box.width, box.y);
        cv::Point2f bl(box.x, box.y + box.height);
        cv::Point2f br(box.x+box.width, box.y+box.height);

        ret.push_back({tl,tr,br,bl});

    }

    return ret;

}