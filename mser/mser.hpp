#pragma once

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/features2d/features2d.hpp"
#include <iostream>
#include <cmath> 


// void pointSelector(cv::Mat src_img);

// static void CallBackPoints(int event, int x, int y, int flags, void* img);

static void CallBackROI(int event, int x, int y, int flags, void* img);

bool doesBoxOverlapCircle(cv::Rect circle, cv::Rect rect);

int tag_squares(cv::Rect thermal_blob, cv::Point image_center);

std::vector< std::vector<float>> convert2xyxy(std::vector<cv::Rect> boxes);

cv::Point calc_centroid(cv::Rect box);

float calc_dist_to_center(cv::Point cluster_centroids, cv::Point image_center);

std::vector<cv::Rect> get_circles(cv::Mat &src);

bool region_filter(cv::Rect mser_box, cv::Rect filter_rect);

cv::Rect roiSelector(cv::Mat src_img);

std::vector<cv::Rect> detectSquares(cv::Mat gray, cv::Rect roi, cv::Ptr<cv::MSER> ms, cv::Rect filter_rect);

// std::vector<cv::Point2f> getCorners(std::vector<cv::Rect> detected_boxes);

std::vector<std::vector<cv::Point2f>> getCorners(std::vector<cv::Rect> detected_boxes);

bool key_point_region_filter(cv::Point2f mser_box, cv::Rect filter_rect);
