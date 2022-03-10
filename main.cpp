#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "nms/utils.hpp"
#include "nms/nms.hpp"
#include <iostream>
#include <cmath> 
#include "mser/mser.hpp"

std::vector<cv::Point> clickedCorners;
int NUM_OF_POINTS = 16; // 4 markers * 4 corners

cv::Mat dst;

cv::Mat createSeeds(std::vector<cv::Rect> boxes, cv::Rect roi)
{
    cv::Mat seeds(roi.size(), CV_32SC1 );
    for(int i=0; i<boxes.size(); i++)
    {
        int new_width = (int)boxes[i].width/2;
        int new_height = (int)boxes[i].height/2; 
        int new_x = boxes[i].x + new_width;
        int new_y = boxes[i].y + new_height;

        cv::Rect markers(new_x, new_y, new_width, new_height);
        rectangle(seeds, markers, cv::Scalar::all(i),-1);
    }

    return seeds;
}

void pointSelector(cv::Mat& src_img)
{
    for(auto const &pt: clickedCorners)
    {
        circle( src_img, pt, 3, cv::Scalar(200,150,56), -1, cv::LINE_AA);
    }
    return;
}

std::vector<std::vector<cv::Point>> groupCorners(std::vector<cv::Point> points)
{
    std::vector<std::vector<cv::Point>> corners;

    corners = {{points[0],points[1],points[2],points[3]},
                {points[4],points[5],points[6],points[7]},
                {points[8],points[9],points[10],points[11]},
                {points[12],points[13],points[14],points[16]}};
    return corners;
}

int MAX_ITERATIONS = 5;
static void CallBackPoints(int event, int x, int y, int flags, void* img)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        if ( clickedCorners.size() < NUM_OF_POINTS)
        {
            clickedCorners.push_back(cv::Point(x,y));
        } else if (clickedCorners.size() > NUM_OF_POINTS) {
            std::cout<<"exceeded max num of points";
        }

        // cv::circle( img, cv::Point(x,y), 1, cv::Scalar(45,123,77), 3, cv::LINE_AA);
    }
    return;
}

using namespace cv;
using namespace std;

Mat applyKMeans(const Mat& source){

	const unsigned int singleLineSize = source.rows * source.cols;
	Mat data = source.reshape(1, singleLineSize);
	data.convertTo(data, CV_32F);
	std::vector<int> labels;
	cv::Mat1f colors;
	cv::kmeans(data, 2,labels,cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 10, 1.),MAX_ITERATIONS,cv::KMEANS_PP_CENTERS,colors);
	for (int i=0; i<labels.size(); i++)
    {
    std::cout<<labels[i]<<endl;
    }
    for (unsigned int i = 0 ; i < singleLineSize ; i++ ){
				data.at<float>(i, 0) = colors(labels[i], 0);
		        data.at<float>(i, 1) = colors(labels[i], 1);
		        data.at<float>(i, 2) = colors(labels[i], 2);
	}

	Mat outputImage = data.reshape(3, source.rows);
	outputImage.convertTo(outputImage, CV_8U);
	return outputImage;
}

// struct initRoi {
//   // on off
//   bool init;
//   bool end;

//   //initial coordination based on EVENT_LBUTTONDOWN
//   int initX;
//   int initY;

//   // actual coordination 
//   int actualX;
//   int actualY;

//   //Selected Rect
//   cv::Rect roiRect; 

//   //Selected Mat roi
//   cv::Mat takenRoi;
// }SelectedRoi;


int main(int argc, char *argv[])
{
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    Size subPixWinSize(7,7), winSize(22,22);
    const int MAX_COUNT = 500;
    VideoCapture cap("/home/orcun/thermal_ext_calib/data/new_board/thermal_ext_board.avi",0);
    

    VideoWriter video("outcpp.avi",CV_FOURCC('M','J','P','G'), 10, Size(640,512));

    cv::Rect filter_rect;
    float rHeight=0.7, rWidth=0.56;

    Mat gray, src, grayCropped,imgWaterShed;
    cap >> src;
    
    std::vector<cv::Point2f> points[2];

    cout<<"rows: "<< src.rows<< " cols: "<<src.cols <<endl;

    // cvtColor(src, gray, COLOR_BGR2GRAY);

    Rect roi = roiSelector(src);
    vector<Rect> thermal_blobs(4);
    vector<Rect> final_thermal_blobs(4);

    filter_rect.height = roi.height*rHeight;
    filter_rect.width = roi.width*rWidth;
    filter_rect.x = (roi.width/2) - (filter_rect.width/2);
    filter_rect.y = (roi.height/2) - (filter_rect.height/2);

    Ptr<MSER> ms = MSER::create();

    while(1){
        cap >> src;
        src.copyTo(dst);
        if (src.empty())
        {
            break;
        }
        // pointSelector(src);
        // imshow("Point select", src);
        cvtColor(src, gray, COLOR_BGR2GRAY);
        // cv::setMouseCallback("Point select", CallBackPoints, 0);
        // std::cout<<clickedCorners.size()<<std::endl;
        grayCropped = gray(roi);
        Mat inp = src(roi);
        if(inp.empty())
        {
            cout<<"empty"<<endl;
        }    
        // Mat image;
        // cv::GaussianBlur(inp, image, cv::Size(0, 0), 3);
        // cv::addWeighted(inp, 1.5, image, -0.5, 0, image);

        // Mat dst = applyKMeans(image.clone());        

        vector<Rect> ret_blob = detectSquares(gray, roi, ms, filter_rect);
        std::vector<std::vector<cv::Point2f>> groupedCorners = getCorners(ret_blob);
        std::vector<cv::Point2f> refinedCorners;
        // goodFeaturesToTrack(gray(roi), points[1], 50, 0.01, 10.0, Mat(), 3, true, 0.04);
        // cornerSubPix(gray(roi), points[1], subPixWinSize, Size((int)filter_rect.width/2,(int)filter_rect.height/2), termcrit);
        
        cout<<"num boxes: "<<ret_blob.size()<< endl;
        Mat markers = createSeeds(ret_blob, roi);
        imshow("markers", markers);
        waitKey(0);
        // cout<<"seeds out "<<endl;
        grayCropped.convertTo(imgWaterShed, 1);
        watershed(imgWaterShed, markers);


        if(groupedCorners.size() != 0)
        {
            for(int i=0; i<groupedCorners[1].size(); i ++)
            {
                for(int c=0; c<groupedCorners[i].size(); c++)
                {
                    Point2f center(groupedCorners[i][c].x, groupedCorners[i][c].y); 
                    circle(dst, center, 3, cv::Scalar(255,0,33), -1 );
                    refinedCorners.push_back(groupedCorners[i][c]);
                }
            }
        }

        // for(int i=0; i<refinedCorners.size(); i++)
        // {
        //     Point2f center(refinedCorners[i].x, refinedCorners[i].y); 
        //     circle(dst, center, 3, cv::Scalar(255,0,33), -1 );
        // }

 
        // for(int a=0; a<ret_blob.size(); a++)
        // {
        //     if (ret_blob[a].area()!=0)
        //     {
                // Mat inp;
                // Mat labels, centers;
                // cout<<"BLOB size: "<<ret_blob[a]<<endl; 
                // src.convertTo(inp,CV_32FC2);
                // imshow("clustered", inp);
                // cout<<"Cropped type: "<< inp.type()<< ", rows and cols: "<<inp.rows<<", "<< inp.cols <<endl; // 5 olmalı

                // int cx = ret_blob[a].x+ ret_blob[a].width/2;
                // int cy = ret_blob[a].y+ ret_blob[a].height/2;

                // if (inp.empty())
                // {
                //     cout<<"img empty"<<endl;
                // }

        //        kmeans(inp, 2, labels,
        //            TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),
        //            3, KMEANS_PP_CENTERS, centers );
        //        imshow("org", gray);
        //     putText(dst, to_string(a), Point(ret_blob[a].x-10, ret_blob[a].y-10),
        //             cv::FONT_HERSHEY_DUPLEX,0.5,cv::Scalar(0,0,0),0.5,false);
        //     rectangle(dst, ret_blob[a], Scalar(0,255,0), 0.1);
        //     }
        // }

        imshow("inference", dst);
        // // imshow("src", src);
        // char key_pressed = (char)(cv::waitKey(1));

        // if (key_pressed == 'r')
        // {
        //     clickedCorners.clear();
        // }

        waitKey(100);
        // video.write(src);
    }
    
    cap.release();
    video.release();
    destroyAllWindows();

    return 0;
}