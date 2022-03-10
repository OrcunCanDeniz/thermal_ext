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
cv::Mat gray, src, dst, inp, k_out;
std::vector<std::vector<cv::Point> > contours;
int thresh = 100;
cv::RNG rng(12345);
std::vector<cv::Vec4i> hierarchy;
int MAX_ITERATIONS = 4;



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

// void thresh_callback(int, void* )
// {
    // cv::Mat canny_output;
    // cv::Canny( k_out, canny_output, thresh, thresh*2 );
    // // for( size_t i = 0; i < circles.size(); i++ ) // draw over canny output
    // // {
    // //     cv::Vec3i c = circles[i];
    // //     cv::Point center = cv::Point(c[0], c[1]);
    // //     // circle center
    // //     // cv::circle(dst, center, -1, cv::Scalar(0,0,0), 3, cv::LINE_AA);
    // //     // circle outline
    // //     float radius = c[2] * 1.15;
    // //     int rounded_radius = std::round(radius*1.15);
    // //     cv::circle( dst, center, rounded_radius, cv::Scalar(0,0,0), -1, cv::LINE_AA);
    // // }
    // cv::findContours( canny_output, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE );
    // cv::Mat drawing_cb = cv::Mat::zeros( canny_output.size(), CV_8UC3 );
    // std::cout<< "Num of conts; " << contours.size() <<std::endl; 
    // for( size_t i = 0; i< contours.size(); i++ )
    // {
        // cv::Scalar color = cv::Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
        // drawContours( drawing_cb, contours, (int)i, color, 0.5, cv::LINE_8, hierarchy, 0 );
    // }
    // std::cout<<drawing_cb.rows<<std::endl;
    // imshow( "Contours", drawing_cb );
// }



using namespace cv;
using namespace std;

Mat applyKMeans(const Mat& source){

	const unsigned int singleLineSize = source.rows * source.cols;
    // Mat gry;
    // cvtColor(source, gry, COLOR_BGR2GRAY);
	Mat data = source.reshape(1, singleLineSize);
	data.convertTo(data, CV_32F);
	std::vector<int> labels;
	cv::Mat1f colors;
	// kmeans(data, 2, labels,
            // TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),
            //    3, KMEANS_PP_CENTERS, centers);
    cv::kmeans(data, 2,labels,cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 10, 1.),MAX_ITERATIONS,cv::KMEANS_PP_CENTERS, colors);

    for (unsigned int i = 0 ; i < singleLineSize ; i++ ){
        // if(labels[i] == 1)
        // {
        //         data.at<float>(i, 0) = 255;
		//         data.at<float>(i, 1) = 255;
		//         data.at<float>(i, 2) = 255;
        // } else {
        //     data.at<float>(i, 0) = 0;
        //     data.at<float>(i, 1) = 0;
        //     data.at<float>(i, 2) = 0;
        // }
        		data.at<float>(i, 0) = colors(labels[i], 0);
		        data.at<float>(i, 1) = colors(labels[i], 1);
		        data.at<float>(i, 2) = colors(labels[i], 2);
	}

	Mat outputImage = data.reshape(3, source.rows);

    outputImage.convertTo(outputImage, CV_8U);
	return outputImage;
}


int main(int argc, char *argv[])
{
    VideoCapture cap("/home/orcun/thermal_ext_calib/data/new_board/thermal_ext_board.avi",0);

    VideoWriter video("outcpp.avi",CV_FOURCC('M','J','P','G'), 10, Size(640,512));

    cv::Rect filter_rect;
    float rHeight=0.66, rWidth=0.45;

    cap >> src;


    cout<<"rows: "<< src.rows<< " cols: "<<src.cols <<endl;

    cvtColor(src, gray, COLOR_BGR2GRAY);

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
        if (src.empty())
        {
            break;
        }
        imshow("Select roi", src);
        cvtColor(src, gray, COLOR_BGR2GRAY);
        

        inp = src(roi);
        if(inp.empty())
        {
            cout<<"empty"<<endl;
        }    

        Mat image;
        // cv::GaussianBlur(inp, image, cv::Size(0, 0), 3);
        // cv::addWeighted(inp, 1.5, image, -0.5, 0, image);

        k_out = applyKMeans(inp.clone());        
        imshow("inference", k_out);

        // namedWindow("cropped canny");
        // resizeWindow("cropped canny", k_out.cols, k_out.rows);
        // imshow( "cropped canny", k_out);
        // const int max_thresh = 255;
        // createTrackbar( "Canny thresh:", "cropped canny", &thresh, max_thresh, thresh_callback );
        // thresh_callback( 0, 0 );
        char key_pressed = (char)(cv::waitKey(1));

        waitKey(10);
        video.write(src);
    }
    
    cap.release();
    video.release();
    destroyAllWindows();

    return 0;
}