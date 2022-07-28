#include <iostream>
#include <cmath>
#include <string>
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/objdetect/objdetect.hpp"

using namespace std;
using namespace cv;

vector<Mat> detectSpeedLimitSigns(Mat img, CascadeClassifier speed_limit_cascade){
    Mat grayImg;

    // Convert to grayscale
    cvtColor(img, grayImg, COLOR_BGR2GRAY);

    // Decrease brightness and increase contrast
    equalizeHist(grayImg, grayImg);

    // Detect the sign
    vector<Rect> signs;
    speed_limit_cascade.detectMultiScale(grayImg, signs, 1.1, 3, 0|CASCADE_SCALE_IMAGE, Size(30, 30));
    cout<<speed_limit_cascade.getFeatureType()<<endl;

    vector<Mat> result;
    // Draw the ellipse
    for (int i=0; i<signs.size(); i++){
        Point center(signs[i].x + signs[i].width*0.5, signs[i].y + signs[i].height*0.5);
        ellipse(img, center, Size( signs[i].width*0.5, signs[i].height*0.5), 0, 0, 360, Scalar(0, 255, 0), 4, 8, 0);
        Mat resultImage = img(Rect(center.x - signs[i].width*0.5,center.y - signs[i].height*0.5,signs[i].width,signs[i].height));
        result.push_back(resultImage);
    }
    return result;
}