#include <iostream>
#include <cmath>
#include <string>
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/objdetect/objdetect.hpp"

using namespace std;
using namespace cv;

string getSpeedLimitDigits(Mat img){
    Mat imgClone = img.clone();
    
    // Convert it to Grayscale
    Mat grayImg;
    cvtColor(imgClone, grayImg, COLOR_BGR2GRAY);
    
    // Use color threshold to leave numbers, and remove background
    Mat thresh, contourImg;
    threshold(grayImg, thresh, 140, 255, THRESH_BINARY_INV);
    thresh.copyTo(contourImg);

    // Read stored Digits from digit classification training
    Mat digitMat;
    FileStorage Digit("digitClassification.yml", FileStorage::READ);
    Digit["Digit"]>>digitMat;
    Digit.release();

    // Read stored Labels from digit classification training
    Mat labelMat;
    FileStorage Label("labels.yml", FileStorage::READ);
    Label["Label"]>>labelMat;
    Label.release();

    // KNN training using digit classification
    Ptr<ml::KNearest>  knn(ml::KNearest::create());
    knn->train(digitMat, ml::ROW_SAMPLE,labelMat);
    cout<<"KNN training is done!"<<endl;

    vector<vector<Point>> contoursPoints;
    vector<Vec4i> hierarchy;

    // Find contours and save them
    findContours(contourImg, contoursPoints, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
    Mat res(imgClone.rows, imgClone.cols, CV_8UC3, Scalar::all(0));

    string result;

    // Follow the hierarchy order
    for (int i=0; i<contoursPoints.size(); i=hierarchy[i][0]){
        Rect r = boundingRect(contoursPoints[i]);
        Mat roi = thresh(r);
        resize(roi, roi, Size(10,10), 0, 0, INTER_LINEAR);
        roi.convertTo(roi,CV_32FC1);
       
        Mat closeLabel;
        float p = knn -> findNearest(roi.reshape(1,1),4, closeLabel);
        char name[4];
        sprintf(name, "%d", (int)p);
        result = result + to_string((int)p);
        putText(res, name, Point(r.x,r.y+r.height), 0, 1, Scalar(0, 255, 0), 2, LINE_8);
    }
    imwrite("Result", res);
    return  result ;
}