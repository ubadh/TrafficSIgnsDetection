#include <iostream>
#include <cmath>
#include <string>
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/objdetect/objdetect.hpp"

using namespace std;
using namespace cv;

void digitClassifierTraining(){
    Mat original = imread("digits.png", 1);
    
    // Convert it to Grayscale
    Mat grayImg;
    cvtColor(original, grayImg, COLOR_BGR2GRAY);
    
    // Use color threshold to leave numbers, and remove background
    Mat thresh, contourImg;
    threshold(grayImg, thresh, 140, 255, THRESH_BINARY_INV);
    thresh.copyTo(contourImg);

    // Store the points of contours
    vector<vector<Point>>contourPoints; 
    
    // Store their hierarchy
    vector<Vec4i> hierarchy;
    findContours(contourImg, contourPoints, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

    Mat digitMat, labelMat;
    
    // Follow the hierarchy order
    for (int i=0; i<contourPoints.size(); i=hierarchy[i][0]){
        
        // Get the bouding rectangle of contour
        Rect rec = boundingRect(contourPoints[i]);
        rectangle(original, Point(rec.x,rec.y), Point(rec.x+rec.width, rec.y+rec.height), Scalar(0,0,255), 2, LINE_8, 0);
        
        // Crop the image
        Mat roi = thresh(rec);
        
        // Resize it to 15 by 15px
        resize(roi, roi, Size(15, 15), 0, 0, INTER_LINEAR);
        
        // Convert it to float
        roi.convertTo(roi, CV_32FC1);

        imshow("original", original);

        // Get the num manually from user or space if not number
        int digit = waitKey(0);
        
        // Convert ASCII to int
        digit -= 0x30;

        // Store label in a matrix
        labelMat.push_back(digit);
        
        // When labeled, rectangle becomes green
        rectangle(original,Point(rec.x, rec.y), Point(rec.x+rec.width, rec.y+rec.height), Scalar(0,255,0), 2, LINE_8, 0);
        
        // Store the 225px of the roi, and make it continuous
        digitMat.push_back(roi.reshape(1,1));
    }
    
    // Store the digit in a file
    FileStorage Digit("digitClassification.yml", FileStorage::WRITE);
    Digit<<"Digit"<<digitMat;
    Digit.release();

    // Make it continuous, and convert to float
    labelMat = labelMat.reshape(1,1);
    labelMat.convertTo(labelMat, CV_32FC1);
    
    // Store the digit in a file
    FileStorage Label("labels.yml", FileStorage::WRITE);
    Label<<"Label"<<labelMat;
    Label.release();

    cout<<"Training has been completed!"<<endl;

    //imshow("original",original);
    waitKey(0);
}
