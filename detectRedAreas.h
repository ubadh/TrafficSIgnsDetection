#include <iostream>
#include <cmath>
#include <string>
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/objdetect/objdetect.hpp"

using namespace std;
using namespace cv;

vector<Rect> detectRedAreas(Mat img){
    Mat3b res = img.clone();
    
    vector<Rect> result;
    
    // Convert from RGB to HSV
    cvtColor(img, img, COLOR_BGR2HSV);

    // Mask red color
    Mat mask1, mask2;
    inRange(img, Scalar(0, 50, 20), Scalar(5, 255, 255), mask1);
    inRange(img, Scalar(170, 50, 20), Scalar(180, 255, 255), mask2);
    Mat1b mask = mask1 | mask2;

    imshow("mask", mask);
    waitKey(0);

    vector<Point> pts;

    findNonZero(mask, pts);

    int radiusTolerance = 2;

    // Pixels within one radius will be in the same label
    vector<int> labels;

    // Clustering algorithm: Partition
    int tmp = radiusTolerance * radiusTolerance;
    int labelNum = partition(pts, labels, [tmp](const Point& lhs, const Point& rhs){
        return ((lhs.x - rhs.x)*(lhs.x - rhs.x) + (lhs.y - rhs.y)*(lhs.y - rhs.y)) < tmp;
    });

    // Save the points
    vector<vector<Point>> contours(labelNum);
    for (int i=0; i<pts.size(); i++){
        contours[labels[i]].push_back(pts[i]);
    }

    // Get bounding rectangles
    vector<Rect> areas;
    if (contours.size()!=0){
        cout<<"Collected red areas!"<<endl;
    }
    for (int i=0; i<contours.size(); i++){
        Rect area = boundingRect(contours[i]);
        if(contours[i].size()>500){
            areas.push_back(area);

            Rect biggerArea = area + Size(100,100) - Point(30,30);

            // Keep the rect within the dimensions of the image
            if(biggerArea.x<0){
                biggerArea.x = 0;
            }
            if(biggerArea.y<0){
                biggerArea.y = 0;
            }
            if(biggerArea.height + biggerArea.y > res.rows){
                biggerArea.height = res.rows - biggerArea.y;
            }
            if(biggerArea.width + biggerArea.x > res.cols){
                biggerArea.width = res.cols - biggerArea.x;
            }
            result.push_back(biggerArea);
        }
    }
    // Rect biggestArea = *max_element(areas.begin(), areas.end(), [](const Rect& lhs, const Rect& rhs) {
    //     return lhs.area() <= rhs.area();
    // });
    
    // rectangle(res, biggestArea, Scalar(0, 0, 255));
    // Rect biggerArea = biggestArea + Size(20,20)-Point(10,10);
    // rectangle(res, biggerArea, Scalar(0, 255, 0));
    return result;
}