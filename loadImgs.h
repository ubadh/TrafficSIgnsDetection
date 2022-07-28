#include <iostream>
#include <cmath>
#include <string>
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/objdetect/objdetect.hpp"

using namespace std;
using namespace cv;

// Load the images
vector<Mat> loadImgs(){
    vector<String> imgs;
    glob("dataset/*.jpeg", imgs, false);
    vector<Mat> images;
    for (int i=0; i<imgs.size(); i++)
        images.push_back(imread(imgs[i]));
    return images;
}