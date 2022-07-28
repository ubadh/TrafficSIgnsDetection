#include <iostream>
#include <cmath>
#include <string>
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/objdetect/objdetect.hpp"

using namespace std;
using namespace cv;

int displaySpeedLimit(string nums){
    if ((nums.find("20") != string::npos) || (nums.find("02") != string::npos)) {
        return 20;
    }
    if ((nums.find("30") != string::npos) || (nums.find("03") != string::npos)) {
        return 30;
    }
    if ((nums.find("50") != string::npos) || (nums.find("05") != string::npos)) {
        return 50;
    }
    if ((nums.find("60") != string::npos) || (nums.find("06") != string::npos)) {
        return 60;
    }
    if ((nums.find("70") != string::npos) || (nums.find("07") != string::npos)) {
        return 70;
    }
    if ((nums.find("80") != string::npos) || (nums.find("08") != string::npos)) {
        return 80;
    }
    if ((nums.find("90") != string::npos) || (nums.find("09") != string::npos)) {
        return 90;
    }
    if ((nums.find("100") != string::npos) || (nums.find("001") != string::npos)) {
        return 100;
    }
    if ((nums.find("120") != string::npos) || (nums.find("021") != string::npos)) {
        return 120;
    }
    if ((nums.find("130") != string::npos) || (nums.find("031") != string::npos)) {
        return 130;
    }
    if ((nums.find("140") != string::npos) || (nums.find("041") != string::npos)) {
        return 140;
    }
    return -1;
}