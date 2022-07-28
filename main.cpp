#include <iostream>
#include <cmath>
#include <string>
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/objdetect/objdetect.hpp"

#include "detectRedAreas.h"
#include "detectSpeedLimitSigns.h"
#include "detectWarningSigns.h"
#include "digitClassifierTraining.h"
#include "displaySpeedLimit.h"
#include "getSpeedLimitDigits.h"
#include "loadImgs.h"

using namespace std;
using namespace cv;

String speed_signs_cascade = "cascades/Speedlimit_HAAR_ 15Stages.xml";
String warning_signs_cascade = "cascades/yieldsign12Stages.xml";

CascadeClassifier speed_limit_cascade;
CascadeClassifier warning_cascade;


int main(){
    
    // Train the digit classifier manually
    digitClassifierTraining();

    // Load all images and store them in vector of Mat
    vector<Mat> allImgs = loadImgs();
    for (int i=0; i<allImgs.size(); i++){
        Mat loadedImg = allImgs[i];

        cout<<"<== Image "<<i+1<<" ==>"<<endl;

        // Load the cascades
        if(!speed_limit_cascade.load(speed_signs_cascade) || !warning_cascade.load(warning_signs_cascade)){ 
            cout<<"Cascade couldn't be loaded!"<<endl; 
            return 1; 
        }

        // Create a clone of the image
        Mat loadImgClone = loadedImg.clone();

        // Detect the red areas
        vector<Rect> redAreas = detectRedAreas(loadImgClone);

        // Recognize which type of signs using cascades
        if(redAreas.size()==0){
            cout<<"Sign not found..."<<endl;
            return 1;
        }

        for (int j=0; j<redAreas.size(); j++){
            Mat img = loadedImg(Rect(redAreas[j]));
            vector<Mat> waningsigns = detectWarningSigns(img, warning_cascade);
            if(waningsigns.size()>0){
                cout<<"Warning Sign"<<endl;
            }

            vector<Mat> speedlimitsigns = detectSpeedLimitSigns(img, speed_limit_cascade);
            if(speedlimitsigns.size()>0){
                cout<<"Speed Limit Sign"<<endl;
                for (int i=0; i<speedlimitsigns.size();i++){
                    // Get the speed limit
                    int digit = displaySpeedLimit(getSpeedLimitDigits(speedlimitsigns[i]));
                    if(digit>0){
                        cout<<"Speed limit is "<<digit<<endl;
                    }
                }
            }
            else {
                cout<<"Sign not recognized..."<<endl;
            }
        }
        imshow("Recognition", loadedImg);
        waitKey(0);
    }
}
