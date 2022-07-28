# ⛔ Traffic Signs Recognition ⚠️

![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)

## Abstract:

Carrying out accurate and efficient Traffic Signs recognition is becoming a key element in modern vehicles. Self-driving cars require traffic signs detection in order to properly parse and understand the road, whereas Driver Alert systems inside new cars need it to understand the road and therefore help protect drivers and keep them in check. Throughout this project, I attempted to use Machine Learning Algorithms to implement a Traffic Signs Recognition System that will be able to differentiate between multiple road signs, and display the speed limits.

## Demo:

## Table of contents:

- [Requirements]()
  - [Homebrew]()
  - [OpenCV]()
- [How to run the program]()
- [Algorithms and Techniques]()
  - [Color exctraction]()
  - []()
  - []()
  - []()
  - []()
  - []()
- []()
- []()
- []()

## Requirements:

### Homebrew

If on macOS, use the following command to install Homebrew:

``` bash
> /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

### OpenCV:

If on macOS, follow the instructions to install openCV:

Install OpenCV

``` bash
> brew install opencv
```

Install pkg-config

``` bash
> brew install pkg-config
```

## How to run the Program:

Use the following commands

``` bash
> g++ $(pkg-config --cflags --libs opencv) -std=c++11  main.cpp -o output
./output
```

## Algorithms and Techniques:

### Overview:

The program follows a rigorous approach to detecting the Traffic Signs. In fact, it starts with a color based step, where the program detects every red pixel, and gathers the regions using a clustering algorithm. Then, using a Machine Learning algorithm and specifically Haar Cascades that I pre-trained, the traffic sign is classified into the corresponding class (Speed Limit, Warning etc...). At last, and in case of a speed limit sign, the program uses a different Machine Learning Algorithm which is the K-nearest neighbor algorithm as well as a modified OCR Tesseract to detect and display the Speed Limit.

### Color extraction:

During this step, the programs begins by converting the image's color scheme from RGB to HSV, then using an inRange function detects and masks all the red areas.

```cpp
// Convert from RGB to HSV
cvtColor(img, img, COLOR_BGR2HSV);
// Mask the red color
Mat mask1, mask2;
inRange(img, Scalar(0, 50, 20), Scalar(5, 255, 255), mask1);
inRange(img, Scalar(170, 50, 20), Scalar(180, 255, 255), mask2);
Mat1b mask = mask1 | mask2;
```

To gather all the red pixels into regions, the program uses a clustering algorithm known as Partition which splits an element set into equivalency classes.

For more information regarding the Partition clustering, check [here](https://docs.opencv.org/2.4/modules/core/doc/clustering.html#partition).

```cpp
// Clustering algorithm: Partition
int tmp = radiusTolerance * radiusTolerance;
int labelNum = partition(pts, labels, [tmp](const Point& lhs, const Point& rhs){
    return ((lhs.x - rhs.x)*(lhs.x - rhs.x) + (lhs.y - rhs.y)*(lhs.y - rhs.y)) < tmp;
});
```

All the points in the same cluster are saved in a vector, and then the formed bounding rectangles represent the regions to be analyzed later.

### Haar Cascades:

For better results, training my own Haar Cascades seemed to be necessary. To do that, there are a few steps to follow.

In fact, to build a proper dataset, both a set of positive samples and negative samples were needed (with negatives being at least double the number of positives). Alongside the sample, a list of all the names of the samples is required. This can easily be done using the command:

``` bash
> find ./positiveSamples -iname "*.jpeg" > positivesamples.txt
> find ./negativeSamples -iname "*.jpeg" > negativesamples.txt
```

In case the dataset is not large enough, rotating the images and changing the perspective can help enlarge the dataset.
To train the dataset, I have used opencv_traincascade.

At the end, it will result in a XML file.

The same steps were followed to get multiple Haar Cascades for different categories.

### Traffic Signs classification
























