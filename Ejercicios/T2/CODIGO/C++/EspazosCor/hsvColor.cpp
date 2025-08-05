#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(){
// Lemos a imaxe en BGR
Mat bgr = imread("../../../data/images/barco.jpg");

// convertimos de bgr a HSV
Mat hsvImage;
cvtColor(bgr, hsvImage, COLOR_BGR2HSV);

Mat hsvChannels[3];
split(hsvImage,hsvChannels);

imshow("Hue",hsvChannels[0]);
imshow("Saturation",hsvChannels[1]);
imshow("Value",hsvChannels[2]);
waitKey(0);

Mat imH, imS, imV, imhsv, imbgr;
vector<Mat> channels;

for (int i=0; i < 7; i++) {
    // Creamos canles HSV de 50x50
    // Facemos Hue = 0, Saturation = 0, Value = i x 40
    imH = Mat::zeros(Size(50,50),CV_8U);
    imS = Mat::zeros(Size(50,50),CV_8U);
    imV = Mat::ones(Size(50,50),CV_8U) * (i * 40);

    channels = {imH, imS, imV};

    merge(channels, imhsv);

    // Convertemos HSV a bgr
    cvtColor(imhsv, imbgr, COLOR_HSV2BGR);

    // Visualozamos
    imshow("V="+to_string(i*40),imbgr);
    waitKey(0);
}

for (int i=0; i < 7; i++) {
    // Creamos canles HSV de 50x50
    // Facemos Hue = 0, Value = 128
    imH = Mat::zeros(Size(50,50),CV_8U);
    imS = Mat::ones(Size(50,50),CV_8U) * (i*40);
    imV = Mat::ones(Size(50,50),CV_8U) * 128;

    channels = {imH, imS, imV};

    merge(channels, imhsv);

    // Convertemos HSV a bgr
    cvtColor(imhsv, imbgr, COLOR_HSV2BGR);

    // Visualizamos
    imshow("S="+to_string(i*40),imbgr);
    waitKey(0);
}

for (int i=0; i < 7; i++) {
    // Create 50x50 HSV channelsCreamos canles HSV de 50x50
    // Facemos Saturation = 128, Value = 128
    imH = Mat::ones(Size(50,50),CV_8U) * (i*30);
    imS = Mat::ones(Size(50,50),CV_8U) * 128;
    imV = Mat::ones(Size(50,50),CV_8U) * 128;

    channels = {imH, imS, imV};

    merge(channels, imhsv);

    // Convertemos HSV a bgr
    cvtColor(imhsv, imbgr, COLOR_HSV2BGR);

    // Visualizamos
    imshow("H="+to_string(i*40),imbgr);
    waitKey(0);
}

return 0;
}
