#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(){
// Lemos a imaxe en BGR
vector<Mat> channels(3);
Mat bgr = imread("../../../data/images/barco.jpg");

Mat ycbImage;
cvtColor(bgr, ycbImage, COLOR_BGR2YCrCb);
split(ycbImage,channels);

imshow("Y",channels[0]);
imshow("Cr",channels[1]);
imshow("Cb",channels[2]);
waitKey(0);

return 0;
}
