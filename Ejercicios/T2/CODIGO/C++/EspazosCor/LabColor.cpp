#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(){
// Lemos a imaxe en BGR
Mat bgr = imread("../../../data/images/barco.jpg");

// convertemos dende bgr a LAB
Mat labImage;
cvtColor(bgr, labImage, COLOR_BGR2Lab);

vector<Mat> channels(3);
split(labImage,channels);

imshow("L",channels[0]);
imshow("A",channels[1]);
imshow("B",channels[2]);
waitKey(0);

return 0;
}
