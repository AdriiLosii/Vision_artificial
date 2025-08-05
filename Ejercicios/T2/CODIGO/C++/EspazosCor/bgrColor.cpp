#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(){
// Lemos a imaxe en BGR
Mat bgr = imread("../../../data/images/barco.jpg");

imshow("Imaxe Orixinal",bgr);
waitKey(0);

Mat bgrChannels[3];
split(bgr,bgrChannels);

imshow("B",bgrChannels[0]);
imshow("G",bgrChannels[1]);
imshow("R",bgrChannels[2]);
waitKey(0);

return 0;
}
