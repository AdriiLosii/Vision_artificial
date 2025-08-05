#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main(){
string filename = "../../../data/images/sample.jpg";

// Lemos a imaxe
Mat image = imread(filename, IMREAD_GRAYSCALE);

Mat laplacian, LOG;
int kernelSize = 3;

// Aplicamos a laplacian
Laplacian(image, laplacian, CV_32F, kernelSize, 1, 0);

Mat img1;

GaussianBlur(image, img1, Size(3,3), 0, 0);

// Normalizamos
normalize(laplacian, laplacian, 0, 1, NORM_MINMAX, CV_32F);

imshow("Laplaciana",laplacian);
waitKey(0);

return 0;
}
