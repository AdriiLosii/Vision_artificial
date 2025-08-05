#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(){
string filename = "../../../data/images/dark-flowers.jpg";
Mat img = imread(filename, IMREAD_GRAYSCALE);
// ecualizamos o histograma
Mat imEq;
equalizeHist(img,imEq);

imshow("Imaxe Orixinal",img);
waitKey(0);
imshow("Histograma Ecualizado", imEq);
waitKey(0);

return 0;
}
