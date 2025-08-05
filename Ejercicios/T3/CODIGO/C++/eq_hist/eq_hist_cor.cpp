#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(){
string filename = "../../../data/images/dark-flowers.jpg";
Mat img = imread(filename);
Mat imEq = img.clone();

// Realizamos a ecualizacion para as canles por separado
vector<Mat> imgChannels(3);
vector<Mat> imEqChannels(3);

split(img,imgChannels);
split(imEq,imEqChannels);

for (int i=0; i<3; i++){
    equalizeHist(imgChannels[i],imEqChannels[i]);
}

merge(imgChannels,img);
merge(imEqChannels,imEq);

imshow("Imaxe Orixinal",img);
waitKey(0);
imshow("Histograma ecualizado", imEq);
waitKey(0);

filename = "../../../data/images/dark-flowers.jpg";
img = imread(filename);
Mat imhsv;

cvtColor(img, imhsv, COLOR_BGR2HSV);

vector<Mat> imhsvChannels(3);
split(imhsv,imhsvChannels);

// Ecualizacion so da candle V 
equalizeHist(imhsvChannels[2],imhsvChannels[2]);
merge(imhsvChannels,imhsv);
// Convertimos ao formato BGR
cvtColor(imhsv,imEq, COLOR_HSV2BGR);

imshow("Imaxe Orixianal",img);
waitKey(0);
imshow("Histograma ecualizado",imEq);
waitKey(0);

return 0;
}
