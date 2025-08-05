#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

// para executar: erosionDilatacion.out imaxe_erosion imaxe_dilatacion

int main(int argc, char** argv){

  //Lemos os argumentos de entrada
  const char* file = argv[1];
  Mat image = imread(file, IMREAD_COLOR);

  if(image.empty())
  {
    cout << "Erro lendo a imaxe" << file<< endl;
    return -1;
  }


imshow("imaxe",image);
waitKey(0);

int kSize = 7;
Mat kernel1 = getStructuringElement(cv::MORPH_ELLIPSE,
       cv::Size(kSize, kSize));

imshow("imaxe",image);
waitKey(0);

Mat imageDilated;
dilate(image, imageDilated, kernel1);

imshow("imaxe",imageDilated);
waitKey(0);

kSize = 3;
Mat kernel2 = getStructuringElement(cv::MORPH_ELLIPSE,
       cv::Size(kSize, kSize));

imshow("imaxe",kernel2*255);
waitKey(0);

Mat imageDilated1, imageDilated2;
dilate(image, imageDilated1, kernel2, Point(-1,-1), 1);
dilate(image, imageDilated2, kernel2, Point(-1,-1), 2);

imshow("imaxe",imageDilated1);
waitKey(0);
imshow("imaxe",imageDilated2);
waitKey(0);


// Imaxe pasada pola linha de comandos
file = argv[2];
image = imread(file, IMREAD_COLOR);
if(image.empty())
  {
    cout << "Erro lendo a imaxe" << file<< endl;
    return -1;
  }


imshow("imaxe",image);
waitKey(0);

Mat imageEroded;
//Primeiro parametro imaxe orixinal e segundo o resultado de erosionar  
erode(image, imageEroded, kernel1);

imshow("imaxe",imageEroded);
waitKey(0);

return 0;
}
