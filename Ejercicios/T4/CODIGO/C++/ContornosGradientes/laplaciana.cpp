#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;


int main(int argc, char** argv){

  //Lemos os argumentos de entrada
  const char* file = argv[1];
  Mat image = imread(file, IMREAD_GRAYSCALE);

  if(image.empty())
  {
    cout << "Erro lendo a imaxe" << file<< endl;
    return -1;
  }

Mat laplacian, LOG;
int kernelSize = 3;

// aplicamos a laplaciana
Laplacian(image, laplacian, CV_32F, kernelSize, 1, 0);

Mat img1;

GaussianBlur(image, img1, Size(3,3), 0, 0);

// Normalizamos as imaxes
normalize(laplacian, laplacian, 0, 1, NORM_MINMAX, CV_32F);

imshow("Laplaciana",laplacian);
waitKey(0);

return 0;
}
