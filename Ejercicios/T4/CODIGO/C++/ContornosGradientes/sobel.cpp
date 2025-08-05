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

Mat sobelx, sobely;

// Aplicamos o Sobel na direccion x
Sobel(image, sobelx, CV_32F, 1, 0);

// Aplicamos o Sobel na direccion y
Sobel(image, sobely, CV_32F, 0, 1);

// Normalizamos a imaxe para visualizacion
normalize(sobelx, sobelx, 0, 1, NORM_MINMAX);
normalize(sobely, sobely, 0, 1, NORM_MINMAX);

imshow("Gradientes X", sobelx);
waitKey(0);
imshow("Gradientes Y", sobely);
waitKey(0);

return 0;
}
