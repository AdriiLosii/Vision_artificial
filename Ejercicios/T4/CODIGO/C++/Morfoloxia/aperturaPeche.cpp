#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

// para executar: aperturaPeche.out imaxe_apertura imaxe_peche
int main(int argc, char** argv){

  //Lemos os argumentos de entrada
  const char* file = argv[1];
  Mat image = imread(file, IMREAD_GRAYSCALE);

  if(image.empty())
  {
    cout << "Erro lendo a imaxe" << file<< endl;
    return -1;
  }


imshow("imaxe",image);
waitKey(0);

// Especificamos o tamanho do kernel
int kernelSize = 10;

// Creamos o kernel
Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2*kernelSize+1, 2*kernelSize+1),
                                    Point(kernelSize, kernelSize));

imshow("imaxe",255*element);
waitKey(0);

Mat imEroded;
// Realizamos a dilatacion
erode(image, imEroded, element, Point(-1,-1),1);

imshow("imaxe",255*imEroded);
waitKey(0);

Mat imOpen;
dilate(imEroded, imOpen, element, Point(-1,-1),1);

imshow("imaxe",255*imOpen);
waitKey(0);

// Tamanho do kernel para a operacion de apertura
int openingSize = 3;

// Seleccionamos un kernel eliptico
element = getStructuringElement(MORPH_ELLIPSE,Size(2 * openingSize + 1, 2 * openingSize + 1),
                                Point(openingSize, openingSize));

Mat imageMorphOpened;
morphologyEx(image, imageMorphOpened, MORPH_OPEN, element, Point(-1,-1),3);
imshow("imaxe",imageMorphOpened*255);
waitKey(0);


  //Lemos os argumentos de entrada
  file = argv[2];
  image = imread(file, IMREAD_GRAYSCALE);

  if(image.empty())
  {
    cout << "Erro lendo a imaxe" << file<< endl;
    return -1;
  }


// tamanho kernel
kernelSize = 10;
// creamos kernel
element = getStructuringElement(MORPH_ELLIPSE, Size(2*kernelSize+1, 2*kernelSize+1),
                                    Point(kernelSize, kernelSize));

Mat imDilated;
// Realizamos a dilatacion
dilate(image, imDilated, element);

imshow("imaxe",imDilated);
waitKey(0);

Mat imClose;
//Erosion
erode(imDilated, imClose, element);

imshow("imaxe",imClose);
waitKey(0);

// Creamos elemento estrutural
int closingSize = 10;
// Seleccionamos forma eliptica
element = getStructuringElement(MORPH_ELLIPSE,Size(2 * closingSize + 1, 2 * closingSize + 1),
                                Point(closingSize, closingSize));

Mat imageMorphClosed;
morphologyEx(image, imageMorphClosed, MORPH_CLOSE, element);
imshow("imaxe",imageMorphClosed);
waitKey(0);

return 0;
}
