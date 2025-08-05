#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// Varaible para imaxe orixe e de bordes
Mat src, edges;

// Limiares alto e baixo da histerese
int lowThreshold = 50;
int highThreshold = 100;

// Valor Max de trackbar
int const maxThreshold = 1000;


// Tamanhos das aperturas do soble para o detector de canny
int apertureSizes [] = {3, 5, 7};
int maxapertureIndex = 2;
int apertureIndex = 0;

// Suavizamos gaussiano
int blurAmount = 0;
int maxBlurAmount = 20;


// Funcion para a chamada do trackbar
void applyCanny(int ,void *)
{
  // Variable para almacenar a imaxe suavizada
  Mat blurredSrc;

  // Suavizamos a imaxe antes de detectar os contornos
  if (blurAmount > 0 )
  {
    GaussianBlur(src, blurredSrc, Size( 2 * blurAmount + 1, 2 * blurAmount + 1), 0);
  }
  else
  {
    blurredSrc = src.clone();
  }

  // Canny require un tamanho de aperture impar!
  int apertureSize = apertureSizes[apertureIndex];

  // Aplicamos canny para obter os bordes
  Canny( blurredSrc, edges, lowThreshold, highThreshold, apertureSize );

  //Visualizamos
  imshow("Bordes",edges);
}

int main(int argc, char** argv){

  //Lemos os argumentos de entrada en linha de comandos
  const char* file = argv[1];

// Lemos a imaxe
src = imread(file, IMREAD_GRAYSCALE);
if(src.empty())
  {
    cout << "Erro lendo a imaxe" << file<< endl;
    return -1;
  }


//Visualizamos
imshow("Bordes",src);

// Creamos a xanela para visualizar a saida.
namedWindow("Bordes",WINDOW_AUTOSIZE);

// Control do Trackbar para o limiar inferior
createTrackbar( "Limiar baixo", "Bordes", &lowThreshold, maxThreshold, applyCanny);

// Control do Trackbar para o limiar alto
createTrackbar( "Limiar alto", "Bordes", &highThreshold, maxThreshold, applyCanny);

// Control do Trackbar para o tamaño da apertura
createTrackbar( "Tamanho apertura", "Bordes", &apertureIndex, maxapertureIndex, applyCanny);

// Control do Trackbar para o suavizado
createTrackbar( "Suavizado", "Bordes", &blurAmount, maxBlurAmount, applyCanny);

waitKey(0);

destroyAllWindows();
}
