#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

// Variables para almacenar imaxes
Mat dst, cimg, gray, img, edges;

int initThresh;
const int maxThresh = 1000;
double th1,th2;

// vector para almacenar os puntos das linhas
vector<Vec4i> lines;

void onTrackbarChange( int , void* )
{ 
  dst = img.clone();

  th1 = initThresh;
  th2 = th1 * 0.4;

  Canny(img,edges,th1,th2);
  
  // Detectamos linhas empregando a transforamda de HoughLinesP (probabilistica)
  HoughLinesP(edges, lines, 2, CV_PI/180, 50, 10, 100);

  // Debuxamos as linhas sobre os puntos detectados
   for( size_t i = 0; i < lines.size(); i++ )
    {
        Vec4i l = lines[i];
        line( dst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 1, LINE_AA);
    }
   
   // Visualizamos os resultados
   imshow("Imaxe resultado", dst);
   imshow("bordes", edges);
}

int main(int argc, char** argv) {
  const char* file = argv[1];
  // Read image (color mode)
  img = imread(file, 1);
  dst = img.clone();

  if(img.empty())
  {
    cout << "Erro lendo a imaxe" << file<< endl;
    return -1;
  }

  // Convertemos a gris
  cvtColor(img, gray, COLOR_BGR2GRAY);

  // Detectamos borde mediane Canny
  // Canny(gray, dst, 50, 200, 3);
  
  // Facemos unha copia da imaxe orixinal
  // cimg = img.clone();

  // A efectos de visualizacion
  namedWindow("bordes",1);
  namedWindow("Imaxe resultado", 1);
 
  // Declaramos un limiar para a transformada de hough
  initThresh = 500;

  // Creamos unha barra deslizante para cambiar dinamicamente o valor do limiar
  createTrackbar("Limiar", "Imaxe resultado", &initThresh, maxThresh, onTrackbarChange);
  onTrackbarChange(initThresh, 0);

  while(true)
  {
    int key;
    key = waitKey( 1 );
    if( (char)key == 27 )
      { break; }
  }

  destroyAllWindows();
}
