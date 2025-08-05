// Incluimos as cabeceiras de Opencv
#include "opencv2/opencv.hpp"
#include <iostream>

//Empregamos os espazos std e cv (opnecv)
using namespace std;
using namespace cv;

int main(){

  VideoCapture cap("chaplin.mp4");
  // Comprobamos se a camara se inicializou correctamente
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }

  while(1){

    Mat frame;
    // captura frame-a-frame
    cap >> frame;
 
    // Se o frame esta baleiro, rompemos o bucle
    if (frame.empty())
      break;

    // Visualizamos o frame
    imshow( "Frame", frame );
           
    // Presiona ESC para sair
    char c=(char)waitKey(25);
    if(c==27)
      break;
  }

  // Cando rematamos, eliminamos os obxectos de video
  cap.release();

  // Pechamos todas a xanelas
  destroyAllWindows();

  return 0;
}
