// Incluimos as cabeceiras de Opencv
#include "opencv2/opencv.hpp"
#include <iostream>

//Empregamos os espazos std e cv (opnecv)
using namespace std;
using namespace cv;

int main(){

  // Creamos o obxecto VideoCapture coa primeira camara 
  VideoCapture cap(0);

  // Comprobamos se a camara se inicializou correctamente
  if(!cap.isOpened()){
    cout << "Erro na apertura do stream de video ou lectura dende ficheiro" << endl;
    return -1;
  }

  int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
  int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
  VideoWriter outavi("output.avi",cv::VideoWriter::fourcc('M','J','P','G'),10, Size(frame_width,frame_height));
  VideoWriter outmp4("output.mp4",cv::VideoWriter::fourcc('X','V','I','D'),10, Size(frame_width,frame_height));

  //Lemos e escribimos dende a webcam ata que se presiona a tecla ESC.
  while(1){

    Mat frame;
    // captura frame-a-frame
    cap >> frame;

    // Se o frame esta baleiro, rompemos o bucle
    if (frame.empty())
      break;

    // Escribimos o frame
    outavi.write(frame);
    outmp4.write(frame);

    imshow("Frame",frame);
    char c=(char)waitKey(25);
    if (c == 27)
      break;
  }

  // Cando rematamos a faena, eliminamos os obxectos de video creados
  cap.release();
  outavi.release();
  outmp4.release();

  // Closes all the frames
  destroyAllWindows();

  return 0;
}
