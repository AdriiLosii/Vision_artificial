#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv){

  //Lemos os argumentos de entrada
  const char* file = argv[1];
  Mat image = imread(file, IMREAD_COLOR);

  if(image.empty())
  {
    cout << "Erro lendo a imaxe" << file<< endl;
    return -1;
  }

  Mat sharpen = (Mat_<int>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
  Mat sharpenOutput;
  filter2D(image, sharpenOutput, -1, sharpen);
  imshow("Imaxe Orixinal",image);
  waitKey(0);
  imshow("Resultado do enfoque (sharpening)",sharpenOutput);
  waitKey(0);

  return 0;
}
