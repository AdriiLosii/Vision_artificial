#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main(){
string filename = "../../../data/images/gaussian-noise.png";
Mat image = imread(filename);

//Diametro para incluir a vecindade para o filtrado
int dia=15;

// Valores elevados mesturar as cores
// para producir áreas de cores semi-guais
double sigmaColor=80;

// Valores grandes ten en conta valores de pixel moi alonxados de pixel de interes
double sigmaSpace=80;

Mat bilateralFiltered;

// Aplicamos o filtro bilatral
bilateralFilter(image, bilateralFiltered, dia,
              sigmaColor,
              sigmaSpace);

imshow("Imaxe Orixinal",image);
waitKey(0);
imshow("Resultado do suavizado bilateral",bilateralFiltered);
waitKey(0);

return 0;
}
