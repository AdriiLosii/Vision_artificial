#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main(){
string filename = "../../../data/images/gaussian-noise.png";
Mat image = imread(filename);

// Definimos o tamaño do kernel
int kernelSize = 5;

Mat medianBlurred;
// Realizamos o suavizado da mediana e almacenamos o resultado nun array de numpy
medianBlur(image,medianBlurred,kernelSize);

imshow("Imaxe Orixinal",image);
waitKey(0);
imshow("Filtro da mediana : KernelSize = 5",medianBlurred);
waitKey(0);
return 0;
}
