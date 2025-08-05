#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main(){
string filename = "../../../data/images/sample.jpg";
Mat image = imread(filename);

if (image.empty())
{
    cout << "Non puiden ler a imaxe" << endl;
}
// Kernel de tamaño 5
int kernelSize = 5;

// Creamos un kernel de 5x5 con todos os elementos a 1
Mat kernel = Mat::ones(kernelSize, kernelSize, CV_32F);

// Normalizamos o kernel para que a suma de todos os elementos sexa 1
kernel = kernel / (float)(kernelSize*kernelSize);

// Imprimimos o kernel
cout << kernel << endl;

// Imaxe de saida
Mat result;

// Aplicamos o filtro (convolucion 2D)
filter2D(image, result, -1 ,
      kernel, Point(-1, -1), 0, BORDER_DEFAULT);

imshow("Imaxe Orixinal",image);
waitKey(0);
imshow("Resultado da convolucion",result);
waitKey(0);

return 0;
}
