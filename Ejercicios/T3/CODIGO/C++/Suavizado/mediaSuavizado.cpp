#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main(){
string filename = "../../../data/images/gaussian-noise.png";
Mat image = imread(filename);

Mat dst1, dst2;

// Filtro da media de tamaño  3x3
blur( image, dst1, Size( 3, 3 ), Point(-1,-1) );

//Filtro da media de tamaño 7x7
blur(image,dst2,Size(7,7),Point(-1,-1));

imshow("Imaxe Orixinal",image);
waitKey(0);
imshow("Filtro da media  1 : KernelSize = 3",dst1);
waitKey(0);
imshow("Filtro da media 2 : KernelSize = 7", dst2);
waitKey(0);

return 0;
}
