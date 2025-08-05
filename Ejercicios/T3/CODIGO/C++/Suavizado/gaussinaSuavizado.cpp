#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main(){
string filename = "../../../data/images/gaussian-noise.png";
Mat image = imread(filename);

Mat dst1, dst2;

// Aplicamos o filtro da gaussiana
GaussianBlur(image, dst1, Size( 5, 5 ), 0, 0 );

// Incrementamos sigma
GaussianBlur(image,dst2,Size(25,25),50,50);

imshow("Imaxe Orixinal",image);
waitKey(0);
imshow("Filtro gaussiana 1 : KernelSize = 5",dst1);
waitKey(0);
imshow("Filtro gaussiana 2 : KernelSize = 25",dst2);
waitKey(0);

return 0;
}
