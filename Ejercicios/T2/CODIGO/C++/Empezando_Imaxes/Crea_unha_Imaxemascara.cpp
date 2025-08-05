// Incluimos as librerias
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main(void){
        // Lemos a imaxe
	string DATA_PATH = "../../imaxes/";
    	Mat image = imread(DATA_PATH + "petroglifos_galiza.jpg");

	// Creamos unha imaxa nova do mesmo tamaño que a orixinal
	Mat mask1 = Mat::zeros(image.size(), image.type());
	imwrite("./mascara.png",mask1);
	imshow("Mascara",mask1);
	waitKey(0);

	mask1(Range(50,200),Range(170,320)).setTo(255);
	imwrite("./mascaraRevisada.png",mask1);
	imshow("Mascara Revisada",mask1);
	waitKey(0);

	Mat mask2;
	inRange(image, Scalar(0,0,150), Scalar(100,100,255), mask2);
	imwrite("./mascara2.png",mask2);
	imshow("Mask 2",mask2);
	waitKey(0);

	return 0;
}
