// Includes necesarios
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main(void){
	//Lemos a imaxe
	Mat image = imread("../../data/images/book1.jpg");
	
	double scalingFactor = 1/255.0;
	double shift = 0;

	//Convertemos dende unsigned char a float(32bit)
	image.convertTo(image, CV_32FC3, scalingFactor, shift);
	imshow("Float32",image);

	//Convertemos dende float a unsigned char
	image.convertTo(image, CV_8UC3, 1.0/scalingFactor, shift);
	imshow("Uint8",image);
	waitKey(0);
	
	return 0;
}
