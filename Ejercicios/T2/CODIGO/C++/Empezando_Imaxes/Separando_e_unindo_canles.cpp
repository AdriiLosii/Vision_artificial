//Incluimos as cabeceiras importantes
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main(void)
{
    	string imagePath = "../petroglifos_galiza.jpg";
	// Lemos a imaxe.Tipo de dato Mat
    	Mat img = imread(imagePath);
	// Amosamos as canles
	Mat imgChannels[3];
	split(img, imgChannels);

	// Escribimos a disco as canles
	imwrite("./imgBlue.png",imgChannels[0]);
	imwrite("./imgGreen.png",imgChannels[1]);
	imwrite("./imgRed.png",imgChannels[2]);

	//Amosamos as canles de cor
	imshow("Blue Channel",imgChannels[0]);
	imshow("Green Channel",imgChannels[1]);
	imshow("Red Channel",imgChannels[2]);
	waitKey(0);

	return 0;
}
