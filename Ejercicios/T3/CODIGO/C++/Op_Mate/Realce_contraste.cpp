// Includes
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main(void){
    // Lemos a imaxe
    Mat image = imread("../../data/images/book1.jpg");
	double contrastPercentage = 30.0;
	// factor para incrementar o contraste
	Mat contrastHigh = image;
	
	// Convertemos a float
	contrastHigh.convertTo(contrastHigh, CV_64F);
	contrastHigh = contrastHigh * (1+contrastPercentage/100.0);
	//imwrite("./altacontraste.png",contrastHigh);
	imshow("Alto Contraste",contrastHigh);
	waitKey(0);
	
	return 0;
}
