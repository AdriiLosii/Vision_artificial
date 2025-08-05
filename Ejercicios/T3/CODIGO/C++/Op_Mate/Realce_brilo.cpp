// Includes
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;


int main(void){
    // Read image
    	Mat image = imread("../../data/images/boy.jpg");
	int brightnessOffset = 50;

	// Sumamos un offset para incrementar o brilo
	Mat brightHigh = image;
	Mat brightHighChannels[3];
	split(brightHigh, brightHighChannels);

	for (int i=0; i < 3; i++){
	    add(brightHighChannels[i],brightnessOffset,brightHighChannels[i]);
	}

	merge(brightHighChannels,3,brightHigh);
	//imwrite("./briloAlto.png",brightHigh);

	imshow("Brilo Alto", brightHigh);
	waitKey(0);
	
	double min, max;
	minMaxLoc(image, &min, &max);
	cout << "Maxima intensidade da orixinal : " << max << endl;
	minMaxLoc(brightHigh, &min, &max);
	cout << "Maxima intensidade da imaxe de alto brillo : " << max << endl;


	return 0;
}
