// Includes necesarios
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

	// Creamos unha copia da imaxe orixinal 
	Mat copiedImage = image.clone();

	Mat copyRoi = image(Range(40,200),Range(180,320));
	imshow("Imaxe ROI",copyRoi);
	waitKey(0);

	// Atopamos a altura e anchura da imaxe e da ROI
	int Height = copiedImage.size().height;
	int Width = copiedImage.size().width;

	int roiHeight = copyRoi.size().height;
	int roiWidth = copyRoi.size().width;
	
	//Copiamos a esquerda
	copyRoi.copyTo(copiedImage(Range(10,10+roiHeight),Range(10,10+roiWidth)));
	// Copiamos a dereita
	copyRoi.copyTo(copiedImage(Range(300,300+roiHeight),Range(80,80+roiWidth)));

	imwrite(DATA_PATH + "/RexionsCopiadas.png",copiedImage);

	imshow("Imaxe Resultante",copiedImage);
	waitKey(0);

	return 0;
}
