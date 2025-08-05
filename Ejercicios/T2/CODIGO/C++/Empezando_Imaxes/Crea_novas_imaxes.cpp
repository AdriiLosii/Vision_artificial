// Incluimos as librerias
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main(void){
	// Lemos as imaxes
	string DATA_PATH = "../../imaxes/";
	Mat image = imread(DATA_PATH+"galiza_petroglifos.jpg");
	imshow("Imaxe de entrada",image);
	waitKey(0);
	
	// Creamos un clon da imaxe existente
	Mat imageCopy = image.clone();

	Mat emptyMatrix = Mat(100,200,CV_8UC3, Scalar(0,0,0));
	//Escribimos a imaxe reultante
	imwrite("./MatrizBaleira.png",emptyMatrix);
	imshow("MatrizBaleira",emptyMatrix);
	waitKey(0);
	
	emptyMatrix.setTo(Scalar(255,255,255));
	imwrite("./MatrizBaleiraBranca.png",emptyMatrix);
	imshow("Matriz Baleira Branca",emptyMatrix);

	Mat emptyOriginal = Mat(emptyMatrix.size(), emptyMatrix.type(), Scalar(100,100,100));
	imwrite("./MatrizBaleiraOrixinal_100.png",emptyOriginal);
	imshow("Matriz Orixinal",emptyOriginal);
	
	return 0;
}




