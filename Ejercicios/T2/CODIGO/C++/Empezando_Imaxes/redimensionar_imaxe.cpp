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

	//Dimunimos o tamaño da imaxe
	int resizeDownWidth = 300;
	int resizeDownHeight = 200;
	Mat resizedDown;
	resize(image, resizedDown, Size(resizeDownWidth, resizeDownHeight), INTER_LINEAR);

	// Aumentamos o tamaño
	int resizeUpWidth = 600;
	int resizeUpHeight = 900;
	Mat resizedUp;
	resize(image, resizedUp, Size(resizeUpWidth, resizeUpHeight), INTER_LINEAR);

	imwrite("./resizedUp.png",resizedUp);
	imwrite("./resizedDown.png",resizedDown);
	imshow("Aumentada",resizedUp);
	imshow("Diminuida",resizedDown);
	waitKey(0);
	
	// Escalamos a imaxe 1.5 veces especificando ambolos dous factores de escala 
	double scaleUpX = 1.5;
	double scaleUpY = 1.5;
	
	// Escalamos a imaxe 0.6 veces especificando un unico factor de escala
	double scaleDown = 0.6;
	
	Mat scaledUp, scaledDown;
	
	resize(image, scaledDown, Size(), scaleDown, scaleDown, INTER_LINEAR);
	
	resize(image, scaledUp, Size(), scaleUpX, scaleUpY, INTER_LINEAR);
	
	imwrite("./scaledUp.png", scaledUp);
	imwrite("./scaledDown.png", scaledDown);
	imshow("Aumentamos",scaledUp);
	imshow("Diminuimos", scaledDown);
	waitKey(0);

	cout << "Tamaño (aumentado) = " << scaledUp.size() << endl;
	cout << "Tamaño (diminuido) = " << scaledDown.size() << endl;

	return 0;
}
