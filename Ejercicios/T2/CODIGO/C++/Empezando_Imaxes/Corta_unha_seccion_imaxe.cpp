// Incluimos as librarias
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

	// Cortamos un rectangulo
	// x coordenadas = 170 a 320
	// y coordenadas = 40 a 200
	Mat crop = image(Range(40,200),Range(170,320));
	imwrite("./crop.png",crop);
	imshow("Imaxe cortada",crop);
	waitKey(0);
	return 0;
}
