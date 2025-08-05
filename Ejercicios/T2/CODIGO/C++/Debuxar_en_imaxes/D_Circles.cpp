// Include librarias
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace std;
using namespace cv;

int main(void){

    // Path a imaxe que queremos ler
    string imageName = "../../../data/images/barco.jpg";

        // cargamos a imaxe na clase Mat
        Mat image;
        image = imread(imageName, IMREAD_COLOR);
            if (image.empty()){
    	cout << "Non puiden ler a imaxe" << endl;
     }
	// debuxamos un circulo
	Mat imageCircle = image.clone();

	circle(imageCircle, Point(250, 125), 100, Scalar(0, 0, 255), 5, LINE_AA);

	imwrite("./Circulo.jpg",imageCircle);
	imshow("Debuxamos un circulo",imageCircle);
	waitKey(0);
	
	// Debuxamos circulo recheo
	Mat imageFilledCircle = image.clone();

	circle(imageFilledCircle, Point(250, 125), 100, Scalar(0, 0, 255), -1, LINE_AA);

	imwrite("./C_recheo.jpg",imageFilledCircle);
	imshow("Debuxamos un circulo recheo",imageFilledCircle);
	waitKey(0);
	return 0;
}

