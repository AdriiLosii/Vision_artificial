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

        // Cargamos imaxe
        Mat image;
        image = imread(imageName, IMREAD_COLOR);
        if (image.empty()){
    		cout << "Non puiden ler a imaxe" << endl;
     	}

	// Debuxamos elipse
	// Nota: o centro e os eixos deben ser enteiros
	Mat imageEllipse = image.clone();

	ellipse(imageEllipse, Point(250, 125), Point(100, 50), 0, 0, 360,
        Scalar(255, 0, 0), 3, LINE_AA);

	ellipse(imageEllipse, Point(250, 125), Point(100, 50), 90, 0, 360,
        Scalar(0, 0, 255), 3, LINE_AA);

	imwrite("./elipse.jpg",imageEllipse);
	imshow("Debuxa elipse", imageEllipse);
	waitKey(0);
	
	// Debuxamos elipse
	// Nota: o centro e os eixos deben ser enteiros
	imageEllipse = image.clone();

	// Elipse incmpleta
	ellipse(imageEllipse, Point(250, 125), Point(100, 50), 0, 180, 360,
        Scalar(255, 0, 0), 3, LINE_AA);

	// Elipse rechea
	ellipse(imageEllipse, Point(250, 125), Point(100, 50), 0, 0, 180,
        Scalar(0, 0, 255), -2, LINE_AA);

	imwrite("./e_rechea.jpg",imageEllipse);

	imshow("Elipse rechea",imageEllipse);
	waitKey(0);
	return 0;
}
