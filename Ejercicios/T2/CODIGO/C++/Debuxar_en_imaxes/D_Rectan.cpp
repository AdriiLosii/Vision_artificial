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

	// Debuxamos rectangulo (grosor e unha numero enteiro)
	Mat imageRectangle = image.clone();
	rectangle(imageRectangle, Point(170, 50), Point(300, 200),
          Scalar(255, 0, 255), 5, LINE_8);

	imwrite("./rectangulo.jpg",imageRectangle);

	imshow("Debuxamos rectangulo",imageRectangle);
	waitKey(0);
	return 0;
}
