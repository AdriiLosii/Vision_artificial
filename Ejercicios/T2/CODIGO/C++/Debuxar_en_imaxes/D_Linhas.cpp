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
	// debuxamos liña
	Mat imageLine = image.clone();

	// Empeza en (322,179) e remata (400,183)
	// cor da liña é vermello
	// grosor 5px
	// Tipo de liña LINE_AA

	line(imageLine, Point(200, 80), Point(280, 80), Scalar(0, 255, 0), 3, LINE_AA);
	
	imwrite("./linha.jpg",imageLine);
	
	imshow("Linha",imageLine);
	waitKey(0);
	
	return 0;
}
