#include "../include/codigoMostra.hpp"
using namespace std;
using namespace cv;

int main(void) {

	// Lemos a imaxe en modo de gris
	Mat image = imread("../data/gaiteiro.jpg",0);
	if (image.empty())
	{
    		cout << "Non puiden ler a imaxe" << endl;
	}

	// Salvamos a imaxe de gris
	imwrite("../data/gaiteiroGris.jpg",image);
	imshow("Imaxe convertida a gris",image);
	waitKey(0);

	return 0;
}
