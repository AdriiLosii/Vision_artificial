#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
using namespace std;
using namespace cv;

int main(){
//emos a imaxe
Mat img = imread("../data/truth.png", IMREAD_GRAYSCALE);

imshow("imaxe",img);
waitKey(0);

//Limair
Mat imThresh;
threshold(img, imThresh, 127, 255, THRESH_BINARY);

// Atopamos as componhentes conectadas
Mat imLabels;
int nComponents = connectedComponents(imThresh,imLabels);

Mat imLabelsCopy = imLabels.clone();

// Atopamos o maximo e minimo na imaxe e as suas localizacions
Point minLoc, maxLoc;
double min, max;
minMaxLoc(imLabels, &min, &max, &minLoc, &maxLoc);

// Normaliza a imaxe entre 0 e 255
imLabels = 255 * (imLabels - min) / (max - min);

// # convertemos a imaxe a unsigned  8-bits
imLabels.convertTo(imLabels, CV_8U);

imshow("imaxe",imLabels);
waitKey(0);
imLabels = imLabelsCopy.clone();
// Display the labels
cout << "Numero de componhentes = " << nComponents;

for (int i=0; i < 6; i++){
	imshow("image",imLabels==i);
	waitKey(0);
}

// copia da imaxe
imLabels = imLabelsCopy.clone();

//A seguinte liña atopa os valores mínimos e máximos de píxeles
// e as súas localizacións na imaxe.
double minValue,maxValue;
minMaxLoc(imLabels, &minValue, &maxValue, &minLoc, &maxLoc);

// Normalizamos a 0 - 255
imLabels = 255 * (imLabels - minValue) / (maxValue - minValue);

// Convertemos a 8-bits
imLabels.convertTo(imLabels, CV_8U);

// Aplicamos mapa de cor
Mat imColorMap;
applyColorMap(imLabels, imColorMap, COLORMAP_JET);

// Visualizamos as etiquetas
imshow("imaxe",imColorMap);
waitKey(0);

return 0;
}
