#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <time.h>

using namespace std;
using namespace cv;

int main(){
// lEMOS A iamxe
Mat img = imread("../data/blob_detection.jpg", IMREAD_GRAYSCALE);

// Detector con parametros por defecto
Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create();

std::vector<KeyPoint> keypoints;
detector->detect(img,keypoints);

// Marcamos os blobs
int x,y;
int radius;
double diameter;
cvtColor(img, img, COLOR_GRAY2BGR);
for (int i=0; i < keypoints.size(); i++){
    KeyPoint k = keypoints[i];
    Point keyPt;
    keyPt = k.pt;
    x=(int)keyPt.x;
    y=(int)keyPt.y;
    // Centro en negro
    circle(img,Point(x,y),5,Scalar(255,0,0),-1);
    // Radio do blob
    diameter = k.size;
    radius = (int)diameter/2.0;
    // Marcamos o blob en verde
    circle(img, Point(x,y),radius,Scalar(0,255,0),2);
}

imshow("Imaxe",img);
waitKey(0);

// Asignamos os parametros do SimpleBlobDetector
SimpleBlobDetector::Params params;

// Cambiamos os limiares
params.minThreshold = 10;
params.maxThreshold = 200;

// Filtramos por Area.
params.filterByArea = true;
params.minArea = 1500;

// Filter by Circularity
params.filterByCircularity = true;
params.minCircularity = 0.1;

// Filtramos por convexidade
params.filterByConvexity = true;
params.minConvexity = 0.87;

// Filtramos por Inercia
params.filterByInertia = true;
params.minInertiaRatio = 0.01;

detector = SimpleBlobDetector::create(params);


// ####CONTINUA O CODIGO AQUI PARA APLICAR DE NOVO ESTE DETECTOR CONFIGURADO
return 0;
}
