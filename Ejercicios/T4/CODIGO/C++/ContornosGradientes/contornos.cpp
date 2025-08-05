#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv){

//Lemos os argumentos de entrada
const char* file = argv[1];
Mat image = imread(file);
if(image.empty())
  {
    cout << "Erro lendo a imaxe" << file<< endl;
    return -1;
  }

Mat imageCopy = image.clone();

Mat imageGray;
// Convertemos a imaxe de gris
cvtColor(image,imageGray,COLOR_BGR2GRAY);

imshow("imaxe",imageGray);
waitKey(0);
// Atopamos todos os contornos na imaxe
vector<vector<Point> > contours;
vector<Vec4i> hierarchy;

findContours(imageGray, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);

cout << "Numero de contornos atopados = " << contours.size();

drawContours(image, contours, -1, Scalar(0,255,0), 6);

imshow("imaxe",image);
waitKey(0);

// Atopamos os contornos externos
findContours(imageGray, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
cout << "Numero de contornos atopados = " << contours.size();

image = imageCopy.clone();

// Debuxamos todos os contornos
drawContours(image, contours, -1, Scalar(0,255,0), 3);

imshow("imaxe",image);
waitKey(0);


// Debuxamos so o 3 contorno
// Nota que a numeración non indica a posición na imaxe!

image = imageCopy.clone();
drawContours(image, contours, 2, Scalar(0,255,0), 3);

imshow("imaxe",image);
waitKey(0);


// Atopamos todos os contornos da imaxe
findContours(imageGray, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);
// Debuxamos todos os contornos
drawContours(image, contours, -1, Scalar(0,255,0), 3);

Moments M;
int x,y;
for (size_t i=0; i < contours.size(); i++){
    // Empregamos os momentos do contornos para atopar o centroide
    x = int(M.m10/double(M.m00));
    y = int(M.m01/double(M.m00));

    // Marcamos o centro
    circle(image, Point(x,y), 10, Scalar(255,0,0), -1);
}

imshow("imaxe",image);
waitKey(0);


for (size_t i=0; i < contours.size(); i++){
    // Empregamos os momentos do contornos para atopar o centroide
    M = moments(contours[i]);
    x = int(M.m10/double(M.m00));
    y = int(M.m01/double(M.m00));

    // marcamos o centroide
    circle(image, Point(x,y), 10, Scalar(255,0,0), -1);

    // marcamos o numero do contorno
    putText(image, to_string(i+1), Point(x+40,y-10), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255),2);
}

imageCopy = image.clone();

imshow("imaxe",image);
waitKey(0);

double area;
double perimeter;
for (size_t i=0; i < contours.size(); i++){
    area = contourArea(contours[i]);
    perimeter = arcLength(contours[i],true);
    cout << "Contorno #" << i+1 << " ten un area = " << area << " e perimetro = " << perimeter << endl;
}

image = imageCopy.clone();
Rect rect;
for (size_t i=0; i < contours.size(); i++){
    // rectangulo vertical
    rect = boundingRect(contours[i]);
    rectangle(image, rect, Scalar(255,0,255), 2);
}
imshow("imaxe",image);
waitKey(0);


image = imageCopy.clone();
RotatedRect rotrect;
Point2f rect_points[4];
Mat boxPoints2f,boxPointsCov;

for (size_t i=0; i < contours.size(); i++){
    // Rectangulo rotado
    rotrect = minAreaRect(contours[i]);
    boxPoints(rotrect, boxPoints2f);
    boxPoints2f.assignTo(boxPointsCov,CV_32S);
    polylines(image, boxPointsCov, true, Scalar(0,255,255), 2);
}

imshow("imaxe",image);
waitKey(0);

image = imageCopy.clone();
Point2f center;
float radius;
for (size_t i=0; i < contours.size(); i++){
    // axustamos a un circulo
    minEnclosingCircle(contours[i],center,radius);
    circle(image,center,radius, Scalar(125,125,125), 2);
}

imshow("imaxe",image);
waitKey(0);


image = imageCopy.clone();
RotatedRect rellipse;
for (size_t i=0; i < contours.size(); i++){
    // Axustamos a unha elipse. Isto so se pode facer
    // cando o noso contorno ten un minimo de 5 puntos
    if (contours[i].size() < 5)
        continue;
    rellipse = fitEllipse(contours[i]);
    ellipse(image, rellipse, Scalar(255,0,125), 2);
}
imshow("imaxe",image);
waitKey(0);

return 0;
}
