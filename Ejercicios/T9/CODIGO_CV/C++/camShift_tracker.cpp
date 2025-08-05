#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(){
string filename = "../../data/face1.mp4";
VideoCapture cap(filename);

Mat frame;
cap >> frame;

// Lemos un frame e atopamos a cara empregando dlib
CascadeClassifier faceCascade;
String faceCascadePath = "../../data/models/haarcascade_frontalface_default.xml";

if( !faceCascade.load( faceCascadePath ) ){ printf("--(!)Erro ao cargar o Haar-cascade\n"); };
Mat frameGray;
cvtColor(frame,frameGray,COLOR_BGR2GRAY);

vector<Rect> faces;
faceCascade.detectMultiScale(frameGray, faces, 1.3, 5);

int x = faces[0].x;
int y = faces[0].y;
int w = faces[0].width;
int h = faces[0].height;

Rect currWindow = Rect((long)x, (long)y, (long)w, (long)h);

Mat roiObject;

// conseguimos a rexion da cara no frame
frame(currWindow).copyTo(roiObject);
Mat hsvObject;
cvtColor(roiObject, hsvObject, COLOR_BGR2HSV);

//obtén a máscara para calcular o histograma do obxecto e tamén elimina o ruído
Mat mask;
inRange(hsvObject, Scalar(0, 50, 50), Scalar(180, 256, 256), mask);

// Divide a imaxe en canles para atopar o histograma
vector<Mat> channels(3);
split(hsvObject, channels);

imshow("Mascara de ROI",mask);
waitKey(0);
imshow("ROI",roiObject);
waitKey(0);
destroyAllWindows();

Mat histObject;

// Inicializa os parámetros para o histograma
int histSize = 180;
float range[] = { 0, 179 };
const float *ranges[] = { range };

//   Busca o histograma e normalízao para que teña valores entre 0 e 255
calcHist( &channels[0], 1, 0, mask, histObject, 1, &histSize, ranges, true, false );
normalize(histObject, histObject, 0, 255, NORM_MINMAX);

// Procesaremos só os 5 primeiros fotogramas
int count=0;
Mat hsv, backProjectImage,frameClone;
while(1)
{
    // Lemos o frame
    cap >> frame;
    if( frame.empty() )
      break;

    // Converter a espazo de cor hsv
    cvtColor(frame, hsv, COLOR_BGR2HSV);
    split(hsv, channels);

    // atopar a imaxe posterior proxectada co histograma obtido anteriormente
    calcBackProject(&channels[0], 1, 0, histObject, backProjectImage, ranges);
    imshow("Imaxe reproxectada",backProjectImage);
    waitKey(0);
    // Calcula a nova xanela usando o cambio medio no marco actual
    RotatedRect rotatedWindow = CamShift(backProjectImage, currWindow, TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));
    // Mostra o frame coa localización da cara rastrexada
     // Obter os vértices de rotatedWindow
    Point2f rotatedWindowVertices[4];
    rotatedWindow.points(rotatedWindowVertices);

    //Mostra o frame coa localización da cara rastrexada
    frameClone = frame.clone();

    rectangle(frameClone, Point(currWindow.x, currWindow.y), Point(currWindow.x+currWindow.width, currWindow.y + currWindow.height), Scalar(255, 0, 0), 2, LINE_AA);
    // Mostra o rectángulo xirado coa información de orientación
    for (int i = 0; i < 4; i++)
      line(frameClone, rotatedWindowVertices[i], rotatedWindowVertices[(i+1)%4], Scalar(0,255,0), 2, LINE_AA);
    imshow("Seguidor de obxectos Mean Shift ",frameClone);
    waitKey(0);

    count += 2;
    if  (count == 10)
        break;
}

cap.release();
return 0;
}
