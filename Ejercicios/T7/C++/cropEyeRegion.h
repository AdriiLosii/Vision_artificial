#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

Mat getCroppedEyeRegion(Mat targetImage){
	String faceCascadePath = "../../data/models/haarcascade_frontalface_default.xml";

	CascadeClassifier faceCascade;
	if( !faceCascade.load( faceCascadePath ) ){ printf("--(!)Erro cargando o modelo de Haar\n"); };

	Mat targetImageGray;

	cvtColor(targetImage,targetImageGray,COLOR_BGR2GRAY);

	vector<Rect> faces;
	faceCascade.detectMultiScale( targetImageGray, faces, 1.3, 5);

	int x = faces[0].x;
	int y = faces[0].y;
	int w = faces[0].width;
	int h = faces[0].height;

	Mat face_roi = targetImage(Range(y,y+h),Range(x,x+w));
	imwrite("../results/face_roi.png",face_roi);
	int face_height, face_width;
	face_height = face_roi.size().height;
	face_width = face_roi.size().width;

	// Aplicamos a formula heuristica para atopar
	int eyeTop = (int)(1.0/6.0 * face_height);
	int eyeBottom = (int)(3.0/6.0*face_height);

	cout << "Altura da rexion de ollos : " << eyeTop << "," << eyeBottom << endl;

	Mat eye_roi = face_roi(Range(eyeTop,eyeBottom),Range(0,face_width));

	// Redimensionamos o tamanho dos ollos ao fixado na base de datos de 96x32
	Mat cropped;
	resize(eye_roi,cropped,Size(96,32));

	return cropped;
}
