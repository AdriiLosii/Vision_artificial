#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(void){
	// Cargamos a imaxe de Musk
	string faceImagePath = "../../../data/images/musk.jpg";
	Mat faceImage = imread(faceImagePath);
	
	// Facemos unha copia
	Mat faceWithGlassesBitwise = faceImage.clone();
	
	// Cargamos a imaxe das gafas con canle alfa
	string glassimagePath = "../../../data/images/sunglass.png";
	Mat glassPNG = imread(glassimagePath,-1);
	
	//Redimensionamos a imaxe para que encaixe sobre as rexión dos ollos
	resize(glassPNG,glassPNG, Size(300,100));
	cout << "Dimension da imaxe = " << glassPNG.size() << endl;
	
	// Separamos as canles de cor e alfa
	Mat glassRGBAChannels[4];
	Mat glassRGBChannels[3];
	split(glassPNG, glassRGBAChannels);
	for (int i = 0; i < 3; i++){
	    // Copiamos as canles dende RGBA a RGB
	    glassRGBChannels[i] = glassRGBAChannels[i];
	}
	Mat glassBGR, glassMask1;
	// Preparamos a imaxe BRG
	
	merge(glassRGBChannels,3,glassBGR);
	// A canle alfa e a cuarta canle da imaxe RGBA 
	glassMask1 = glassRGBAChannels[3];
	imwrite("./sunglassRGB.png",glassBGR);
	imwrite("./sunglassAlpha.png",glassMask1);

	imshow("Canles de cor das gafas", glassBGR);
	imshow("Canle alfa das gafas", glassMask1);
	waitKey(0);
	
	// Conseguimos a rexion dos ollos da imaxe de Musk
	Mat eyeROI = faceWithGlassesBitwise(Range(150,250),Range(140,440));

	// Igualamos as dimensions da mascara das gafas e da imaxe de Musk.
	// Como a imaxe de Musk ten tres canles, creamos tamén unha imaxe de tres canles para
	// a imaxe das gafas
	Mat glassMask;
	Mat glassMaskChannels[] = {glassMask1,glassMask1,glassMask1};
	merge(glassMaskChannels,3,glassMask);
	
	// Invertimos a mascara da rexion dos ollos de Musk
	Mat eye;
	Mat glassMaskNOT;
	bitwise_not(glassMask1, glassMaskNOT);

	Mat eyeChannels[3];
	Mat eyeROIChannels[3];
	Mat maskedGlass;
	Mat eyeRoiFinal;
	
	split(eyeROI,eyeROIChannels);

	for (int i = 0; i < 3; i++)
	{
	    bitwise_and(eyeROIChannels[i], glassMaskNOT, eyeChannels[i]);
	}
	
	merge(eyeChannels,3,eye);
	imwrite("./glassMaskNOT.png",glassMaskNOT);
	

	Mat glassMaskNOTChannels[] = {glassMaskNOT,glassMaskNOT,glassMaskNOT};
	Mat glassMaskNOTMerged;
	merge(glassMaskNOTChannels,3,glassMaskNOTMerged);
	
	bitwise_and(eyeROI, glassMaskNOTMerged, eye);
	
	
	// Enmascaramos a rexion dos ollos
	Mat sunglass;
	Mat sunglassChannels[3];
	Mat glassBGRChannels[3];
	
	split(glassBGR,glassBGRChannels);
	
	for (int i=0; i < 3; i++)
	    bitwise_and(glassBGRChannels[i], glassMask1, sunglassChannels[i]);
	
	merge(sunglassChannels,3,sunglass);
	multiply(glassBGR, glassMask, maskedGlass);
	
	// Comninamos as gafas coa cara
	bitwise_or(eye, sunglass, eyeRoiFinal);

	imwrite("./maskedEyeRegionBitwise.png",eye);
	imwrite("./maskedSunglassRegionBitwise.png",sunglass);
	imwrite("./augmentedEyeAndSunglassBitwise.png",eyeRoiFinal);

	imshow("Mascara da rexion dos ollos",eye);
	imshow("Mascara das gafas",sunglass);
	imshow("Combinacion",eyeRoiFinal);
	waitKey(0);
	
	
	eyeRoiFinal.copyTo(eyeROI);
	imwrite("./withSunglassesBitwise.png",faceWithGlassesBitwise);

	imshow("Musk con gafas de sol",faceWithGlassesBitwise);
	waitKey(0);
	
	return 0;
}
