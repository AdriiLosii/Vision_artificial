// Include librerias
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace std;
using namespace cv;

int main(void){

    // Path a imaxe que queremos ler
    string imageName = "../../../data/images/barco.jpg";


    // cargamos a imaxe na clase Mat
    Mat image;
    image = imread(imageName, IMREAD_COLOR);
    if (image.empty()){
    	cout << "Non puiden ler a imaxe" << endl;
     }

	// Escribimos na imaxe
	Mat imageText = image.clone();
	string text = "Estou estudando Robótica";
	double fontScale = 1.5;
	int fontFace = FONT_HERSHEY_SIMPLEX;
	Scalar fontColor = Scalar(250, 10, 10);
	int fontThickness = 2;

	putText(imageText, text, Point(20, 350), fontFace, fontScale, fontColor, fontThickness, LINE_AA);

	imwrite("./text.jpg",imageText);
	imshow("Debuxando texto na imaxe", imageText);
	waitKey(0);
	
	int pixelHeight = 20;

	// Acha o tammanho da fonte
	fontScale = getFontScaleFromHeight(fontFace, pixelHeight, fontThickness);
	cout << "fontScale = " << fontScale << endl;
	
	Mat imageTextFontScale;
	imageTextFontScale = image.clone();
	putText(imageTextFontScale, text, Point(20, 350), fontFace, fontScale, fontColor, fontThickness, LINE_AA);

	imwrite("./text2.jpg",imageTextFontScale);
	imshow("Texto escrito empregando fontscale",imageTextFontScale);
	waitKey(0);
	
	Mat imageGetTextSize;
	imageGetTextSize = image.clone();
	int imageHeight = imageGetTextSize.rows;
	int imageWidth = imageGetTextSize.cols;

	// achamos a altura e anchura da caixa de texto e da sua liña base
	int baseLine = 0;
	Size textSize = getTextSize(text,fontFace,fontScale,fontThickness, &baseLine);
	int textWidth = textSize.width;
	int textHeight = textSize.height;

	cout << "TextWidth = " << textWidth << ", TextHeight = " << textHeight << ", baseLine = " << baseLine << endl;

	// coordendas da esquina esquerda inferior da caixa de texto
	//  xccordinate sera tal que o texto estea centrao
	int xcoordinate = (imageWidth - textWidth)/2;
	// a coordenada y sera para que a caixa de texto esteña 10 pixels por encima do fondo da imaxe
	int ycoordinate = (imageHeight - baseLine - 10);

	cout << "TextBox esquina inferior esqueda = (" << xcoordinate << "," << ycoordinate << ")" << endl;

	// Debuxamos a caixa de texto
	Scalar canvasColor = Scalar(255, 255, 255);
	Point canvasBottomLeft (xcoordinate,ycoordinate+baseLine);
	Point canvasTopRight (xcoordinate+textWidth, ycoordinate-textHeight);
	
	rectangle(imageGetTextSize, canvasBottomLeft, canvasTopRight, canvasColor, -1);

	cout << "Canvas esquina einferior esqueda = " << canvasBottomLeft << ", dereita superior = " << canvasTopRight << endl;

	// debuxamos a liña base ( para referencia )
	int lineThickness = 2;
	Point lineLeft (xcoordinate, ycoordinate);
	Point lineRight (xcoordinate+textWidth, ycoordinate);
	Scalar lineColor = Scalar(0,255,0);

	line(imageGetTextSize, lineLeft, lineRight, lineColor, lineThickness, LINE_AA);

	// debuxamos o texto
	putText(imageGetTextSize, text, Point(xcoordinate,ycoordinate), fontFace, fontScale, fontColor, fontThickness, LINE_AA);

	imwrite("./text3.jpg",imageGetTextSize);

	imshow("Escribimos na imaxe achando a escala do texto",imageGetTextSize);
	waitKey(0);
	return 0;
}
