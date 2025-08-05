#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <ctype.h>

using namespace std;
using namespace cv;

// Enche o vector con cores aleatorias
void getRandomColors(vector<Scalar> &colors, int numColors)
{
  RNG rng(0);
  for(int i=0; i < numColors; i++)
    colors.push_back(Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255)));
}

int main(){
string videoFileName =  "../../data/cycle.mp4";
VideoCapture cap(videoFileName);

int width = cap.get(CAP_PROP_FRAME_WIDTH);
int height = cap.get(CAP_PROP_FRAME_HEIGHT);

VideoWriter out ("sparse-output.mp4",VideoWriter::fourcc('M','P','4','V'), 20, Size(width,height));

TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,10,0.03);

// Colle o primeiro fra eme atopa as esquinas nel
Mat old_frame;
cap >> old_frame;

Mat old_gray;
cvtColor(old_frame,old_gray,COLOR_BGR2GRAY);

vector<Point2f> old_points;
vector<uchar> status;
vector<float> err;
vector<Point2f> new_points;

vector<Point2f> good_new;
vector<Point2f> good_old;

vector<Scalar> colors;

Point2f pt1, pt2;

goodFeaturesToTrack(old_gray,
                   old_points,
                   100,          // maxCorners
                   0.3,          // qualityLevel
                   7,            // minDistance
                   Mat(),        // mask
                   7             // blockSize
                   );

Mat display_frame;

// Crea unha imaxe de máscara para debuxar as pistas
Mat mask = Mat::zeros(old_frame.size().height, old_frame.size().width,CV_8UC3);

int count = 0;

Mat frame, frame_gray;


while (1){
    cap >> frame;
    if (frame.empty())
        cout << "sobre" << endl;
    cvtColor(frame,frame_gray,COLOR_BGR2GRAY);
    count += 1;

    // calcular o fluxo óptico
    calcOpticalFlowPyrLK(old_gray, frame_gray, old_points, new_points,
                        status, err,
                        Size(15,15), // winSize
                        2,           // maxLevel
                        termcrit     // criteria
                        );

    // Limpando os puntos bos dos fotogramas anteriores
    good_new.clear();
    good_old.clear();

    for(int i = 0; i < new_points.size(); i++ )
    {
        if (status[i] == 1){
            good_new.push_back(new_points[i]);
            good_old.push_back(old_points[i]);
        }
    }

    getRandomColors(colors, new_points.size());

    // debuxa as pistas
    for(int j = 0; j < good_new.size(); j++)
    {
        pt1 = good_new[j];
        pt2 = good_old[j];
        line(mask, pt1, pt2, colors[j], 2, LINE_AA);
        circle(frame, pt1, 3, colors[j], -1);
    }

    add(frame,mask,display_frame);
    out.write(display_frame);

    if (count % 5 == 0){
        imshow("Imaxe",display_frame);
	waitKey(10);
    }
    if (count > 50)
        break;

    // Agora actualice o cadro anterior e os_puntos_anteriores
    old_gray = frame_gray.clone();
    old_points.clear();
    old_points = good_new;
}

cap.release();
out.release();

return 0;
}
