#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>


using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    // Lista de trackers OpenCV 4

    string trackerTypes[8] = {"BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "CSRT", "MOSSE"};
    // vector <string> trackerTypes(types, std::end(types));

    //Crea un rastreador
    string trackerType = trackerTypes[6];

    Ptr<Tracker> tracker;

        if (trackerType == "BOOSTING")
            tracker = TrackerBoosting::create();
        else if (trackerType == "MIL")
            tracker = TrackerMIL::create();
        else if (trackerType == "KCF")
            tracker = TrackerKCF::create();
        else if (trackerType == "TLD")
            tracker = TrackerTLD::create();
        else if (trackerType == "MEDIANFLOW")
            tracker = TrackerMedianFlow::create();
        else if (trackerType == "GOTURN")
            tracker = TrackerGOTURN::create();
        else if (trackerType == "CSRT")
            tracker = TrackerCSRT::create();
        else if (trackerType == "MOSSE")
            tracker = TrackerMOSSE::create();
        else
        {
            cout << "Tracker invalida" << endl;
            cout << "Traker disponiles :" << endl;
            for ( int i = 0; i < sizeof(trackerTypes)/sizeof(trackerTypes[0]); i++)
                cout << i << " : " << trackerTypes[i] << endl;
            return -1;
        }


    // Ler o video
    VideoCapture video("../../data/hockey.mp4");

    // Sae se o vídeo non está aberto
    if(!video.isOpened())
    {
        cout << "Non podo ler o video" << endl;
        return 1;

    }

    // Ler o primeiro cadro
    Mat frame;
    bool ok = video.read(frame);

    // Definir caixa delimitadora inicial
    Rect2d bbox(204, 131, 97, 222);

    // Descomenta a liña de abaixo para seleccionar un cadro delimitador diferente
    // bbox = selectROI(frame, false);
    cout << "Caixa delimitadora inicial : " << bbox << endl;

    // Mostra un cadro delimitador.
    rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
    imshow("Tracking", frame);

    tracker->init(frame, bbox);

    while(video.read(frame))
    {

        // inicio de tempo
        double timer = (double)getTickCount();

        // Actualiza o resultado do seguimento
        bool ok = tracker->update(frame, bbox);

        // Calcular fotogramas por segundo (FPS)
        float fps = getTickFrequency() / ((double)getTickCount() - timer);


        if (ok)
        {
            // Seguimento exitoso: debuxa o obxecto rastrexado
            rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
        }
        else
        {
            // Detectouse un erro no seguimento.
            putText(frame, "fallo deTracking detected", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
        }

        // Mostra o tipo de rastreador no frame
        putText(frame, trackerType + " Tracker", Point(100,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);

        // Visualiza FPS sobre o fotograma
        putText(frame, "FPS : " + std::to_string(fps), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);

        // Visualiza o fotograma
        imshow("Tracking", frame);

        // ESC para sair
        int k = waitKey(1);
        if(k == 27)
        {
            break;
        }
    }
}
