#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

using namespace cv;
using namespace std;

vector<string> trackerTypes = {"BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"}; 

// creamos o tracker
Ptr<Tracker> createTrackerByName(string trackerType) 
{
  Ptr<Tracker> tracker;
  if (trackerType ==  trackerTypes[0])
    tracker = TrackerBoosting::create();
  else if (trackerType == trackerTypes[1])
    tracker = TrackerMIL::create();
  else if (trackerType == trackerTypes[2])
    tracker = TrackerKCF::create();
  else if (trackerType == trackerTypes[3])
    tracker = TrackerTLD::create();
  else if (trackerType == trackerTypes[4])
    tracker = TrackerMedianFlow::create();
  else if (trackerType == trackerTypes[5])
    tracker = TrackerGOTURN::create();
  else if (trackerType == trackerTypes[6])
    tracker = TrackerMOSSE::create();
  else if (trackerType == trackerTypes[7])
    tracker = TrackerCSRT::create();
  else {
    cout << "Nome de tracker incorrecta" << endl;
    cout << "Os tracker disponhibles son: " << endl;
    for (vector<string>::iterator it = trackerTypes.begin() ; it != trackerTypes.end(); ++it)
      std::cout << " " << *it << endl;
  }
  return tracker;
}

// Enche o vector con cores aleatorias
void getRandomColors(vector<Scalar> &colors, int numColors)
{
  RNG rng(0);
  for(int i=0; i < numColors; i++)
    colors.push_back(Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255))); 
}

int main(int argc, char * argv[]) 
{
  cout << "Tracker por defecto CSRT" << endl;
  cout << "Os tracker disponhibles son:" << endl;
  for (vector<string>::iterator it = trackerTypes.begin() ; it != trackerTypes.end(); ++it)
    std::cout << " " << *it << endl;
  
  // inicializa tipo de rastreador. Cambia isto para probar diferentes rastreadores.
  string trackerType = "CSRT";

  // establecer valores predeterminados para o algoritmo de seguimento e o vídeo
  string videoPath = "../../data/cars_road.mp4";
  
  // Inicialice MultiTracker cun algoritmo de seguimento
  vector<Rect> bboxes;

  // crear un obxecto de captura de vídeo para ler vídeos
  cv::VideoCapture cap(videoPath);
  Mat frame;

  // saír se non pode ler o ficheiro de vídeo
  if(!cap.isOpened()) 
  {
    cout << "Error ao ler o video " << videoPath << endl;
    return -1;
  }

  // ler o primeiro cadro
  cap >> frame;
  
   // debuxa caixas delimitadoras sobre obxectos
   // O comportamento predeterminado de selectROI é debuxar un cadro comezando polo centro
   // cando fromCenter está definido como falso, podes debuxar un cadro comezando pola esquina superior esquerda
  bool showCrosshair = true;
  bool fromCenter = false;
  cout << "\n==========================================================\n";
  cout << "OpenCV di que prema c para cancelar o proceso de selección de obxectos" << endl;
  cout << "Non funciona. Preme Escapar para saír do proceso de selección" << endl;
  cout << "\n==========================================================\n";
  cv::selectROIs("MultiTracker", frame, bboxes, showCrosshair, fromCenter);
  
  // saír se non hai obxectos que rastrexar
  if(bboxes.size() < 1)
    return 0;
  
  vector<Scalar> colors;  
  getRandomColors(colors, bboxes.size()); 
  
  // Crear multitracker
  Ptr<MultiTracker> multiTracker = cv::MultiTracker::create();

  // inicializar multitracker
  for(int i=0; i < bboxes.size(); i++)
    multiTracker->add(createTrackerByName(trackerType), frame, Rect2d(bboxes[i]));  
  
  // procesar vídeo e rastrexar obxectos
  cout << "\n==========================================================\n";
  cout << "Comezou o seguimento, preme ESC para saír." << endl;
  while(cap.isOpened()) 
  {
    // obter fotograma do vídeo
    cap >> frame;
  
    // parar o programa se chega ao final do vídeo
    if (frame.empty()) break;

    //actualiza o resultado do seguimento cun novo marco
    multiTracker->update(frame);

    // debuxa obxectos rastrexados
    for(unsigned i=0; i<multiTracker->getObjects().size(); i++)
    {
      rectangle(frame, multiTracker->getObjects()[i], colors[i], 2, 1);
    }
  
    // mostrar marco
    imshow("MultiTracker", frame);
    
    // saír no botón x
    if  (waitKey(1) == 27) break;
    
   }

 
}
