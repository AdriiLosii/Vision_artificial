#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

// compara rectangulos
bool rectAreaComparator(Rect &r1, Rect &r2)
{
  return r1.area() < r2.area();
}

// colores para visualizacion
Scalar blue(255,128,0);
Scalar red(64,64,255);

int main(int, char**)
{

  // Inicializamos o Ho<<g
  HOGDescriptor hog(
                    Size(64, 128), //winSize
                    Size(16, 16),  //blocksize
                    Size(8, 8),    //blockStride,
                    Size(8, 8),    //cellSize,
                    9,             //nbins,
                    0,             //derivAperture,
                    -1,            //winSigma,
                    HOGDescriptor::HistogramNormType (0),             //histogramNormType,
                    0.2,           //L2HysThresh,
                    1,             //gammal correction,
                    64,            //nlevels=64
                    0);            //signedGradient
  // Inicializamos o detector de peons.
  vector< float > svmDetector = hog.getDefaultPeopleDetector();
  hog.setSVMDetector(svmDetector);
  float hitThreshold = 0.0;
  Size winStride = Size(8, 8);
  Size padding = Size(32, 32);
  float scale = 1.05;
  float finalThreshold = 1;
  bool useMeanshiftGrouping = 0;
  vector<double> weights;


  // vector para almacenar os rectangulos das persoas
  std::vector<Rect> objects;

  // cargamos o video
  VideoCapture cap("../../data/boy-walking.mp4");

  // confirmamos se lemos video
  if (!cap.isOpened())
  {
    cerr << "Non atopamos o video" << endl;
    return EXIT_FAILURE;
  }

  // Variables para almacenar os fotogramos
  Mat frame, frameDisplayDetection, frameDisplay, output;

  // Specify Kalman filter type
  int type = CV_32F;

// Inicializamos o filtro de Kalman.
// Estado interno de 6 elementos (x, y, width, vx, vy, vw)
// Medidas ten  3 elementos (x, y, width ).
// Nota: Height = 2 x width, logo non é parte do estado do sistema
  KalmanFilter KF(6, 3, 0, type);

  /*
   Matriz de transicion
   [
     1, 0, 0, dt, 0,  0,
     0, 1, 0, 0,  dt, 0,
     0, 0, 1, 0,  0,  dt,
     0, 0, 0, 1,  0,  0,
     0, 0, 0, 0,  1,  0,
     0, 0, 0, dt, 0,  1
   ]

   porqe

   x = x + vx * dt
   y = y + vy * dt
   w = y + vw * dt

   vx = vx
   vy = vy
   vw = vw

   */
  setIdentity(KF.transitionMatrix);

  /*
   Matriz de medidas
   [
    1, 0, 0, 0, 0,  0,
    0, 1, 0, 0, 0,  0,
    0, 0, 1, 0, 0,  0,
   ]

  porque estamos detectando solo x, y y w.
    Estas cantidades se actualizan.

  */
  setIdentity(KF.measurementMatrix);

  // Variable to store detected x, y e w
  Mat measurement = Mat::zeros(3, 1, type);

  // Variables para almacenar el objeto detectado y el objeto rastreado
  Rect objectTracked, objectDetected;

  // Variables para almacenar los resultados de la predicción y la actualización .
  Mat updatedMeasurement, predictedMeasurement;

  // Variable para indicar que se actualizó la medición
  bool measurementWasUpdated = false;

  // Variable de tiempo
  double ticks, preTicks;

  // Leer fotogramas hasta que se detecte el objeto por primera vez
  while(cap.read(frame))
  {

    // Detectar objeto
    hog.detectMultiScale(frame, objects, weights, hitThreshold, winStride, padding,
                         scale, finalThreshold, useMeanshiftGrouping);

    // Temporizador de actualización
    ticks = (double) cv::getTickCount();


    if(objects.size() > 0 )
    {
      // primeiro obxecto detectado
      objectDetected = *std::max_element(objects.begin(), objects.end(), rectAreaComparator);

      // actualiamos as medidas
      measurement.at<float>(0) = objectDetected.x;
      measurement.at<float>(1) = objectDetected.y;
      measurement.at<float>(2) = objectDetected.width;

      // Actualizar estado. Tenga en cuenta que x, y, w se establecen en valores medidos.
       // vx = vy = vw porque todavía no tenemos idea de las velocidades.
      KF.statePost.at<float>(0) = measurement.at<float>(0);
      KF.statePost.at<float>(1) = measurement.at<float>(1);
      KF.statePost.at<float>(2) = measurement.at<float>(2);
      KF.statePost.at<float>(3) = 0;
      KF.statePost.at<float>(4) = 0;
      KF.statePost.at<float>(5) = 0;

      // Establecer valores diagonales para matrices de covarianza.
       // processNoiseCov es Q
      setIdentity(KF.processNoiseCov, Scalar::all(1e-2));

      // measurementNoiseCov e R
      setIdentity(KF.measurementNoiseCov, Scalar::all(1e-2));
      break;
    }
  }

  // dt para matriz de transición
  double dt = 0;

  // Generador de números aleatorios para seleccionar frames aleatoriamente para actualizar
  RNG rng( 0xFFFFFFFF );

  // Bucle sobre todos los fotogramas
  while(cap.read(frame))
  {
    // Variable para mostrar el resultado del seguimiento
    frameDisplay = frame.clone();

    // Variable para mostrar el resultado de la detección
    frameDisplayDetection = frame.clone();

    // Actualiza dt para la matriz de transición.
     // dt = tiempo transcurrido.

    preTicks = ticks;
    ticks = (double) cv::getTickCount();
    dt = (ticks - preTicks) / cv::getTickFrequency();

    KF.transitionMatrix.at<float>(3) = dt;
    KF.transitionMatrix.at<float>(10) = dt;
    KF.transitionMatrix.at<float>(17) = dt;

    // Paso de predicción del filtro de Kalman
    predictedMeasurement = KF.predict();


    // Borrar objetos detectados en el cuadro anterior.
    objects.clear();

    // Detectar objetos en el marco actual
    hog.detectMultiScale(frame, objects, weights, hitThreshold, winStride, padding,
                         scale, finalThreshold, useMeanshiftGrouping);

    //Encuentra el objeto más grande
    objectDetected = *std::max_element(objects.begin(), objects.end(), rectAreaComparator);

    // Mostrar rectángulo detectado
    rectangle(frameDisplayDetection, objectDetected, red, 2, 4);

// Actualizaremos las medidas el 15% del tiempo.
     // Los marcos se eligen al azar.
    bool update = rng.uniform( 0.0, 1.0) < 0.15;

    if (update)
    {
      // Paso de actualización del filtro de Kalman
      if(objects.size() > 0 )
      {
      // Copia x, y, w del rectángulo detectado
        measurement.at<float>(0) = objectDetected.x;
        measurement.at<float>(1) = objectDetected.y;
        measurement.at<float>(2) = objectDetected.width;

        // Realice el paso de actualización de Kalman
        updatedMeasurement = KF.correct(measurement);
        measurementWasUpdated = true;
      }
      else
      {
        // Medida no actualizada porque no se detectó ningún objeto
        measurementWasUpdated = false;
      }

    }
    else
    {
      // Medida no actualizada
      measurementWasUpdated = false;

    }

    if(measurementWasUpdated)
    {
      // Utilice la medición actualizada si la medición se actualizó
      objectTracked = Rect(updatedMeasurement.at<float>(0), updatedMeasurement.at<float>(1),updatedMeasurement.at<float>(2),2 * updatedMeasurement.at<float>(2));
    }
    else
    {
      // Si la medición no se actualizó, use valores predichos.
      objectTracked = Rect(predictedMeasurement.at<float>(0), predictedMeasurement.at<float>(1),predictedMeasurement.at<float>(2),2 * predictedMeasurement.at<float>(2));
    }

    //Dibujar objeto rastreado
    rectangle(frameDisplay, objectTracked, blue, 2, 4);

    // Texto que indica seguimiento o detección.
    putText(frameDisplay,"Tracking", Point(20,40),FONT_HERSHEY_SIMPLEX, 0.75, blue, 2);
    putText(frameDisplayDetection,"Deteccion", Point(20,40),FONT_HERSHEY_SIMPLEX, 0.75, red, 2);

    // Concatenar el resultado detectado y el resultado rastreado verticalmente
    vconcat(frameDisplayDetection, frameDisplay, output);

    // Mostrar resultado.
    imshow("Obxecto seguido", output);
    int key = waitKey(5);
    // Romper si se presiona ESC
    if ( key == 27 )
    {
      break;
    }

  }
  return EXIT_SUCCESS;
}
