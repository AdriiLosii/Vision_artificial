import os
import sys
import cv2
import random
import numpy as np

# Para detectertar a cara maxima no array de faces(x,y,w,h)
def maxRectArea(rects):
  area = 0
  maxRect = rects[0].copy()
  for rect in rects:
    x, y, w, h = rect.ravel()
    if w*h > area:
      area = w*h
      maxRect = rect.copy()
  maxRect = maxRect[:, np.newaxis]
  return maxRect

blue = (255, 0, 0)
red = (0, 0, 255)

if __name__ == '__main__' :

  # Inicializamos o detector HoG
  winSize = (64, 128)
  blockSize = (16, 16)
  blockStride = (8, 8)
  cellSize = (8, 8)
  nbins = 9
  derivAperture = 1
  winSigma = -1
  histogramNormType = 0
  L2HysThreshold = 0.2
  gammaCorrection = True
  nlevels = 64
  signedGradient = False

  hog = cv2.HOGDescriptor(winSize, blockSize, blockStride,
                          cellSize, nbins, derivAperture,
                          winSigma, histogramNormType, L2HysThreshold,
                          gammaCorrection, nlevels, signedGradient)

  svmDetector = cv2.HOGDescriptor_getDefaultPeopleDetector()
  hog.setSVMDetector(svmDetector)

  # cargamos o video
  filename = "../data/boy-walking.mp4"
  cap = cv2.VideoCapture(filename)

  # Confirmamonos se abrimos o video
  if not cap.isOpened():
    print("Incapaz de ler o video")
    sys.exit(1)

  # Variables para almacenar os fotogramos
  frameDisplay = []
  # Inicializamos o filtro de Kalman.
  # Estado interno de 6 elementos (x, y, width, vx, vy, vw)
  # Medidas ten  3 elementos (x, y, width ).
  # Nota: Height = 2 x width, logo non é parte do estado do sistema
  KF = cv2.KalmanFilter(6, 3, 0)

  # La matriz de transición tiene la forma
  # [
  #   1, 0, 0, dt, 0,  0,
  #   0, 1, 0, 0,  dt, 0,
  #   0, 0, 1, 0,  0,  dt,
  #   0, 0, 0, 1,  0,  0,
  #   0, 0, 0, 0,  1,  0,
  #   0, 0, 0, dt, 0,  1
  # ]
  # porque
  # x = x + vx * dt
  # y = y + vy * dt
  # w = y + vw * dt

  # vx = vx
  # vy = vy
  # vw = vw
  KF.transitionMatrix = cv2.setIdentity(KF.transitionMatrix)

  # La matriz de medición tiene la forma
  # [
  #  1, 0, 0, 0, 0,  0,
  #  0, 1, 0, 0, 0,  0,
  #  0, 0, 1, 0, 0,  0,
  # ]
  # porque estamos detectando solo x, y e w.
   # Estas cantidades se actualizan.
  KF.measurementMatrix = cv2.setIdentity(KF.measurementMatrix)

  # Variable para almacenar x, y e w detectadas.
  measurement = np.zeros((3, 1), dtype=np.float32)
  # Variables para almacenar el objeto detectado y el objeto rastreado
  objectTracked = np.zeros((4, 1), dtype=np.float32)
  objectDetected = np.zeros((4, 1), dtype=np.float32)

  # Variables para almacenar los resultados de la predicción y la actualización.
  updatedMeasurement = np.zeros((3, 1), dtype=np.float32)
  predictedMeasurement = np.zeros((6, 1), dtype=np.float32)

  # Variable para indicar que se actualizó la medición
  measurementWasUpdated = False

  # Variable de tiempo
  ticks = 0
  preTicks = 0

  # Leer fotogramas hasta que se detecte el objeto por primera vez
  success = True
  while success:
    sucess, frame = cap.read()
    objects, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(32, 32),
                                            scale=1.05, hitThreshold=0, finalThreshold=1,
                                            useMeanshiftGrouping=False)

    # Temporizador de actualización
    ticks = cv2.getTickCount()

    if len(objects) > 0:
      # Copiar los valores máximos del área facial en el filtro de Kalman
      objectDetected = maxRectArea(objects)
      measurement = objectDetected[:3].astype(np.float32)

      # Estado de actualización. Tenga en cuenta que x, y, w se establecen en valores medidos.
       # vx = vy = vw porque todavía no tenemos idea de las velocidades.
      KF.statePost[0:3, 0] = measurement[:, 0]
      KF.statePost[3:6] = 0.0

      # Establecer valores diagonales para matrices de covarianza.
       # processNoiseCov es Q
      KF.processNoiseCov = cv2.setIdentity(KF.processNoiseCov, (1e-2))
      KF.measurementNoiseCov = cv2.setIdentity(KF.measurementNoiseCov, (1e-2))
      break

  # dt para matriz de transición
  dt = 0.0
  # Generador de números aleatorios para seleccionar cuadros aleatoriamente para actualizar
  random.seed(42)

  # Bucle sobre el resto de los fotogramas
  while True:
    success, frame = cap.read()
    if not success:
      break

    # Variable para mostrar el resultado del seguimiento
    frameDisplay = frame.copy()
    # Variable para mostrar el resultado de la detección
    frameDisplayDetection = frame.copy()

    # Actualice dt para la matriz de transición.
     # dt = tiempo transcurrido.
    preTicks = ticks
    ticks = cv2.getTickCount()
    dt = (ticks - preTicks) / cv2.getTickFrequency()

    KF.transitionMatrix[0, 3] = dt
    KF.transitionMatrix[1, 4] = dt
    KF.transitionMatrix[2, 5] = dt

    predictedMeasurement = KF.predict()

  # Detectar objetos en el marco actual
    objects, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(32, 32),
                                            scale=1.05, hitThreshold=0, finalThreshold=1,
                                            useMeanshiftGrouping=False)
    if len(objects) > 0:
    # Encuentra el objeto más grande
      objectDetected = maxRectArea(objects)

    # Mostrar rectángulo detectado
      x1, y1, w1, h1 = objectDetected.ravel()
      cv2.rectangle(frameDisplayDetection, (x1, y1), (x1+w1, y1+h1), red, 2, 4)

    # Actualizaremos las mediciones el 15% del tiempo.
     # Los marcos se eligen al azar.
    update = random.randint(0, 100) < 15

    if update:
      # Paso de actualización del filtro de Kalman
      if len(objects) > 0:
      # Copie x, y, w del rectángulo detectado
        measurement = objectDetected[0:3].astype(np.float32)

      # Realice el paso de actualización de Kalman
        updatedMeasurement = KF.correct(measurement)
        measurementWasUpdated = True
      else:
      # Medida no actualizada porque no se detectó ningún objeto
        measurementWasUpdated = False
    else:
      # Medida no actualizada
      measurementWasUpdated = False

    if measurementWasUpdated:
    # Utilice la medición actualizada si la medición se actualizo
      objectTracked[0:3, 0] = updatedMeasurement[0:3, 0].astype(np.int32)
      objectTracked[3, 0] = 2*updatedMeasurement[2, 0].astype(np.int32)
    else:
    # Se a medición non se actualizou, use os valores previstos.
      objectTracked[0:3, 0] = predictedMeasurement[0:3, 0].astype(np.int32)
      objectTracked[3, 0] = 2*predictedMeasurement[2, 0].astype(np.int32)

    # Debuxa o obxecto rastrexado
    x2, y2, w2, h2 = objectTracked.ravel()
    cv2.rectangle(frameDisplay, (x2, y2), (x2+w2, y2+h2), blue, 2, 4)

    # Texto que indica seguimento ou detección.
    cv2.putText(frameDisplay, "Tracking", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, blue, 2)
    cv2.putText(frameDisplayDetection, "Detecion", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, red, 2)

    # Concatena verticalmente o resultado detectado e o resultado seguido
    output = np.concatenate((frameDisplayDetection, frameDisplay), axis=0)

    # Mostrar resultado.
    cv2.imshow("seguidor de obxectos", output)

    # Pausa se se preme ESC
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()
cv2.destroyAllWindows()
