import cv2,sys,os,time,dlib
import numpy as np
import faceBlendCommon as fbc

FACE_DOWNSAMPLE_RATIO = 2
RESIZE_HEIGHT = 360

predictions2Label = {0:"Sen gafas", 1:"Con gafas"}

def svmPredict(model, samples):
  return model.predict(samples)[1].ravel()

def prepareData(data):
  featureVectorLength = len(data[0])
  features = np.float32(data).reshape(-1,featureVectorLength)
  return features

def computeHOG(hog, data):
  hogData = []
  for image in data:
    hogFeatures = hog.compute(image)
    hogData.append(hogFeatures)

  return hogData


if __name__ == '__main__':

  # cargamos os modelos para a estimación de cara e pose.
  modelPath = "../data/models/shape_predictor_68_face_landmarks.dat"
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(modelPath)

  # Inicialiamos parametros para HoG
  winSize = (96,32)
  blockSize = (8,8)
  blockStride = (8,8)
  cellSize = (4,4)
  nbins = 9
  derivAperture = 0
  winSigma = 4.0
  histogramNormType = 1
  L2HysThreshold =  2.0000000000000001e-01
  gammaCorrection = 1
  nlevels = 64

  hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,
                          derivAperture,winSigma,histogramNormType,
                          L2HysThreshold,gammaCorrection,nlevels,1)

  # Posteriormente voltaremos a cargar o modelos
  savedModel = cv2.ml.SVM_load("results/eyeGlassClassifierModel.yml")
  # iniciamos a web cam
  cap = cv2.VideoCapture(0)

  # comprobamos se abre a canle da camara
  if (cap.isOpened()== False):
    print("error abrindo a canle da camara")

  while(1):
    try:
      t = time.time()
      # lemos o frame
      ret, frame = cap.read()
      height, width = frame.shape[:2]
      IMAGE_RESIZE = np.float32(height)/RESIZE_HEIGHT
      frame = cv2.resize(frame,None,
                         fx=1.0/IMAGE_RESIZE,
                         fy=1.0/IMAGE_RESIZE,
                         interpolation = cv2.INTER_LINEAR)

      landmarks = fbc.getLandmarks(detector, predictor, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), FACE_DOWNSAMPLE_RATIO)
      print("tempo para detectar as marcas : {}".format(time.time() - t))
      #collemos o punto para o detector de marcar
      x1 = landmarks[0][0]
      x2 = landmarks[16][0]
      y1 = min(landmarks[24][1], landmarks[19][1])
      y2 = landmarks[29][1]

      cropped = frame[y1:y2,x1:x2,:]
      cropped = cv2.resize(cropped,(96, 32), interpolation = cv2.INTER_CUBIC)

      testHOG = computeHOG(hog, np.array([cropped]))
      testFeatures = prepareData(testHOG)
      predictions = svmPredict(savedModel, testFeatures)
      frameClone = np.copy(frame)
      cv2.putText(frameClone, "Predicion = {}".format(predictions2Label[int(predictions[0])]), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
      print("Predicion = {}".format(predictions2Label[int(predictions[0])]))

      cv2.imshow("Frame orixinal", frameClone)
      cv2.imshow("Zona dos ollos", cropped)
      if cv2.waitKey(1) & 0xFF == 27:
        break

      print("Tempo total : {}".format(time.time() - t))
    except Exception as e:
      frameClone = np.copy(frame)
      cv2.putText(frameClone, "Cara NON detectada", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
      cv2.imshow("Frame Orixinal", frameClone)
#      cv2.imshow("Ollo", cropped)
      #Pulsa escape para sair
      if cv2.waitKey(1) & 0xFF == 27:
        break
      print(e)

  cv2.destroyAllWindows()
