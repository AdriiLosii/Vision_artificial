import os
import glob
import cv2
import numpy as np

# Inicialozamos parametros HoG
winSize = (64, 128)
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9
derivAperture = 1
winSigma = -1
histogramNormType = cv2.HOGDESCRIPTOR_L2HYS
L2HysThreshold = 0.2
gammaCorrection = True
nlevels = 64
signedGradient = False

# HOG
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride,
                        cellSize, nbins, derivAperture,
                        winSigma, histogramNormType, L2HysThreshold,
                        gammaCorrection, nlevels, signedGradient)

# cargamos o modelo entrenado
model = cv2.ml.SVM_load('../data/models/pedestrian.yml')

sv = model.getSupportVectors()
rho, aplha, svidx = model.getDecisionFunction(0)
svmDetectorTrained = np.zeros(sv.shape[1] + 1, dtype=sv.dtype)
svmDetectorTrained[:-1] = -sv[:]
svmDetectorTrained[-1] = rho

# Instanciamos o SVMDetector entrenado por nos en HOG
hog.setSVMDetector(svmDetectorTrained)

# HOG de Opencv para detectar peons
hogDefault = cv2.HOGDescriptor(winSize, blockSize, blockStride,
                               cellSize, nbins, derivAperture,
                               winSigma, histogramNormType, L2HysThreshold,
                               gammaCorrection, nlevels, signedGradient)
svmDetectorDefault = cv2.HOGDescriptor_getDefaultPeopleDetector()
hogDefault.setSVMDetector(svmDetectorDefault)

# Lemos as imaxes dos peons dende disco
imageDir = '../data/images/pedestrians'
imagePaths = glob.glob(os.path.join(imageDir, '*.jpg'))

# Normalizamos todas as imaxes a mesma altura
finalHeight = 800.0

for imagePath in imagePaths:
  print('procesando: {}'.format(imagePath))

  # Lemos a imaxe
  im = cv2.imread(imagePath, cv2.IMREAD_COLOR)

  # redimensionamos a imaxe a altura de finalHeight
  scale = finalHeight / im.shape[0]
  im = cv2.resize(im, None, fx=scale, fy=scale)

  # detectMultiScale co detector entrenado por nos
  bboxes, weights = hog.detectMultiScale(im, winStride=(8, 8), padding=(32, 32),
                                         scale=1.05, finalThreshold=2,
                                         hitThreshold=1.0)

  # detectMultiScale empregando o detector por defecto de OpenCV
  bboxes2, weights2 = hogDefault.detectMultiScale(im, winStride=(8, 8), padding=(32, 32),
                                                  scale=1.05, finalThreshold=2,
                                                  hitThreshold=0)

  # Imprimimos os peons detectados
  if len(bboxes) > 0:
    print('Detector entrenado :: peon detectado: {}'.format(bboxes.shape[0]))
  if len(bboxes2) > 0:
    print('Detector OpenCV :: peon detectado: {}'.format(bboxes2.shape[0]))

  # Debuxamos os rectangulos contedores
  # Roxo = detector OpenCV, verde = Detector entrenado
  for bbox in bboxes:
    x1, y1, w, h = bbox
    x2, y2 = x1 + w, y1 + h
    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), thickness=3, lineType=cv2.LINE_AA)

  for bbox in bboxes2:
    x1, y1, w, h = bbox
    x2, y2 = x1 + w, y1 + h
    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), thickness=3, lineType=cv2.LINE_AA)

  # Visualozamos o resultado final
  cv2.imshow("imaxe",im)
  cv2.waitKey(0)
  # Slavamos a disco
  imResultPath = os.path.join('results', os.path.basename(imagePath))
  cv2.imwrite(imResultPath, im)


