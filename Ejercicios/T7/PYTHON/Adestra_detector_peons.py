import os
import glob
import cv2
import numpy as np


# returns os path as imaxe dun cartafol
# coas extensions definidas en imgExts
def getImagePaths(folder, imgExts):
  imagePaths = []
  for x in os.listdir(folder):
    xPath = os.path.join(folder, x)
    if os.path.splitext(xPath)[1] in imgExts:
      imagePaths.append(xPath)
  return imagePaths


# Lemos as imaxes dun cartafol
# return a lista de imaxe e etiquetas
def getDataset(folder, classLabel):
  images = []
  labels = []
  imagePaths = getImagePaths(folder, ['.jpg', '.png', '.jpeg'])
  for imagePath in imagePaths:
    # print(imagePath)
    im = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    images.append(im)
    labels.append(classLabel)
  return images, labels

# achas as caracteristicas HOG
def computeHOG(hog, images):
  hogFeatures = []
  for image in images:
    hogFeature = hog.compute(image)
    hogFeatures.append(hogFeature)
  return hogFeatures


# cambiamos formato de HoG para SVM
def prepareData(hogFeatures):
  featureVectorLength = len(hogFeatures[0])
  data = np.float32(hogFeatures).reshape(-1, featureVectorLength)
  return data

#Inicializa SVM 
def svmInit(C, gamma):
  model = cv2.ml.SVM_create()
  model.setGamma(gamma)
  model.setC(C)
  model.setKernel(cv2.ml.SVM_LINEAR)
  model.setType(cv2.ml.SVM_C_SVC)
  model.setTermCriteria((cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER,
                         1000, 1e-3))
  return model


# adestramos o SVM sobre os datos e as etiquetas
def svmTrain(model, samples, labels):
  model.train(samples, cv2.ml.ROW_SAMPLE, labels)


# predecimos as etiquetas
def svmPredict(model, samples):
  return model.predict(samples)[1]


# Avaliamos o modelos
# tendo en conta as predicion e o ground truth
def svmEvaluate(model, samples, labels):
  labels = labels[:, np.newaxis]
  pred = model.predict(samples)[1]
  correct = np.sum((labels == pred))
  err = (labels != pred).mean()
  print('label -- 1:{}, -1:{}'.format(np.sum(pred == 1),
          np.sum(pred == -1)))
  return correct, err * 100


# creamos o directorio
def createDir(folder):
  try:
    os.makedirs(folder)
  except OSError:
    print('{}: xa existe'.format(folder))
  except Exception as e:
    print(e)

#Inicializa HOG 
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

# Inicializa HOG
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride,
                      cellSize, nbins,derivAperture,
                      winSigma, histogramNormType, L2HysThreshold,
                      gammaCorrection, nlevels,signedGradient)

# Flags para seleccionar adestarmento, test e consulta
trainModel = True
testModel = True
queryModel = True

# Para a base de datos do INRIA de personas
rootDir = '../data/images/INRIAPerson/'

# Path para cartafoles de adestramento e test
trainDir = os.path.join(rootDir, 'train_64x128_H96')
testDir = os.path.join(rootDir, 'test_64x128_H96')

# ================================ adestramento =====================
if trainModel:
    # Lemos imaxe pos e neg
    trainPosDir = os.path.join(trainDir, 'posPatches')
    trainNegDir = os.path.join(trainDir, 'negPatches')

    # etiqueta 1 para os positios e -1 para os negativos
    trainPosImages, trainPosLabels = getDataset(trainPosDir, 1)
    trainNegImages, trainNegLabels = getDataset(trainNegDir, -1)

    # comprobamos tamanhos
    print(set([x.shape for x in trainPosImages]))
    print(set([x.shape for x in trainNegImages]))

    # informamos do que lemos no disco
    print('positivo - {}, {} || negativo - {}, {}'
        .format(len(trainPosImages),len(trainPosLabels),
        len(trainNegImages),len(trainNegLabels)))

    # Concatenamos imaxes/etiqueta para positivo/negativo
    trainImages = np.concatenate((np.array(trainPosImages),
                       np.array(trainNegImages)),
                                  axis=0)
    trainLabels = np.concatenate((np.array(trainPosLabels),
                       np.array(trainNegLabels)),
                                  axis=0)
    # Hog 
    hogTrain = computeHOG(hog, trainImages)

    # formato para  SVM
    trainData = prepareData(hogTrain)

    # Comprobamos dimensions
    print('trainData: {}, trainLabels:{}'
            .format(trainData.shape, trainLabels.shape))
    # Creamos obxecto SVM e entrenamos
    model = svmInit(C=0.01, gamma=0)
    svmTrain(model, trainData, trainLabels)
    model.save('../data/models/pedestrian.yml')

# ================================ Test  ===============
if testModel:
    # cargamos dende disco ficheir xml
    model = cv2.ml.SVM_load('../data/models/pedestrian.yml')
    # avaliamos o modelo
    testPosDir = os.path.join(testDir, 'posPatches')
    testNegDir = os.path.join(testDir, 'negPatches')
    
    testPosImages, testPosLabels = getDataset(testPosDir, 1)
    testNegImages, testNegLabels = getDataset(testNegDir, -1)

    # HoG sobre as imaxes
    hogPosTest = computeHOG(hog, np.array(testPosImages))
    testPosData = prepareData(hogPosTest)
    
    # Clasificamos e calcualos TP e FP 
    posCorrect, posError = svmEvaluate(model, testPosData, 
                                       np.array(testPosLabels))

    tp = posCorrect
    fp = len(testPosLabels) - posCorrect
    print('TP: {}, FP: {}, Total: {}, error: {}'
            .format(tp, fp, len(testPosLabels), posError))
    # O mesmo pero para imaxes negativas
    hogNegTest = computeHOG(hog, np.array(testNegImages))
    testNegData = prepareData(hogNegTest)
    negCorrect, negError = svmEvaluate(model, testNegData, 
                                       np.array(testNegLabels))

    # achamos TP e FP
    tn = negCorrect
    fn = len(testNegData) - negCorrect
    print('TN: {}, FN: {}, Total: {}, error: {}'
            .format(tn, fn, len(testNegLabels), negError))
    # achamos Precision e Recall
    precision = tp * 100 / (tp + fp)
    recall = tp * 100 / (tp + fn)
    print('Precision: {}, Recall: {}'.format(precision, recall))

# ================================ Consulta =============================================
# Executar o detector de obxectos nunha imaxe de consulta para atopar peóns
# Cargaremos o modelo de novo e probaremos o modelo
# Isto é só para explicar como cargar un modelo SVM
# Podes usar o modelo directamente dende memoria (xa o temos aí)
if queryModel:
    # cargamos o modelo
    model = cv2.ml.SVM_load('../data/models/pedestrian.yml')
    sv = model.getSupportVectors()
    rho, aplha, svidx = model.getDecisionFunction(0)
    svmDetector = np.zeros(sv.shape[1] + 1, dtype=sv.dtype)
    svmDetector[:-1] = -sv[:]
    svmDetector[-1] = rho

  
    hog.setSVMDetector(svmDetector)

    filename = "../data/images/pedestrians/3.jpg"
    queryImage = cv2.imread(filename, cv2.IMREAD_COLOR)

    #executamos o noso detector nunha imaxe cunha altura fixa
    finalHeight = 800.0
    scale = finalHeight / queryImage.shape[0]
    queryImage = cv2.resize(queryImage, None, fx=scale, fy=scale)

    # detectMultiScale detectará a n niveis de imaxe aumentando a escala
     # e reducindo o tamaño da imaxe por un factor de 1,05
    bboxes, weights = hog.detectMultiScale(queryImage, winStride=(8, 8),
                                           padding=(32, 32), scale=1.05,
                                           finalThreshold=2, hitThreshold=1.0)
    # Visualizamos as caixas na iamxe
    for bbox in bboxes:
      x1, y1, w, h = bbox
      x2, y2 = x1 + w, y1 + h
      cv2.rectangle(queryImage, (x1, y1), (x2, y2),
                    (0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
    cv2.imshow("Imaxe consulta",queryImage)
    cv2.waitKey(0)

# Iniciamos HOG
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

# Cargamos o modelo entrenao por nos
model = cv2.ml.SVM_load('../data/models/pedestrian.yml')
sv = model.getSupportVectors()
rho, aplha, svidx = model.getDecisionFunction(0)
svmDetectorTrained = np.zeros(sv.shape[1] + 1, dtype=sv.dtype)
svmDetectorTrained[:-1] = -sv[:]
svmDetectorTrained[-1] = rho
# iniciamos SVMDetector entrenado por nos en HOG
hog.setSVMDetector(svmDetectorTrained)

# Detector de persoas de OpenCV con HoG
hogDefault = cv2.HOGDescriptor(winSize, blockSize, blockStride,
                               cellSize, nbins, derivAperture,
                               winSigma, histogramNormType,
                               L2HysThreshold,gammaCorrection,
                               nlevels, signedGradient)
svmDetectorDefault = cv2.HOGDescriptor_getDefaultPeopleDetector()
hogDefault.setSVMDetector(svmDetectorDefault)

# lemos as imaxes do directorio de persoas
imageDir = '../data/images/pedestrians'
imagePaths = glob.glob(os.path.join(imageDir, '*.jpg'))

#executamos o detector para imaxes con altura fixa
finalHeight = 800.0

for imagePath in imagePaths:
    print('procesando: {}'.format(imagePath))

     # Lemos a imaxe
    im = cv2.imread(imagePath, cv2.IMREAD_COLOR)

    # redimensionamos a imaxe para que teña unha altura prederminada de finalHeight
    scale = finalHeight / im.shape[0]
    im = cv2.resize(im, None, fx=scale, fy=scale)

      # Detecta persoas usando detectores SVM adestrados e predeterminados
    # detectMultiScale usando un detector adestrado por nós
    bboxes, weights = hog.detectMultiScale(im, winStride=(8, 8),
                                    padding=(32, 32),scale=1.05,
                                    finalThreshold=2,hitThreshold=1.0)

   # detectMultiScale modelos predetemiando
    bboxes2, weights2 = hogDefault.detectMultiScale(im, winStride=(8, 8),
                                padding=(32, 32),scale=1.05,
                                finalThreshold=2,hitThreshold=0)

# Debuxa rectángulos atopados na imaxe. Debuxaremos
     # caixas verdes para persoas detectadas polo modelo adestrado e
     # caixas vermellas para as persoas detectadas polo modelo predeterminado de OpenCV.
    if len(bboxes) > 0:
        print('Detector entrenado :: peons detectados: {}'
                .format(bboxes.shape[0]))
    if len(bboxes2) > 0:
        print('Detector de OpCV :: peons detectados: {}'
                .format(bboxes2.shape[0]))

    # Debuxa caixas delimitadoras detectadas sobre a imaxe
    # Vermello = detector predeterminado, Verde = Detector adestrado por nos
    for bbox in bboxes:
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        cv2.rectangle(im, (x1, y1), (x2, y2),
                      (0, 255, 0), thickness=3,
                      lineType=cv2.LINE_AA)

    for bbox in bboxes2:
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        cv2.rectangle(im, (x1, y1), (x2, y2),
                      (0, 0, 255), thickness=3,
                      lineType=cv2.LINE_AA)
    # Visualizamos os resultados finais
    cv2.imshow("Imaxe",im)
    # salvamos a disco
    imResultPath = os.path.join('resultados', os.path.basename(imagePath))
    cv2.imwrite(imResultPath, im)
    cv2.waitKey(0)


