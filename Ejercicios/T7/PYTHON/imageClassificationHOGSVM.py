import cv2
import numpy as np
from cropEyeRegion import getCroppedEyeRegion
import os

predictions2Label = {0: "Sen gafas", 1: "Con gafas"}

def getTrainTest(path, class_val, test_fraction = 0.2):
  testData = []
  trainData = []
  trainLabels = []
  testLabels = []
  inputDir = os.path.expanduser(path)

  # Lemos as imaxes dende disco
  if os.path.isdir(inputDir):
    images = os.listdir(inputDir)
    images.sort()
    nTest = int(len(images) * test_fraction)

  for counter, img in enumerate(images):

    im = cv2.imread(os.path.join(inputDir, img))
    # engadimos nTest para os datos de test
    if counter < nTest:
      testData.append(im)
      testLabels.append(class_val)
    else:
      # engadimos nTrain aos datos de entrenamento
      trainData.append(im)
      trainLabels.append(class_val)

  return trainData, trainLabels, testData, testLabels

def svmInit(C, gamma):
  model = cv2.ml.SVM_create()
  model.setGamma(gamma)
  model.setC(C)
  model.setKernel(cv2.ml.SVM_RBF)
  model.setType(cv2.ml.SVM_C_SVC)
  # model.setDegree(4)

  return model

def svmTrain(model, samples, responses):
  model.train(samples, cv2.ml.ROW_SAMPLE, responses)
  return model

def svmPredict(model, samples):
  return model.predict(samples)[1].ravel()

def svmEvaluate(model, samples, labels):
  predictions = svmPredict(model, samples)
  accuracy = (labels == predictions).mean()
  print('Porcentaxe de precision: %.2f %%' % (accuracy * 100))
  return accuracy

def prepareData(data):
  featureVectorLength = len(data[0])
  features = np.float32(data).reshape(-1, featureVectorLength)
  return features

def computeHOG(hog, data):

  hogData = []
  for image in data:
    hogFeatures = hog.compute(image)
    hogData.append(hogFeatures)

  return hogData

# Path1 is class 0 and Path2 is class 1
path1 = '../data/images/glassesDataset/cropped_withoutGlasses2'
path2 = '../data/images/glassesDataset/cropped_withGlasses2'

# Inicializamos os parametros de hog 
winSize = (96, 32)
blockSize = (8, 8)
blockStride = (8, 8)
cellSize = (4, 4)
nbins = 9
derivAperture = 0
winSigma = 4.0
histogramNormType = 1
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 1
nlevels = 64

hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize,
                      nbins,derivAperture, winSigma,
                      histogramNormType,L2HysThreshold,
                      gammaCorrection, nlevels, 1)

# conseguimos as imaxe de entrenamento e test para ambas clases
negTrainImages, negTrainLabels, negTestImages, negTestLabels = \
getTrainTest(path1, 0, .2)
posTrainImages, posTrainLabels, posTestImages, posTestLabels = \
getTrainTest(path2, 1, .2)

# Engadimos imaxes positivas e negativas
trainImages = np.concatenate((np.array(negTrainImages),
                        np.array(posTrainImages)),
                            axis=0)
testImages = np.concatenate((np.array(negTestImages),
                          np.array(posTestImages)),
                          axis=0)

# Engadimos as etiqueteas positivas e negativas 
trainLabels = np.concatenate((np.array(negTrainLabels),
                          np.array(posTrainLabels)),
                          axis=0)
testLabels = np.concatenate((np.array(negTestLabels),
                          np.array(posTestLabels)),
                          axis=0)

### Calculo das caracteristicas
trainHOG = computeHOG(hog, trainImages)
testHOG = computeHOG(hog, testImages)

# Cambiamos o formato de Hog para o SVM
trainFeatures = prepareData(trainHOG)
testFeatures = prepareData(testHOG)

###########  SVM adestramenot  ##############
model = svmInit(C=2.5, gamma=0.02)  # C = 0.1, gamma 10 for
#linear kernel
model = svmTrain(model, trainFeatures, trainLabels)
model.save("./results/eyeGlassClassifierModel.yml")

##########  SVM test  ###############
savedModel = cv2.ml.SVM_load("./results/eyeGlassClassifierModel.yml")

# precision
accuracy = svmEvaluate(savedModel, testFeatures, testLabels)

# test para unha imaxe separada
filename = "../data/images/glassesDataset/glasses_4.jpg"
testImage = cv2.imread(filename)
cropped = getCroppedEyeRegion(testImage)
testHOG = computeHOG(hog, np.array([cropped]))
testFeatures = prepareData(testHOG)
predictions = svmPredict(savedModel, testFeatures)
print("Predicion = {}"
      .format(predictions2Label[int(predictions[0])]))

cv2.imshow("Imaxe Test",testImage)
cv2.waitKey(0)
cv2.imshow("zona de ollos",cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Test para unha imaxe separada
filename = "../data/images/glassesDataset/no_glasses1.jpg"
testImage = cv2.imread(filename)
cropped = getCroppedEyeRegion(testImage)
testHOG = computeHOG(hog, np.array([cropped]))
testFeatures = prepareData(testHOG)
predictions = svmPredict(savedModel, testFeatures)
print("Predicion = {}"
      .format(predictions2Label[int(predictions[0])]))
cv2.imshow("Imaxe test",testImage)
cv2.waitKey(0)
cv2.imshow("zona de ollos",cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()

