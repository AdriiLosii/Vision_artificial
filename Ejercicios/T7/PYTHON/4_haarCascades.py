import cv2
import numpy as np

# cargamos dende disco o ficheiro xml.
faceCascade = cv2.CascadeClassifier('../data/models/haarcascade_frontalface_default.xml')
faceNeighborsMax = 10
neighborStep = 1

# Lemos a imaxe
frame = cv2.imread("../data/images/hillary_clinton.jpg")
frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

count = 1
for neigh in range(1, faceNeighborsMax, neighborStep):
    faces = faceCascade.detectMultiScale(frameGray, 1.2, neigh)
    frameClone = np.copy(frame)
    for (x, y, w, h) in faces:
        cv2.rectangle(frameClone, (x, y),
                      (x + w, y + h),
                      (255, 0, 0),2)

    cv2.putText(frameClone,
    "# Vecinhos = {}".format(neigh), (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)
    cv2.imshow("Frame",frameClone)
    cv2.waitKey(0)

cv2.destroyAllWindows()

# detectamos caras co Haar cascade
faceCascade = cv2.CascadeClassifier('../data/models/haarcascade_frontalface_default.xml')
smileCascade = cv2.CascadeClassifier('../data/models/haarcascade_smile.xml')
smileNeighborsMax = 90
neighborStep = 10

frame = cv2.imread("../data/images/hillary_clinton.jpg")

frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(frameGray, 1.4, 5)

# caixas contedoras
for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y),
                  (x + w, y + h),
                  (255, 0, 0), 2)
    faceRoiGray = frameGray[y: y + h, x: x + w]
    faceRoiOriginal = frame[y: y + h, x: x + w]

# detectamos sorrisos
for neigh in range(1, smileNeighborsMax, neighborStep):
    smile = smileCascade.detectMultiScale(faceRoiGray,
                          1.5, neigh)

    frameClone = np.copy(frame)
    faceRoiClone = frameClone[y: y + h, x: x + w]
    for (xx, yy, ww, hh) in smile:
        cv2.rectangle(faceRoiClone, (xx, yy),
                      (xx + ww, yy + hh),
                      (0, 255, 0), 2)

    cv2.putText(frameClone,
              "#Vecinhos = {}".format(neigh),
              (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
              (0, 0, 255), 4)
    cv2.imshow("Frame",frameClone)
    cv2.waitKey(0)

cv2.destroyAllWindows()

