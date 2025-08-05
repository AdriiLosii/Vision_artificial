import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math


def main():
    # Inicialización de la cámara, detector de mano, clasificador y variables relacionadas
    capture = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    classifier = Classifier("./gestureModel.h5", "./etiquetas.txt")
    offset = 20
    imgSize = 300
    labels = ["A", "B", "C", "OK", "THUMBS_UP"]

    while capture.isOpened():
        pressed_key = cv2.waitKey(1)
        # Capturar el fotograma de la cámara
        success, img = capture.read()
        imgOutput = img.copy()

        # Detectar la mano en el fotograma
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Crear una imagen blanca con dimensiones predefinidas
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            imgCropShape = imgCrop.shape
            aspectRatio = h / w

            if aspectRatio > 1:
                # Ajustar el ancho si la relación de aspecto es mayor que 1
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                print(prediction, index)
            else:
                # Ajustar la altura si la relación de aspecto es menor o igual a 1
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

            # Dibujar rectángulos y etiquetas en la imagen de salida
            cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x + w + offset, y - offset), (255, 0, 0), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 25), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 0), 4)

        cv2.imshow("Image", imgOutput)

        if pressed_key == 27:
            break

    cv2.destroyAllWindows()
    capture.release()

if __name__ == '__main__':
    main()