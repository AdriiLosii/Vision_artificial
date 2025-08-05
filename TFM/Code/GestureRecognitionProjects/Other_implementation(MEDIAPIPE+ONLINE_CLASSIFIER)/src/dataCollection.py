import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Inicialización de la cámara, detector de mano, y variables relacionadas
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
folder = "./data/THUMBS_UP"
counter = 0

while True:
    # Capturar el fotograma de la cámara
    success, img = cap.read()

    # Detectar la mano en el fotograma
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Crear una imagen blanca con dimensiones predefinidas
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Recortar la región de interés (ROI) alrededor de la mano
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
        else:
            # Ajustar la altura si la relación de aspecto es menor o igual a 1
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Mostrar las imágenes recortada y blanca
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # Mostrar la imagen original con la detección de mano
    cv2.imshow("Image", img)

    # Esperar una tecla y realizar acciones asociadas
    key = cv2.waitKey(1)
    if key == ord("s"):
        # Guardar la imagen blanca en la carpeta especificada
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
    elif key == 27:  # 27 corresponde a la tecla Esc
        break

# Liberar recursos al finalizar
cv2.destroyAllWindows()
cap.release()
