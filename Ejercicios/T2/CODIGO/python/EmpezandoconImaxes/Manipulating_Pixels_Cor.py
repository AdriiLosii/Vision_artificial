# Importamos librarias
import cv2
import numpy as np

# Path a imaxe a ser cargada
imagePath = "./data/images/number_zero.jpg"
testImage = cv2.imread(imagePath,1)

#Access color pixel
print(testImage[0,0])

#Modificamos o pixel
testImage[0,0] = (0,255,255)
cv2.imshow("Cero 1",testImage)
cv2.imwrite("./data/cero1.png",testImage)

testImage[1,1] = (255,255,0)
cv2.imshow("Cero 2",testImage)
cv2.imwrite("./data/cero2.png",testImage)

testImage[2,2] = (255,0,255)
cv2.imshow("Cero 3",testImage)
cv2.imwrite("./data/cero3.png",testImage)

#Modificamos a rexión de interese
testImage[0:3,0:3] = (255,0,0)
testImage[3:6,0:3] = (0,255,0)
testImage[6:9,0:3] = (0,0,255)

cv2.imshow("Cero", testImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("./data/cero.png",testImage)
