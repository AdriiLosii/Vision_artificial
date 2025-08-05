# Importamos librarias
import cv2
import numpy as np

# Path a imaxe a ser cargada
imagePath = "./data/images/panther.png"

# Lemos a imaxe
# Nota que pasamos un flag = -1 mentres lemos a imaxe ( lera a imaxe tal cual e!)
imgPNG = cv2.imread(imagePath,-1)
print("image Dimension ={}".format(imgPNG.shape))

#Primiro 3 canles seran combinadas apara formar a amaxe BGR
#A mascara esta na canle alfa da imaxe orixinal
imgBGR = imgPNG[:,:,0:3]
imgMask = imgPNG[:,:,3]

cv2.imshow("Canles cor",imgBGR)
cv2.imshow("Canle alfa",imgMask)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("./data/colorChannels.png",imgBGR)
cv2.imwrite("./data/alphaChannel.png",imgMask)
