# Importamos as librerias
import cv2
import numpy as np

# Path á imaxe que queremos cargar
imagePath = "./data/images/musk.jpg"

# Lemos a imaxe
img = cv2.imread(imagePath)

# Visualizamos a imaxe
cv2.imshow("Imaxe",img)

# Convertimos o espazo de cor BGR a RGB
imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
cv2.imshow("Imaxe RGB",imgRGB)
cv2.imwrite("./data/imgRGB.png",imgRGB)

cv2.imshow("Canle Blue",img[:,:,0])
cv2.imshow("Canle Verde",img[:,:,1])
cv2.imshow("Canle vermella",img[:,:,2])
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite("./data/blueChannel.png",img[:,:,0])
# cv2.imwrite("./data/greenChannel.png",img[:,:,1])
# cv2.imwrite("./data/redChannel.png",img[:,:,2])
