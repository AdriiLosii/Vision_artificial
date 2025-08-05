# Importamos as librerias
import cv2
import numpy as np

# Path a imaxe a ser cargada
imagePath ="./data/images/musk.jpg"

# Lemos a imaxe
img = cv2.imread(imagePath)
print("dimensions da imaxe {}".format(img.shape))

# Separamos as compñentes B,G,R
b,g,r = cv2.split(img)
# zr=np.floor(np.zeros(b.shape),dtype=int)
# print("dimensions da imaxe {}".format(zr.dtype))
# mergedOutput = cv2.merge((b,b,b))

# Amosamos as canles
cv2.imshow("Canel Blue",b)
cv2.imshow("Canle Green",g)
cv2.imshow("Canle Red",r)
# cv2.imwrite("./data/blueChannel.png",b)
# cv2.imwrite("./data/greenChannel.png",g)
# cv2.imwrite("./data/redChannel.png",r)

# Unimos as canles BGR de novo nunha imaxe
mergedOutput = cv2.merge((b,g,r))
# Visualizamos a imaxe composta
cv2.imshow("Imaxe unida",mergedOutput)
cv2.waitKey(0)
cv2.destroyAllWindows()
