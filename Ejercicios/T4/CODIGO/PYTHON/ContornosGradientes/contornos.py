import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)
matplotlib.rcParams['image.cmap'] = 'gray'

imagePath = "../../data/Contour.png"
image = cv2.imread(imagePath)
if image is None:
    print("Non poiden ler a imaxe")

imageCopy = image.copy()
# a gris
imageGray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)


# Visualizamos
plt.figure()
plt.subplot(121)
plt.imshow(image[:,:,::-1])
plt.title("Imaxe orixinal")
plt.subplot(122)
plt.imshow(imageGray)
plt.title("Imaxe de gris")
plt.show()

# Atopamos os contornos na imaxe (todos)
contours, hierarchy = cv2.findContours(imageGray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

print("Numero de contornos atopados = {}".format(len(contours)))
print("\nXerarquia : \n{}".format(hierarchy))

# Debuxamos os contornos
cv2.drawContours(image, contours, -1, (0,255,0), 3)
cv2.imshow("imaxe",image)
cv2.waitKey(0)

# Atopamos contorno externos
contours, hierarchy = cv2.findContours(imageGray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Numero de contornos atopados = {}".format(len(contours)))
image = imageCopy.copy()
# Debuxmos todos os contorno
cv2.drawContours(image, contours, -1, (0,255,0), 3)
cv2.imshow("imaxe",image)
cv2.waitKey(0)

# Debuxamos o contorno 3º. A numeracion pode non corresponderse coa posicions
image = imageCopy.copy()
cv2.drawContours(image, contours[2], -1, (0,0,255), 3)
cv2.imshow("imaxe",image)
cv2.waitKey(0)

# Atopamos todos os contornos
contours, hierarchy = cv2.findContours(imageGray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# Debuxamos os contornos
cv2.drawContours(image, contours, -1, (0,255,0), 3)

for cnt in contours:
    # Empregamos os momentos para achar o centroide
    M = cv2.moments(cnt)
    x = int(round(M["m10"]/M["m00"]))
    y = int(round(M["m01"]/M["m00"]))

    # Marcamos o centro
    cv2.circle(image, (x,y), 10, (255,0,0), -1);

cv2.imshow("imaxe",image)
cv2.waitKey(0)

for index,cnt in enumerate(contours):
    # Empregamos os momentos para achar o centroide
    M = cv2.moments(cnt)
    x = int(round(M["m10"]/M["m00"]))
    y = int(round(M["m01"]/M["m00"]))
    
    # Marcamos o centro
    cv2.circle(image, (x,y), 10, (255,0,0), -1);
    
    # Marcamos o indice do contorno
    cv2.putText(image, "{}".format(index + 1), (x+40, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2);

imageCopy = image.copy()

cv2.imshow("imaxe",image)
cv2.waitKey(0)

for index,cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    print("Contorno #{} ten un area = {} e perimetro = {}".format(index+1,area,perimeter))

image = imageCopy.copy()
for cnt in contours:
    # Rectagulo verticas
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,255), 2)

cv2.imshow("imaxe",image)
cv2.waitKey(0)

image = imageCopy.copy()
for cnt in contours:
    # caixa xirada
    box = cv2.minAreaRect(cnt)
    boxPts = np.int0(cv2.boxPoints(box))
    cv2.drawContours(image, [boxPts], -1, (0,255,255), 2)

cv2.imshow("imaxe",image)
cv2.waitKey(0)

image = imageCopy.copy()
for cnt in contours:
    # axustamos a un circulo
    ((x,y),radius) = cv2.minEnclosingCircle(cnt)
    cv2.circle(image, (int(x),int(y)), int(round(radius)), (125,125,125), 2)

cv2.imshow("imaxe",image)
cv2.waitKey(0)

image = imageCopy.copy()
for cnt in contours:
    #Axustamos a unha elipse
    # O contorno ten que ter como minimo 5 puntos
    if len(cnt) < 5:
        continue
    ellipse = cv2.fitEllipse(cnt)
    cv2.ellipse(image, ellipse, (255,0,125), 2)

cv2.imshow("imaxe",image)
cv2.waitKey(0)
