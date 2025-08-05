
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['image.cmap'] = 'gray'


# Lemos e visualizamos as imaxes
img_rec = cv2.imread('./data/rectangle.jpg', cv2.IMREAD_GRAYSCALE)
# Comprobamos se se leu a imaxe
if img_rec is None:
    print("Non poiden ler a imaxe do rectangulo")
    
img_cir = cv2.imread('./data/circle.jpg', cv2.IMREAD_GRAYSCALE)
if img_cir is None:
    print("Non poiden ler a imaxe do circulo")

plt.figure(figsize = [20,5])
plt.subplot(121);  plt.imshow(img_rec)
plt.subplot(122);  plt.imshow(img_cir)
print(img_rec.shape)


# ### Operador AND 
result = cv2.bitwise_and(img_rec, img_cir, mask = None)
plt.imshow(result)


# Operador OR 
result = cv2.bitwise_or(img_rec, img_cir, mask = None)
plt.imshow(result)


# OPerador XOR 
result = cv2.bitwise_xor(img_rec, img_cir, mask = None)
plt.imshow(result)


