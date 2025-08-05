import cv2
import time
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt 


#Funcion que implementa a piramide gaussiana 
def gen_gaussian_pyramid(I, levels=6):
    G = I.copy()
    gpI = [G]
    for i in range(levels):
        G = cv2.pyrDown(G)
        gpI.append(G)
    return gpI

# Función que implementa a piramide LoG
def gen_laplacian_pyramid(gpI):
    """gpI é unha piramide gaussiana xerada 
    empregando a anterior función gen_gaussian_pyramid()
    listada na cela 7."""
    num_levels = len(gpI)-1
    lpI = [gpI[num_levels]]
    for i in range(num_levels,0,-1):
        #forzamos a que os tamaños das imaxe a restar sexan iguais.
        #dado que pyrDown/pyUp traballan con multiplos pares pode haber variacións de +/- 1 pixel
        # entre a imaxe da piramide e a sobremostreada e fallaría a operacion de resta punto a punto.
        GE = cv2.pyrUp(gpI[i],dstsize=(gpI[i-1].shape[1], gpI[i-1].shape[0]))
        L = cv2.subtract(gpI[i-1],GE)
        lpI.append(L)
    return lpI

# Función que reconstruye la imagen
def reconstruct(lsA):
    ls_ = lsA[0]
    for i in range(1,len(lsA)):
        ls_ = cv2.pyrUp(ls_, dstsize=(lsA[i].shape[1], lsA[i].shape[0]))
        ls_ = cv2.add(ls_, lsA[i])
    return ls_


# MAIN():
plt.rcParams['image.cmap'] = 'gray'

# Cargamos las imagenes
bkgnd = cv2.imread('../Exercicios_clase/imaxes/1/bkgnd.png', 0)
raccoon = cv2.imread('../Exercicios_clase/imaxes/1/imgA.png', 0)
mask = cv2.imread('../Exercicios_clase/imaxes/1/mask2.png', 0)

# Por si no se encuentran las imagenes
if (bkgnd is None or raccoon is None or mask is None):
    raise Exception("No encuentro las imagenes en el sitio indicado")

# Creamos las piramides gaussianas
n_levels = 6
bkgnd_G_piramid = gen_gaussian_pyramid(bkgnd, levels=n_levels)
raccoon_G_piramid = gen_gaussian_pyramid(raccoon, levels=n_levels)
mask_G_piramid = gen_gaussian_pyramid(mask, levels=n_levels)
list_gaussians = [bkgnd_G_piramid, raccoon_G_piramid, mask_G_piramid]

# Creamos las piramides Laplaciana de Gaussiana
bkgnd_L_piramid = gen_laplacian_pyramid(bkgnd_G_piramid)
raccoon_L_piramid = gen_laplacian_pyramid(raccoon_G_piramid)
list_laplacians = [bkgnd_L_piramid, raccoon_L_piramid]


# Visualizacion
plt.figure(figsize=[15,15])

# Gaussianas
for i in range(1, (n_levels+1)*3+1, 1):
    plt.figure(1); plt.subplot(3, n_levels+1, i); plt.imshow(list_gaussians[(i-1)//(n_levels+1)][(i-1)%(n_levels+1)])

# Laplacianas
for i in range(1, (n_levels+1)*2+1, 1):
    plt.figure(2); plt.subplot(2, n_levels+1, i); plt.imshow(list_laplacians[(i-1)//(n_levels+1)][(i-1)%(n_levels+1)])

# Normalizamos la las piramides gaussianas de la máscara
mask_norm = []
for i in range(0, len(mask_G_piramid)):
    mask_norm.append(cv2.normalize(mask_G_piramid[i], mask_G_piramid[i], 0, 1, cv2.NORM_MINMAX))

# Reconstruimos la imagen
reconstruida = []
for i in range(1, len(mask_norm)):
    reconstruida.append(((1-mask_G_piramid[i-1])*bkgnd_L_piramid[-i])+(mask_G_piramid[i-1]*raccoon_L_piramid[-i]))
    plt.figure(3); plt.subplot(1, len(mask_norm), i); plt.imshow(reconstruida[i-1])

# Reconstrución a partir da piramide LoG
blended = reconstruct(reconstruida[::-1])

#Visualizamos a reconstruccion
plt.figure(figsize=(10,10))
plt.title('Reconstrución a partir de la LoG')
plt.figure(4); plt.imshow(blended)
plt.show()