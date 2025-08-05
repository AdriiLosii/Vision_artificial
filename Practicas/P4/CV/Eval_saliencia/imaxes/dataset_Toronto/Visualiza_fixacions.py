import cv2
import scipy.io as sio
import numpy as np
import glob
import os
  
"""
Este programa esta feito para ilustrar como se poden ler as
fixacions da estrutura de datos origfixdata.mat creada por N. Bruce
dende Matlab para a súa base de datos (Toronto). Esta estrutura é un
array de celas onde en cada posición hai unha imaxe que ten un valor
de 1 na posición onde un obxervador humano fixou un punto (seguidor ocular).
Este ficheiro pode lerse dende scipy e este script ilustra isto.
"""

#Modifica isto segundo a tua estrutura de ficheiros
path_to_origfixdata_mat = "./Fixacions/" #path ata o ficheiro origfixdata.mat
path_mapas_densidade = "./MapasDensidade/" #path ao cartafol cos mapas de densidade
path_to_imaxes = "./Imaxes/"  #path ao directorio onde estan as imaxes da base de datos

#Neste caso imos ler todas as imaxes da base de datos (sera longo de visualizar)

#Lemos a estrutra .mat das fixacions. Na variable fixacions poderemos
#acceder asos datos como fixacions['white'][0,index] e devolveramos unha
#imaxe onde nos pixeles que teñen fixacion teñen un valor de 1.
#index debe variar entre 0 e numero maximo de imaxes na base de datos (120 en total)
fixacions =  sio.loadmat(path_to_origfixdata_mat + 'origfixdata.mat')

#imos ler todas as imaxe e visualiamos as tres 
# opcions: fixacions, mapas de densidades e fixacion sobre a imaxe orixinal

for index in range(len(fixacions['white'][0])):
    #Visualizamos as fixacions (*255 pq con valor 1 non se ven!)
    cv2.imshow('Mapa de Fixacions',fixacions['white'][0,index]*255)

    #Visualizamos o mapa de densidade (e o mapa de fixacion suavizado cunha gaussiana. Xa esta feito!)
    print(path_mapas_densidade + "d" + str(index) + ".jpg")
    mapa_den = cv2.imread(path_mapas_densidade + "d" + str(index+1) + ".jpg")
    # if mapa_den == None:
    #     raise Exception("Non atopo a imaxe")
    cv2.imshow('Mapa de densidade',mapa_den)

    #Visualizamos sobre a imaxe orixinal e o mapa de densidade
    # as fixacions cun circulo vermello
    #A visualizacion asi e mais doada!

    # Atopamos os pixels con valor == 1 (fixacions)
    # eses puntos seran os centroides dun circulo vermello
    # de radio 5
    centroides = np.argwhere(fixacions['white'][0,index] == 1)
    for i in range(len(centroides)):
        cv2.circle(mapa_den, (centroides[i][1],centroides[i][0]), 5, (0,0,255))
    cv2.imshow('Mapa densidade con fixacions',mapa_den)

    #Visualizamos sobre a imaxe orixinal as fixacions cun circulo vermello
    #A visualizacion asi e mais doada!

    # Atopamos os pixels con valor == 1 nas fixacions
    # eses puntos seran os centroides dun circulo vermello
    # de radio 5
    centroides = np.argwhere(fixacions['white'][0,index] == 1)
    im = cv2.imread(path_to_imaxes + str(index+1) + ".jpg")
    # if im == None:
    #     raise Exception("Non atopo a imaxe RGB")
    for i in range(len(centroides)):
        cv2.circle(im, (centroides[i][1],centroides[i][0]), 5, (0,0,255))
    cv2.imshow('Orixinal con fixacions',im)
    cv2.waitKey(0)
