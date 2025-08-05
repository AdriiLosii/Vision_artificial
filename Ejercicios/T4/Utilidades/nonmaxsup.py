# Xose R. Fdez-Vidal (e-correo: xose.vidal@usc.es)
# Codigo docente para o grao de Robótica da EPSE de Lugo.
# Dept. Física Aplicada, Universidade de Santiago de Compostela, 
# GALIZA,  2022 

# Implementamos o algoritmo para a busqueda de maximos locais
# dada unha distancia e unha orientación local en cada punto da imaxe
# Version en python da funcion de P. Kovesi implementada en matlab
#OLLO: este algortimo asume que os angulos estan en 0 pi. En caso contrario,
#debemos convertilos: Por exemplo:
#    orient = np.arctan2(-Y, X)  # Angulos entre -pi a +pi
#    # Convertimos os angulos ao rango  0-pi
#    neg = orient < 0  
#    orient = orient * ~neg + (orient + np.pi) * neg
#    orient = orient * 180 / np.pi  # Convertimos a graos


#Importamos as librerias
import numpy as np



def nonmaxsup(in_img, orient, radius):
    """
    Description:
        Realiza unha supresion de non-maximos sobre unha imaxe
        empregando a orientación aportada e o radio

    Input:
        in_img  - imaxe de entrada
        orient  - angulos de orientación para detectar a caracteristica
                  Imaxe orientacion (0-180, positiva, anti-clockwise)
        radius  - Distancia para mirar a un lado e outro do pixel para
                  para determinar se é un maximo local (1.2 - 1.5)

    Output:
        im_out  - imaxe de maximos locais
    """
    # reservamos memoria para aumentar a rapidez
    rows, cols = in_img.shape
    im_out = np.zeros([rows, cols])
    iradius = np.ceil(radius).astype(int)

    # Pre-calculamos of offsets x e y relativos a cada pixel para cada angulos
    angle = np.arange(181) * np.pi / 180  # Angulos cun incremento de 1 grao (en radians)
    xoff = radius * np.cos(angle)  # offsets x e y de cada punto para o raidio e angulo determinado
    yoff = radius * np.sin(angle)  # dende cada posición referecncia

    hfrac = xoff - np.floor(xoff)  # parte enterira do offset xoff 
    vfrac = yoff - np.floor(yoff)  # parte enterira do offset yoff

    orient = np.fix(orient)  #redondeo ao enteiro mais proximo hacia cero

    # Facemos o enmmalado dende deixando a zona limitrofe aos bordes
    # de tamanho o radio fixado para a buscqueda
    col, row = np.meshgrid(np.arange(iradius, cols - iradius),
                           np.arange(iradius, rows - iradius))

    # obtemos as orienación en cada punto da imaxe
    ori = orient[row, col].astype(int)

    # x, y localozacion a un lado e outro do punto en cuestion
    x = col + xoff[ori]
    y = row - yoff[ori]

    # quedamonos coa parte enteira arredor de cada x,y
    fx = np.floor(x).astype(int)
    cx = np.ceil(x).astype(int)
    fy = np.floor(y).astype(int)
    cy = np.ceil(y).astype(int)

    # Valores da imaxe nos posicion enteiras
    tl = in_img[fy, fx]  # arriba esquerda
    tr = in_img[fy, cx]  # arriba dereita
    bl = in_img[cy, fx]  # abaixo esquerda
    br = in_img[cy, cx]  # abaixo dereit

    # Interpolacion Bi-linear para estimar o valor en x,y
    upperavg = tl + hfrac[ori] * (tr - tl)
    loweravg = bl + hfrac[ori] * (br - bl)
    v1 = upperavg + vfrac[ori] * (loweravg - upperavg)

    # comprobamos o valor a cada lado
    map_candidate_region = in_img[row, col] > v1

    x = col - xoff[ori]
    y = row + yoff[ori]

    fx = np.floor(x).astype(int)
    cx = np.ceil(x).astype(int)
    fy = np.floor(y).astype(int)
    cy = np.ceil(y).astype(int)

    tl = in_img[fy, fx]
    tr = in_img[fy, cx]
    bl = in_img[cy, fx]
    br = in_img[cy, cx]

    upperavg = tl + hfrac[ori] * (tr - tl)
    loweravg = bl + hfrac[ori] * (br - bl)
    v2 = upperavg + vfrac[ori] * (loweravg - upperavg)

    # maximo local
    map_active = in_img[row, col] > v2
    map_active = map_active * map_candidate_region
    im_out[row, col] = in_img[row, col] * map_active

    return im_out
