import numpy as np
import matplotlib.pyplot as plt
import cv2
import cmapy


def superpos_colormap(im_bk, im_fg, alpha = 0.4, beta = 0.6, cmap_name = 'jet'):
    cmap = cmapy.cmap(cmap_name)

    # Se a imaxe de fondo e de cor
    if len(im_bk.shape) == 3:
        if len(im_fg.shape) == 3:
            sm = cv2.LUT(im_fg, cmap)
        else:
            sm = np.zeros(im_bk.shape).astype(np.uint8)
            for i in range(3):
                sm[:,:,i]=im_fg
            sm = cv2.LUT(sm, cmap)
        cv2.imshow('mapa saliencia', sm)
        cv2.imshow('Saliencia sobre orixinal',cv2.addWeighted(im_bk, alpha, sm, beta, 0))
        cv2.waitKey(0)
    # se a imaxe de fonde e gris
    elif len(im_bk.shape) == 2 and len(im_fg.shape) == 2:
        cv2.imshow('mapa saliencia', cv2.applyColorMap(im_fg, cmap))
        cv2.imshow('Saliencia sobre orixinal',cv2.applyColorMap(cv2.addWeighted(im_bk, alpha, im_fg, beta, 0), cmap))
        cv2.waitKey(0)
    else:
        print('dimensions incorrectas')
        return
