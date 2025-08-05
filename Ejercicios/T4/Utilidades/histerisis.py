# Xose R. Fdez-Vidal (e-correo: xose.vidal@usc.es)
# Codigo docente para o grao de Robótica da EPSE de Lugo.
# Dept. Física Aplicada, Universidade de Santiago de Compostela, 
# GALIZA,  2022 

import numpy as np

def hysthresh(im, T1, T2):
    """
	Descricion:
		Limiar con histérisis.

	Input:
		im  - iamxe de entrada.
		T1  - limite superior.
		T2  - limite inferior

	Output:
		bw  - imaxe binaria umbralizada.
	"""
    # algunhas calculo precios sinxelos
    rows, cols = im.shape
    rc = rows * cols
    rcmr = rc - rows
    rp1 = rows + 1

    bw = im.ravel()  # convertimos a imaxe nun vector columna
    pix = np.where(bw > T1) # atopamos os indices de todos os puntos > T1
    pix = pix[0]
    npix = pix.size # numero de pixeles con valor  > T1

    # Creamos un stack de arrays
    stack = np.zeros(rows * cols)
    stack[0:npix] = pix  # colocamos todos os puntos de borde no stack
    stp = npix  # punteiro ao stack
    for k in range(npix):
        bw[pix[k]] = -1         # Marcamos os puntos como borde = -1

    # Precomputamos un array, O (O maiuscula), cos valores de offset que corresponden aos 8 veciños
    # circundantes en calquera punt. Nota que a imaxe se transformou nun vector
    # columna, se facemos un reshape a imaxe volta a ser 2D e os
    # indices dos pixel circundantes de n serán:
    #              n-rows-1   n-1   n+rows-1
    #
    #               n-rows     n     n+rows
    #
    #              n-rows+1   n+1   n+rows+1

    O = np.array([-1, 1, -rows - 1, -rows, -rows + 1, rows - 1, rows, rows + 1])

    while stp != 0:  # mentres o stack non está baleiro
        v = int(stack[stp-1])  # sacamos o seguinte indice do stack
        stp -= 1

        if rp1 < v < rcmr:  # Prevenimos de non serar indices non correctos
            # Agora comprobamos se os pixeles circundantes deben ser 
            # the stack to be processed as well
            index = O + v  # Calculate indices of points around this pixel.introducidos no stack para ser procesados
            for l in range(8):
                ind = index[l]
                if bw[ind] > T2:  # Se value > T2,
                    stp += 1  # metemos o indice no stack.
                    stack[stp-1] = ind
                    bw[ind] = -1  # marcamolo como un punto de borde

    bw = (bw == -1)  # poñemos ceros nos puntos que non son de borde
    bw = np.reshape(bw, [rows, cols])  # retornamos a imaxe as dimension orixinais
    return bw