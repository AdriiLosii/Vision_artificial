import cv2
import os
import numpy as np 
from skimage.feature import hog
from skimage.filters import sobel


def image_processed(file_path):
    try:
        img = cv2.imread(file_path)
        if img is None or img.ndim != 3:
            raise ValueError("Archivo de imagen no válido o no en formato BGR.")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_resized = cv2.resize(img, (128, 128))

        # Aplicar un filtro de Sobel para resaltar bordes
        img_resized = sobel(img_resized)

        # Extraer las características HOG de la imagen filtrada
        hog_features, _ = hog(img_resized, visualize=True)

        return hog_features

    except Exception as err:
        print(f"Error procesando {file_path}: {err}")
        return np.zeros((3780,), dtype=float)  # Ajusta según el número de características HOG

def make_csv():
    mypath = './data'
    file_name = open('dataset.csv', 'w')

    for each_folder in sorted(os.listdir(mypath)):
        if each_folder.startswith('.'):
            continue

        for each_number in sorted(os.listdir(os.path.join(mypath, each_folder))):
            if each_number.startswith('.'):
                continue

            label = each_folder
            file_loc = os.path.join(mypath, each_folder, each_number)
            data = image_processed(file_loc)
            data_string = ','.join(map(str, data))
            file_name.write(data_string + ',' + label + '\n')

        print('Folder ' + each_folder + ' processed.')

    file_name.close()
    print('Dataset created.')

if __name__ == "__main__":
    make_csv()
