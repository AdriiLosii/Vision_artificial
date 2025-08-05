Instrucciones de los códigos:
- HandTracker.py: Este código simplemente sirve para la detección simple de una mano,
sirvió de base para el resto de códigos como "DataGenerator.py" y "GestureRecog.py"
(No tiene utilidad como sí en el proyecto)

- DataGenerator.py: Sirve para tomar imágenes de las detecciones de la mano y crear así
nuestra propia base de datos.
Método de uso: Al ejecutarlo activará la webcam y se mostrarán un número de cuadrados
(configurable), se deberá de cubrir todos estos con la palma de nuestra mano, una vez estemos
posicionados pulsaremos la tecla "Z" y comenzará la detección de la mano, una vez estemos
realizando el gesto deseado pulsaremos la tecla "S" para tomar capturas que queramos (la imagen
se congelará pero seguirá tomando fotos de cada frame aunque no lo veamos). Se deberá de especificar
el directorio para cada clase que queramos implementar.

- BuildDataset.py: Al ejecutarlo tomará la base de datos en el directorio especificado y
comenzará el procesado y extracción de características de cada imagen, creando así
nuestro archivo "dataset.csv".

- TrainModel.ipynb: Se trata de un NOTEBOOK en el que su ejecución entrenará nuestro modelo,
imprimiendo distintos tipos de información en el proceso.

- GestureRecog.py: Es el programa principal, al ejecutarlo el inicio será igual que en
"DataGenerator.py", pulsaremos la tecla "Z" cuando estemos posicionados y a continuación
se mostrará la predicción del modelo creado anteriormente.

NOTA: EL DATASET Y LA BASE DE DATOS ESTÁN SUBIDAS A ONEDRIVE POR LIMITACIONES DE ENTREGA
DEL CAMPUS VIRTUAL
ENLACE: https://nubeusc-my.sharepoint.com/:f:/g/personal/adrian_losada_alvarez_rai_usc_es/En281aTnyxhPigiEj2msClkBhXJkkfgRV59CPn6rMNqi5A?e=XrhY2i