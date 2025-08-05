import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split


#########################################################
#   Manejo de datos y metricas de rendimiento
#########################################################
class Data:
    trainData: np.array
    testData: np.array
    trainLabel: np.array
    testLabel: np.array

    def __init__(self, data, split_rate=0.2, bias=True, normal=False):
        self.trainData: np.array
        self.testData: np.array
        self.trainLabel: np.array
        self.testLabel: np.array
        self.split_rate = split_rate
        self.data = data
        self.bias = bias
        self.normal = normal
        self.prepare_data()

    def prepare_data(self):
        if self.normal:
            self.normalizer()

        if self.bias:
            self.data = np.insert(self.data, 0, 1, axis=1)

        self.trainData, self.testData, self.trainLabel, self.testLabel = train_test_split(self.data[:, :-1], self.data[:, -1], test_size=self.split_rate, random_state=42)

    def normalizer(self):
        norm = np.linalg.norm(self.data[:, :-1])
        self.data[:, :-1] = self.data[:, :-1] / norm


#########################################################
#   Clasificador Bayesiano lineal
#########################################################
class ByLinear:
    def __init__(self, data):
        self.data = data
        self.class_name_list = np.unique(data.trainLabel)
        self.class_name_list.sort()
        self.means = None
        self.priors = None
        self.covariance_matrix = None

    def calculate_prior(self):
        prior = np.zeros(self.class_name_list.size)
        for index, className in enumerate(self.class_name_list):
            prior[index] = self.data.trainLabel[self.data.trainLabel == className].size \
                           / self.data.trainLabel.size
        self.priors = prior

    def calculate_mean(self):
        means = np.zeros((self.class_name_list.size, self.data.trainData.shape[1]))
        for index, className in enumerate(self.class_name_list):
            row_data = self.data.trainData[self.data.trainLabel == className]
            mean = np.asmatrix(np.mean(row_data, axis=0))
            means[index] = mean

        self.means = means

        # Calcular la matriz de covarianza compartida
        all_data = self.data.trainData
        shared_covariance_matrix = np.cov(all_data.T)
        self.covariance_matrix = shared_covariance_matrix

    def predict(self, data):
        probs = np.asmatrix(np.zeros((data.shape[0], self.priors.size)))
        for index, class_label in enumerate(self.class_name_list):
            probs[:, index] = self.probability(data, index)
        return np.argmax(probs, axis=1)

    def probability(self, data, index):
        X = np.asmatrix(data)
        cov_matrix_det = np.linalg.det(self.covariance_matrix)
        cov_matrix_inv = np.linalg.pinv(self.covariance_matrix)
        Xm = X - self.means[index]
        Xm_covariance = (Xm @ cov_matrix_inv) @ Xm.T
        Xm_covariance_sum = Xm_covariance.sum(axis=1)
        return -0.5 * Xm_covariance_sum - 0.5 * np.log(cov_matrix_det) + np.log(self.priors[index])

    def fix(self):
        self.calculate_prior()
        self.calculate_mean()


#########################################################
#   Métrica de Mahalanobis
#########################################################
class MahalanobisMetric:
    def __init__(self, data, alpha=None):
        self.data = data
        self.class_name_list = np.unique(data.trainLabel)
        self.class_name_list.sort()
        self.means = None
        self.covariance_matrices = None
        self.alpha = alpha

    def calculate_means(self):
        means = np.zeros((self.class_name_list.size, self.data.trainData.shape[1]))
        for index, class_name in enumerate(self.class_name_list):
            class_data = self.data.trainData[self.data.trainLabel == class_name]
            class_mean = np.mean(class_data, axis=0)
            means[index] = class_mean
        self.means = means

    def calculate_covariance_matrices(self):
        covariance_matrices = {}
        for class_name in self.class_name_list:
            class_data = self.data.trainData[self.data.trainLabel == class_name]
            if self.alpha is not None:
                covariance_matrix = np.cov(class_data, rowvar=False) + np.eye(class_data.shape[1]) * self.alpha
            else:
                covariance_matrix = np.cov(class_data, rowvar=False)
            covariance_matrices[class_name] = covariance_matrix
        self.covariance_matrices = covariance_matrices

    def predict(self, data):
        if self.means is None or self.covariance_matrices is None:
            raise ValueError("Model parameters not initialized. Call fix method first.")

        predictions = []
        for sample in data:
            min_distance = float('inf')
            predicted_class = None
            for index, class_name in enumerate(self.class_name_list):
                mean = self.means[index]
                covariance_matrix = self.covariance_matrices[class_name]
                distance = self.mahalanobis_distance(sample, mean, covariance_matrix)
                if distance < min_distance:
                    min_distance = distance
                    predicted_class = class_name
            predictions.append(predicted_class)
        return np.array(predictions)

    def mahalanobis_distance(self, x, mean, covariance):
        deviation = x - mean
        inv_covariance = np.linalg.inv(covariance)
        distance = np.sqrt(np.dot(np.dot(deviation.T, inv_covariance), deviation))
        return distance

    def fix(self):
        self.calculate_means()
        self.calculate_covariance_matrices()


#########################################################
#   Funciones varias
#########################################################
def calculate_metrics(predicted, gold):
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    for p, g in zip(predicted, gold):
        if p == 1 and g == 1:
            true_pos += 1
        if p == 0 and g == 0:
            true_neg += 1
        if p == 1 and g == 0:
            false_pos += 1
        if p == 0 and g == 1:
            false_neg += 1

    # Verificación para evitar la división por cero
    if true_pos + false_pos == 0:
        precision = 0.0
    else:
        precision = true_pos / float(true_pos + false_pos)

    if true_pos + false_neg == 0:
        recall = 0.0
    else:
        recall = true_pos / float(true_pos + false_neg)

    if precision + recall == 0.0:
        fscore = 0.0
    else:
        fscore = 2 * precision * recall / (precision + recall)

    return precision * 100.0, recall * 100.0, fscore * 100.0


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def load_data(path, array=True):
    train = pd.read_csv(path)
    if array:
        train = train.to_numpy()
    return train

def obtener_valor_maximo(archivo_csv):
    # Inicializar el valor máximo con un valor muy pequeño
    valor_maximo = float('-inf')

    with open(archivo_csv, 'r') as csvfile:
        # Crear un lector de CSV
        reader = csv.reader(csvfile)

        # Saltar la primera fila
        next(reader)

        # Iterar sobre las filas
        for row in reader:
            # Iterar sobre los valores en cada fila (menos la última columa que corresponde a las etiquetas)
            for i in range(len(row)-1):
                # Convertir el valor a flotante y actualizar el máximo si es necesario
                row[i] = float(row[i])
                if row[i] > valor_maximo:
                    valor_maximo = row[i]
    
    return valor_maximo

def normalizaCSV(archivo_csv, archivo_normalizado):
    valor_maximo = obtener_valor_maximo(archivo_csv)
    with open(archivo_csv, 'r') as csvfile:
        # Crear un lector de CSV y un escritor de CSV
        reader = csv.reader(csvfile)
        writer = csv.writer(open(archivo_normalizado, 'w', newline=''))

        # Escribir labels y pasar a la siguiente fila
        writer.writerow(next(reader))

        # Iterar sobre las filas
        for row in reader:
            # Dividir cada valor por el valor máximo y escribir la fila normalizada
            fila_normalizada = [float(row[i]) / valor_maximo for i in range(len(row)-1)]
            fila_normalizada.append(str(row[len(row)-1]))   # Agregamos la etiqueta
            writer.writerow(fila_normalizada)