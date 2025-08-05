import numpy as np
import pandas as pd
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


if __name__ == '__main__':

    listdata = ['Haralick_features.csv', 'LBP_features.csv']

    for index, path in enumerate(listdata):
        rawData = load_data(path)
        data = Data(rawData, bias=False)

        model_q = ByLinear(data)
        model_q.fix()

        predicted_train = model_q.predict(data.trainData)
        predicted_test = model_q.predict(data.testData)

        print(f"Dataset {index}, train ACC = {accuracy_metric(data.trainLabel, predicted_train)}")
        print(f"Dataset {index}, test ACC = {accuracy_metric(data.testLabel, predicted_test)}")
        precision, recall, f_score = calculate_metrics(predicted_train, data.trainLabel)
        print(f"Dataset {index}, train precision = {precision}, train recall = {recall}, train F-score = {f_score}")
        precision, recall, f_score = calculate_metrics(predicted_test, data.testLabel)
        print(f"Dataset {index}, test precision = {precision}, test recall = {recall}, test F-score = {f_score} \n")