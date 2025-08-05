import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split




#Manexo de datos e metricas de rendemento
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
        self.split_rare = split_rate
        self.data = data
        self.bias = bias
        self.normal = normal
        self.prepare_data()

    def prepare_data(self):
        if self.normal:
            self.normalizer()

        if self.bias:
            self.data = np.insert(self.data, 0, 1, axis=1)

        self.trainData, self.testData, self.trainLabel, self.testLabel = train_test_split(self.data[:, :-1],
                                                                                          self.data[:, -1],
                                                                                          test_size=self.split_rare,
                                                                                          random_state=42)

    def normalizer(self):
        norm = np.linalg.norm(self.data[:, :-1])
        self.data[:, :-1] = self.data[:, :-1] / norm


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

    #Medidas de rendemento
    recall = true_pos / float(true_pos + false_neg)
    precision = true_pos / float(true_pos + false_pos)
    fscore = 2 * precision * recall / (precision + recall)

    return  precision, recall, fscore


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

#########################################################
#Clasificador Bayesiano cuadratico
#########################################################

class ByQuadratic:
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
        covariance_matrix = []
        for index, className in enumerate(self.class_name_list):
            row_data = self.data.trainData[self.data.trainLabel == className]
            mean = np.asmatrix(np.mean(row_data, axis=0))
            means[index] = mean
            cov_matrix = (row_data - mean).T @ (row_data - mean) / self.data.trainData.shape[0]
            covariance_matrix.append(cov_matrix)

        self.means = means
        self.covariance_matrix = covariance_matrix

    def predict(self, data):
        probs = np.asmatrix(np.zeros((data.shape[0], self.priors.size)))
        for index, class_abel in enumerate(self.class_name_list):
            probs[:, index] = self.probability(data, index)
        return np.argmax(probs, axis=1)

    def probability(self, data, index):
        X = np.asmatrix(data)
        cov_matrix_det = np.linalg.det(self.covariance_matrix[index])
        cov_matrix_inv = np.linalg.pinv(self.covariance_matrix[index])
        Xm = X - self.means[index]
        Xm_covariance = (Xm @ cov_matrix_inv) @ Xm.T
        Xm_covariance_sum = Xm_covariance.sum(axis=1)
        return -0.5 * Xm_covariance_sum - 0.5 * np.log(cov_matrix_det) + np.log(self.priors[index])

    def fix(self):
        self.calculate_prior()
        self.calculate_mean()



if __name__ == '__main__':

    # Exemplo de xoguete sobre os clasificdores Bayesiano cuadratico (ByQuadratic)

    listdata = ['dataset1.csv', 'dataset2.csv']

    for index, path in enumerate(listdata):
        rawData = load_data(path)
        data = Data(rawData, bias=False)
        #Instantacionamos e adestramos
        model_q = ByQuadratic(data)
        model_q.fix()

        #Predecimos co modelos adestrado os datos de adestramento
        # e test. Ollo: non se gardan os datos derivados do adestramento
        #Ti debes gardalos en disco e lelos cando os precises.
        predicted_train = model_q.predict(data.trainData)
        predicted_test = model_q.predict(data.testData)

        print(f"Dataset {index}, Train ACC = {accuracy_metric(data.trainLabel, predicted_train)}")
        print(f"Dataset {index}, test ACC = {accuracy_metric(data.testLabel, predicted_test)} \n")
