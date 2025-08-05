import numpy as np

class Mahalanobis:
    def __init__(self, data):
        self.data = data
        self.mean_vector = None
        self.covariance_matrix = None
        self.inverse_covariance_matrix = None
    
    def fix(self):
        self.compute_parameters()
    
    def compute_parameters(self):
        # Calcular la media y la matriz de covarianza
        self.mean_vector = np.mean(self.data, axis=0)
        self.covariance_matrix = np.cov(self.data.T)
        
        # Calcular la matriz de covarianza inversa (pseudoinversa para evitar singularidad)
        self.inverse_covariance_matrix = np.linalg.pinv(self.covariance_matrix)
    
    def predict(self, x):
        # Calcular la distancia de Mahalanobis entre el vector de características x y la media
        mahalanobis_distance = np.sqrt((x - self.mean_vector).dot(self.inverse_covariance_matrix).dot((x - self.mean_vector).T))
        return mahalanobis_distance
