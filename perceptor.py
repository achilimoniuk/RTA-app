import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
import pickle

iris = load_iris()

df = pd.DataFrame(data = np.c_[iris["data"], iris["target"]], columns = iris["feature_names"]+["target"])

df.drop(df.index[df["target"] == 2], inplace = True)
X = df.loc[:, ["petal length (cm)", "sepal length (cm)"]].values
y = df.loc[:, ["target"]].values

class Perceptron:
    
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = None
        self.errors_ = None
    
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    def predict(self, X):
        return np.where(self._net_input(X) >= 0, 1, -1)
    
    def _net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

model = Perceptron()
model.fit(X,y)


filename = 'model.pkl'
pickle.dump(model, open(filename, 'wb'))
