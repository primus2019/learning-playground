from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


class LinearRegressor:
    
    def __init__(self,
                 random_state=0,
                ):
        self.rng = np.random.default_rng(seed=random_state)
        return None

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearRegressor:
        self.beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.matmul(X, self.beta) + self.rng.random(X.shape[0])

def test():
    total_size, y_size = 20, 2
    rng = np.random.default_rng(seed=0)
    data = np.linspace(start=[-100] * 50, stop=[100] * 50, num=total_size).T
    endog = data[:, -y_size]
    error = rng.uniform(low=-1.0, high=1.0, size=(50, total_size - y_size))
    exog = data[:, :-y_size] + error

    model = LinearRegressor().fit(exog, endog)
    y_pred = model.predict(exog)
    plt.plot(endog, label='true')
    plt.plot(y_pred, label='predict')
    plt.show()

if __name__ == '__main__':
    test()