from __future__ import annotations

from multiprocessing import Pool, cpu_count
from itertools import product
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris


class LogisticRegressor:

    def __init__(self,
                 random_state=0,
                ):
        self.rng = np.random.default_rng(seed=random_state)
        return None

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            tol=1e-6,
            max_iter=3,
            lr=.001,
           ) -> LogisticRegressor:
        '''Fit model with exogenous and exdogenous
        # params
        - X
        '''
        self.weight = self.rng.random((X.shape[1], 1))
        iteration = 0
        hist_loss = 0
        while iteration < max_iter:
            with Pool(cpu_count()) as task:
                res = task.starmap(self.objective, product(X, y))
            losses = [val[0] for val in res]
            grads = np.asarray([val[1] for val in res])
            self.weight = self.weight# - lr * grads.mean(axis=0).reshape((-1, 1))

            print('weights  \n{}'.format(self.weight))
            # print('gradient \n{}'.format(grads.mean(axis=0).reshape((-1, 1))))
            # print('adjust {}'.format(np.mean(grads, axis=0) * lr))

            print('iteration {}, loss {}'.format(iteration, -np.mean(losses)))

            temp_loss = -np.mean(losses)
            # if good, get weight
            if np.abs(hist_loss - temp_loss) < tol:
                break

            # if not good, fit again
            hist_loss = temp_loss
            iteration += 1
        return self

    def logis(self, x):
        error = self.rng.random()
        return np.divide(1, 1 + np.exp(-np.matmul(self.weight.T, x.T) + error))

    def objective(self,
                  x,
                  y,
                 ):
        y_pred = self.logis(x)
        # print(f'prediction: {y_pred}')
        grad = (y_pred - y * x).T
        loss = y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)
        return loss, grad

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.fromiter(map(self.logis, X), dtype=np.dtype('float32'))

def test():
    X, y = load_iris(return_X_y=True)
    X_train, X_test = X[:, :-20], X[:, -20:]
    y_train, y_test = y[:-20], y[-20:]

    model = LogisticRegressor().fit(X_train, y_train)
    
    y_pred = model.predict(X_test)

    plt.plot(y_test, label='true')
    plt.plot(y_pred, label='predict')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    test()