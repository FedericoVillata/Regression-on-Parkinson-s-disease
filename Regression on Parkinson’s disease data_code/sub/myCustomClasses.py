import ssl

import numpy as np
import pandas as pd
#import Minimization as *


class RegressionLLS:
    def __init__(self, x, y):
        self.x = x
        self.Nf = x.shape[1]
        self.y = y
        self.w_hat = None

    def train(self):
        self.w_hat = np.linalg.inv(self.x.T @ self.x) @ (self.x.T @ self.y)
        return self.w_hat

    def predict(self, X_te):
        y_hat_tr = self.x @ self.w_hat
        y_hat_te = X_te @ self.w_hat
        return y_hat_tr, y_hat_te
