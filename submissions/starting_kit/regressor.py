import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.base import BaseEstimator

class Regressor(BaseEstimator):
    def __init__(self):
        self.reg =  xgb.XGBRegressor(max_depth=7, n_estimators=800, gamma=0.0, min_child_weight=1, 
                          subsample=0.9, colsample_bytree=0.6, colsample_bylevel=0.8,
                          reg_alpha=0.2, reg_lambda=0.5, objective='reg:squarederror')

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        pred = self.reg.predict(X)
        n = len(X)
        return pred.reshape(n,1)

