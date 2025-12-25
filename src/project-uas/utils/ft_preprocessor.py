import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FTPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols, cat_cols, scaler, cat_encoders):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.scaler = scaler
        self.cat_encoders = cat_encoders

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for col in self.cat_cols:
            X[col] = self.cat_encoders[col].transform(X[col].astype(str))

        X_num = self.scaler.transform(X[self.num_cols])
        X_cat = X[self.cat_cols].values

        return np.hstack([X_num, X_cat])
