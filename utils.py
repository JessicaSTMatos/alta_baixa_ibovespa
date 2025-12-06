import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ---- 1. Classe para remover features (se necessário) ----
class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, feature_to_drop=None):
        # Por padrão não dropa nada no Ibovespa
        # Se quiser remover por exemplo: ["Delta_lag30", "Spread"] etc, passe ao chamar a classe
        self.feature_to_drop = feature_to_drop if feature_to_drop is not None else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if set(self.feature_to_drop).issubset(X.columns):
            X.drop(self.feature_to_drop, axis=1, inplace=True)
        return X


# ---- 2. Scaler com StandardScaler (melhor para dinâmica do mercado) ----
class StandardScalerWithNames(BaseEstimator, TransformerMixin):
    def __init__(self, features=None):
        self.features = features  # Lista das colunas que deseja normalizar
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        if self.features is None:
            self.features = X.columns.tolist()
        self.scaler.fit(X[self.features])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.features] = self.scaler.transform(X[self.features])
        return X


# ---- 3. Scaler alternativo com MinMax (caso queira testar comparativo) ----
class MinMaxScalerWithNames(BaseEstimator, TransformerMixin):
    def __init__(self, features=None):
        self.features = features
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        if self.features is None:
            self.features = X.columns.tolist()
        self.scaler.fit(X[self.features])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.features] = self.scaler.transform(X[self.features])
        return X
