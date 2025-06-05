import pymc as pm
import numpy as np
import pandas as pd

import os
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

from sklearn.preprocessing import StandardScaler


class BayesMapClassifier:
    def __init__(self):
        self.model = None
        self.params = None
        self.scaler = StandardScaler()


    def fit(self, data):
        X = data.drop(columns=["Valor_ql"]).astype(np.float64).values
        y = data["Valor_ql"].astype(int).values

        n_obs, n_features = X.shape
        K = len(np.unique(y))

        with pm.Model() as model:
            beta0 = pm.Normal("beta0", mu=0, sigma=10, shape=K)
            betas = pm.Normal("betas", mu=0, sigma=10, shape=(n_features, K))

            logits = pm.math.dot(X, betas) + beta0.dimshuffle("x", 0)

            
            y_obs = pm.Categorical("y_obs", logit_p=logits, observed=y)


            self.params = pm.find_MAP(progressbar=False)
            self.model = model




    def predict_proba(self, data):
        X_new = data.astype(np.float64).values

        # Recupera os parâmetros
        beta0 = np.array(self.params["beta0"])  # shape: (K,)
        betas = np.array(self.params["betas"])  # shape: (n_features, K)

        # Calcula os logits
        logits = np.dot(X_new, betas) + beta0  # shape: (n_amostras, K)

        # Aplica softmax para obter probabilidades
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # para estabilidade numérica
        probas = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return np.argmax(probas, axis=1)


    def predict(self, data):
        probas = self.predict_proba(data)
        
        # Retorna a classe com maior probabilidade para cada amostra
        return np.argmax(probas, axis=1)  # Retorna valores discretos de 0 a 7



        
    def update(self, x, y):
        y_new = y.astype(int)  # garantir que os valores são inteiros
        X_new = self.scaler.transform(x)

        n_obs, n_features = X_new.shape
        n_classes = len(np.unique(y_new))  # deve ser 8 no seu caso

        with pm.Model() as self.model:
            # Priori com base nos parâmetros anteriores (se houver)
            if self.params:
                betas_prior = self.params["betas"]
                beta0_prior = self.params["beta0"]
            else:
                betas_prior = np.zeros((n_features, n_classes))
                beta0_prior = np.zeros(n_classes)

            # Priori dos parâmetros
            betas = pm.Normal("betas", mu=betas_prior, sigma=10, shape=(n_features, n_classes))
            beta0 = pm.Normal("beta0", mu=beta0_prior, sigma=10, shape=(n_classes,))

            # Média linear (logits)
            mu = pm.math.dot(X_new, betas) + beta0

            # Softmax → probabilidades
            p = pm.math.softmax(mu)

            # Verossimilhança categórica
            y_obs = pm.Categorical("y_obs", p=p, observed=y_new)

            # Atualiza os parâmetros com MAP
            self.params = pm.find_MAP()
