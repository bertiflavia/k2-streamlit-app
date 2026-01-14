import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import os

# caminhos relativos ao projeto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ====== K2 paths ======
MODEL_K2_PATH = os.path.join(BASE_DIR, "model", "ANN_k2.keras")
DATA_K2_PATH  = os.path.join(BASE_DIR, "data", "k2_data.csv")

# ====== K1 paths ======
MODEL_K1_PATH = os.path.join(BASE_DIR, "model", "ANN_k1.keras")
DATA_K1_PATH  = os.path.join(BASE_DIR, "data", "k1_data.csv")


def prever_k2(d, p, tw, a0):
    # 1) montar input
    inputs = np.array([[d, p, tw, a0]], dtype=float)

    # 2) scaler igual ao treino
    df = pd.read_csv(DATA_K2_PATH, delimiter=';')
    X_train = df.drop(columns=['k2'])

    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(inputs)

    # 3) carregar modelo e prever
    model = load_model(MODEL_K2_PATH, compile=False)
    k2_pred = model.predict(X_scaled, verbose=0).flatten()[0]

    # 4) resultado
    results = pd.DataFrame(
        [[d, p, tw, a0, k2_pred]],
        columns=['d', 'p', 'tw', 'a0', 'k2_pred']
    )
    return results


def prever_k1(fc, hs, a, As):
    # 1) montar input
    inputs = np.array([[fc, hs, a, As]], dtype=float)

    # 2) scaler igual ao treino
    df = pd.read_csv(DATA_K1_PATH, delimiter=';')
    X_train = df.drop(columns=['k1'])

    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(inputs)

    # 3) carregar modelo e prever
    model = load_model(MODEL_K1_PATH, compile=False)
    k1_pred = model.predict(X_scaled, verbose=0).flatten()[0]

    # 4) resultado
    results = pd.DataFrame(
        [[fc, hs, a, As, k1_pred]],
        columns=['fc', 'hs', 'a', 'As', 'k1_pred']
    )
    return results

