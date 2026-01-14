import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import os

# caminhos relativos ao projeto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "ANN_k2.keras")
DATA_PATH  = os.path.join(BASE_DIR, "data", "k2_data.csv")


def prever_k2(d, p, tw, a0):
    # 1. Montar input
    inputs = np.array([[d, p, tw, a0]])

    # 2. Ajustar scaler (igual ao seu código)
    df = pd.read_csv(DATA_PATH, delimiter=';')
    X_train = df.drop(columns=['k2'])

    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(inputs)

    # 3. Carregar modelo e prever
    model = load_model(MODEL_PATH)

    k2_pred = model.predict(X_scaled).flatten()

    # 4. Resultado em DataFrame
    results = pd.DataFrame(
        [[d, p, tw, a0, k2_pred[0]]],
        columns=['d', 'p', 'tw', 'a0', 'k2_pred']
    )

    return results
