# modelo.py - Treinamento de IA Simulado
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Simular Dataset Fictício
def gerar_dataset():
    np.random.seed(42)
    dias = np.arange(1, 101)
    insulina = np.random.randint(60, 200, 100)
    glicose = np.random.randint(70, 300, 100)
    evolucao = (glicose < 140) & (insulina < 150)
    return pd.DataFrame({
        "Dia": dias,
        "Insulina": insulina,
        "Glicose": glicose,
        "Evolucao": evolucao.astype(int)
    })

# Treinamento do Modelo
def treinar_modelo():
    df = gerar_dataset()
    X = df[["Insulina", "Glicose"]]
    y = df["Evolucao"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(X_train, y_train)
    return modelo, scaler, df
