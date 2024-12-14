from flask import Flask, render_template, request, jsonify
from modelo import treinar_modelo
import numpy as np

app = Flask(__name__)

# Carregar Modelo
modelo, scaler, historico_df = treinar_modelo()

@app.route("/", methods=["GET", "POST"])
def index():
    previsao = None
    historico = historico_df.to_dict("records")

    if request.method == "POST":
        insulina = float(request.form["insulina"])
        glicose = float(request.form["glicose"])

        # Normalizar e Prever
        entrada = scaler.transform([[insulina, glicose]])
        risco = modelo.predict(entrada)[0]
        previsao = "Evolução Boa" if risco == 1 else "Evolução Ruim"

        # Adicionar ao Histórico
        historico_df.loc[len(historico_df)] = [len(historico_df) + 1, insulina, glicose, risco]
        historico = historico_df.to_dict("records")

    return render_template("index.html", previsao=previsao, historico=historico)

if __name__ == "__main__":
    app.run(debug=True)
