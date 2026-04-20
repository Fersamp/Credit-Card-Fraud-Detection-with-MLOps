import mlflow
import mlflow.pyfunc
import pandas as pd

mlflow.set_tracking_uri("http://127.0.0.1:5000")

print("Carregando modelo...")

# caminho do modelo salvo no MLflow (seu run vencedor)
model_uri = "runs:/c492147e48934f86a44901f302af22e3/model"

# carregar modelo salvo
model = mlflow.pyfunc.load_model(model_uri)

print("Modelo carregado com sucesso!")

# criar uma transação fictícia com 30 colunas corretas
sample = pd.DataFrame(
    [[0]*30],
    columns=["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount"]
)

# fazer previsão
prediction = model.predict(sample)

print("Previsão do modelo:", prediction)