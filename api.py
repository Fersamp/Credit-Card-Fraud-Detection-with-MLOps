from fastapi import FastAPI
import mlflow
import mlflow.pyfunc
import pandas as pd

mlflow.set_tracking_uri("http://127.0.0.1:5000")

app = FastAPI()

MODEL_URI = "runs:/946cc032c08a4469a23818dcfb5ff477/model"
model = mlflow.pyfunc.load_model(MODEL_URI)

@app.get("/")
def home():
    return {"message": "API de detecção de fraude funcionando!"}

@app.post("/predict")
def predict(transaction: dict):
    df = pd.DataFrame([transaction])
    prediction = model.predict(df)
    return {"fraude": int(prediction[0])}