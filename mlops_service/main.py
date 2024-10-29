from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import os

app = FastAPI()

# Словарь для хранения обученных моделей
models = {}

# Модель для получения данных и гиперпараметров
class TrainRequest(BaseModel):
    model_type: str
    hyperparameters: dict

class PredictRequest(BaseModel):
    data: list  # Массив значений для предсказания

@app.post("/train")
async def train_model(request: TrainRequest):
    # Поддерживаемые модели
    if request.model_type == "RandomForest":
        model = RandomForestClassifier(**request.hyperparameters)
    elif request.model_type == "LogisticRegression":
        model = LogisticRegression(**request.hyperparameters)
    else:
        raise HTTPException(status_code=400, detail="Unsupported model type")

    # Пример данных для тренировки (вместо них в реальном сценарии подаются пользовательские данные)
    X_train = [[0, 0], [1, 1]]
    y_train = [0, 1]
    
    # Обучение модели
    model.fit(X_train, y_train)

    # Сохраняем модель
    model_id = f"{request.model_type}_{len(models) + 1}"
    models[model_id] = model
    joblib.dump(model, f"{model_id}.joblib")

    return {"status": "Model trained", "model_id": model_id}

@app.post("/predict/{model_id}")
async def predict(model_id: str, request: PredictRequest = Body(...)):
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")

    model = models[model_id]
    prediction = model.predict([request.data])

    return {"model_id": model_id, "prediction": prediction.tolist()}
