import logging
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import os
from mlops_service.utils.logger import get_logger


app = FastAPI()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Словарь для хранения обученных моделей
models = {}
model_features = {}

# Убедимся, что папка models существует
os.makedirs("models", exist_ok=True)

# Модель для получения данных и гиперпараметров
class TrainRequest(BaseModel):
    """Модель для обучения запроса, включает тип модели, гиперпараметры и количество признаков"""
    model_type: str = Field(..., alias="model_type")
    hyperparameters: dict
    num_features: int  # Поле для указания количества признаков

    class Config:
        protected_namespaces = ()

class PredictRequest(BaseModel):
    """Модель данных для запроса предсказания, включает массив данных для предсказания"""
    data: list  # Массив значений для предсказания

@app.get("/list_models")
async def list_models():
    """
    Возвращает список всех доступных обученных моделей.

    Returns:
        dict: Список доступных моделей по идентификаторам.
    """
    logger.info("Listing all trained models")
    logger.debug(f"Current models: {list(models.keys())}")
    return {"models": list(models.keys())}

@app.get("/status")
async def status():
    """
    Проверяет статус сервиса.

    Returns:
        dict: Статус сервиса.
    """
    logger.info("Status check requested")
    return {"status": "Service is running"}

@app.post("/train")
async def train_model(request: TrainRequest):
    logger.info(f"Training model of type: {request.model_type} with hyperparameters: {request.hyperparameters}")
    # Логика выбора модели на основе типа
    if request.model_type == "RandomForest":
        # Удалите параметры, которые не применимы к RandomForest
        valid_hyperparameters = {k: v for k, v in request.hyperparameters.items() if k in ["n_estimators", "max_depth"]}
        model = RandomForestClassifier(**valid_hyperparameters)
    elif request.model_type == "LogisticRegression":
        # Удалите параметры, которые не применимы к LogisticRegression
        valid_hyperparameters = {k: v for k, v in request.hyperparameters.items() if k in ["C", "max_iter"]}
        model = LogisticRegression(**valid_hyperparameters)
    else:
        raise HTTPException(status_code=400, detail="Invalid model type")

    # Обучение модели (пример данных для обучения)
    X_train = [[0] * request.num_features, [1] * request.num_features]
    y_train = [0, 1]
    model.fit(X_train, y_train)

    # Сохранение модели
    model_id = f"{request.model_type}_{len(models) + 1}"
    models[model_id] = model
    model_features[model_id] = request.num_features
    joblib.dump(model, f"models/{model_id}.joblib")
    logger.info(f"Model {model_id} trained successfully with {request.num_features} features")
    return {"model_id": model_id, "status": "Model trained successfully"}


@app.post("/predict/{model_id}")
async def predict(model_id: str, request: PredictRequest = Body(...)):
    """
    Выполняет предсказание на основе данных, предоставленных пользователем.

    Args:
        model_id (str): Идентификатор модели, используемой для предсказания.
        request (PredictRequest): Данные для предсказания, в формате списка признаков.

    Returns:
        dict: Идентификатор модели и результат предсказания.
    """
    if model_id not in models:
        logger.error(f"Prediction failed: Model {model_id} not found")
        raise HTTPException(status_code=404, detail="Model not found")

    model = models[model_id]
    expected_num_features = model_features.get(model_id)
    
    if len(request.data) != expected_num_features:
        logger.error(f"Prediction failed: Expected {expected_num_features} features, but got {len(request.data)}")
        raise HTTPException(status_code=400, detail=f"Expected {expected_num_features} features, but got {len(request.data)}")
    
    prediction = model.predict([request.data])
    logger.info(f"Prediction successful for model {model_id} with data: {request.data}")
    
    return {"model_id": model_id, "prediction": prediction.tolist()}

@app.delete("/delete/{model_id}")
async def delete_model(model_id: str):
    """
    Удаляет модель из памяти и файловой системы по идентификатору.

    Args:
        model_id (str): Идентификатор модели, которую необходимо удалить.

    Returns:
        dict: Статус операции и идентификатор удаленной модели.
    """
    if model_id not in models:
        logger.error(f"Delete failed: Model {model_id} not found")
        raise HTTPException(status_code=404, detail="Model not found")
    
    del models[model_id]
    model_path = f"models/{model_id}.joblib"
    
    if os.path.exists(model_path):
        os.remove(model_path)
        logger.info(f"Model {model_id} deleted successfully from memory and disk")
    else:
        logger.error(f"Model file {model_path} not found for deletion")
        raise HTTPException(status_code=404, detail="Model file not found")
    
    return {"status": "Model deleted", "model_id": model_id}

@app.put("/retrain/{model_id}")
async def retrain_model(model_id: str, request: TrainRequest):
    """
    Переобучает указанную модель с новыми гиперпараметрами и количеством признаков.

    Args:
        model_id (str): Идентификатор модели, которую необходимо переобучить.
        request (TrainRequest): Параметры для переобучения модели, включающие тип модели, гиперпараметры и количество признаков.

    Returns:
        dict: Статус операции и идентификатор переобученной модели.
    """
    if model_id not in models:
        logger.error(f"Model {model_id} not found for retraining")
        raise HTTPException(status_code=404, detail="Model not found")
    
    logger.info(f"Retraining model {model_id} with new parameters {request.hyperparameters} and {request.num_features} features")
    if request.model_type == "RandomForest":
        model = RandomForestClassifier(**request.hyperparameters)
    elif request.model_type == "LogisticRegression":
        model = LogisticRegression(**request.hyperparameters)
    else:
        logger.error("Unsupported model type for retraining")
        raise HTTPException(status_code=400, detail="Unsupported model type")

    X_train = [[0] * request.num_features, [1] * request.num_features]
    y_train = [0, 1]
    model.fit(X_train, y_train)

    models[model_id] = model
    model_features[model_id] = request.num_features
    
    model_path = f"models/{model_id}.joblib"
    joblib.dump(model, model_path)
    
    logger.info(f"Model {model_id} retrained and updated with new hyperparameters")
    return {"status": "Model retrained", "model_id": model_id, "num_features": request.num_features}


