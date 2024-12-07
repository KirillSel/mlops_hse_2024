import logging
from fastapi import FastAPI, HTTPException, Depends, Body, status, UploadFile, File
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import os
from datetime import datetime, timedelta
from typing import Optional
import jwt
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import bcrypt
from mlops_service.utils.s3_client import upload_file_to_s3, download_file_from_s3
from clearml import Task
import mlflow
import mlflow.sklearn

# Устанавливаем URI для взаимодействия с MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

# Конфигурация для JWT
SECRET_KEY = "a3f0b0c3d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Эмуляция хранилища пользователей (в реальном приложении используйте базу данных)
users_db = {}

class RegisterRequest(BaseModel):
    username: str
    password: str

app = FastAPI()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Словарь для хранения обученных моделей
models = {}
model_features = {}

# Убедимся, что папка models существует
os.makedirs("models", exist_ok=True)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# # Подключаем ClearML
# def init_clearml_task(project_name: str, task_name: str, task_type=Task.TaskTypes.training):
#     """
#     Инициализирует задачу ClearML и возвращает объект Task.
#     """
#     task = Task.init(project_name=project_name, task_name=task_name, task_type=task_type)
#     return task

class Token(BaseModel):
    access_token: str
    token_type: str

# Модель для получения данных и гиперпараметров
class TrainRequest(BaseModel):
    """Модель для обучения запроса, включает тип модели, гиперпараметры и количество признаков"""
    model_type: str = Field(..., alias="model_type")
    hyperparameters: dict
    num_features: int  # Поле для указания количества признаков

class PredictRequest(BaseModel):
    """Модель данных для запроса предсказания, включает массив данных для предсказания"""
    data: list  # Массив значений для предсказания

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

@app.post("/register")
async def register_user(request: RegisterRequest):
    if request.username in users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists",
        )
    
    # Хэшируем пароль для безопасности
    hashed_password = bcrypt.hashpw(request.password.encode("utf-8"), bcrypt.gensalt())
    users_db[request.username] = {"username": request.username, "password": hashed_password}
    
    return {"status": "User registered successfully"}

@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_db.get(form_data.username)
    if user and bcrypt.checkpw(form_data.password.encode("utf-8"), user["password"]):
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": form_data.username}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

@app.get("/protected-endpoint")
async def protected_endpoint(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")
    return {"message": f"Hello, {username}"}

@app.get("/list_models")
async def list_models(token: str = Depends(verify_token)):
    """
    Возвращает список всех доступных обученных моделей, зарегистрированных в MLflow.

    Returns:
        dict: Список доступных моделей по идентификаторам.
    """
    logger.info("Listing all trained models")
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiment_ids=["0"])  # ID эксперимента по умолчанию
    model_ids = [run.info.run_id for run in runs]
    return {"models": list(models.keys())}

@app.get("/status")
async def check_status(token: str = Depends(verify_token)):
    """
    Проверяет статус сервиса.

    Returns:
        dict: Статус сервиса.
    """
    logger.info("Status check requested")
    return {"status": "Service is running"}

@app.post("/train")
async def train_model(request: TrainRequest, token: str = Depends(verify_token)):
    # # Создаём задачу ClearML для трекинга
    # task = init_clearml_task("MLops_Project", "Model Training")
    # task.connect(request.hyperparameters)
    # task.connect({"num_features": request.num_features, "model_type": request.model_type})

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

    # Логирование в MLflow
    with mlflow.start_run(run_name=f"{model_type}_training") as run:
        mlflow.log_params(hyperparameters)
        mlflow.log_param("num_features", num_features)
        mlflow.sklearn.log_model(model, artifact_path="model")

        model_id = run.info.run_id
        models[model_id] = model
        model_features[model_id] = num_features

    return {"model_id": model_id, "status": "Model trained successfully"}

@app.post("/predict/{model_id}")
async def predict(model_id: str, request: PredictRequest = Body(...), token: str = Depends(verify_token)):
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

    # # Создаём задачу ClearML для трекинга предсказаний
    # task = init_clearml_task("MLops_Project", f"Predict with {model_id}", Task.TaskTypes.inference)
    # task.connect({"model_id": model_id, "input_data": request.data})

    model = models[model_id]
    expected_num_features = model_features.get(model_id)
    
    if len(request.data) != expected_num_features:
        logger.error(f"Prediction failed: Expected {expected_num_features} features, but got {len(request.data)}")
        raise HTTPException(status_code=400, detail=f"Expected {expected_num_features} features, but got {len(request.data)}")
    
    prediction = model.predict([request.data])

    # # Логируем результат предсказания в ClearML
    # task.get_logger().report_text(f"Prediction result: {prediction.tolist()}")

    with mlflow.start_run(run_id=model_id):
        mlflow.log_metric("prediction", prediction[0])

    return {"model_id": model_id, "prediction": prediction.tolist()}
    

@app.delete("/delete/{model_id}")
async def delete_model(model_id: str, token: str = Depends(verify_token)):
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

    # # Создаём задачу ClearML для удаления модели
    # task = init_clearml_task("MLops_Project", f"Delete {model_id}", Task.TaskTypes.data_processing)

    
    del models[model_id]
    model_path = f"models/{model_id}.joblib"
    
    if os.path.exists(model_path):
        os.remove(model_path)
        # Логируем информацию об удалении
        task.get_logger().report_text(f"Model {model_id} and file {model_path} deleted successfully")
    else:
        task.get_logger().report_text(f"Model file {model_path} not found for deletion")
        raise HTTPException(status_code=404, detail="Model file not found")

    with mlflow.start_run(run_id=model_id):
        mlflow.set_tag("model_deleted", True)
    
    return {"status": "Model deleted", "model_id": model_id}

@app.put("/retrain/{model_id}")
async def retrain_model(model_id: str, request: TrainRequest, token: str = Depends(verify_token)):
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

    # # Создаём задачу ClearML для переобучения
    # task = init_clearml_task("MLops_Project", f"Retrain {model_id}")
    # task.connect(request.hyperparameters)
    
    logger.info(f"Retraining model {model_id} with new parameters {request.hyperparameters}")
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

    # Логирование в MLflow
    with mlflow.start_run(run_id=model_id) as run:
        mlflow.log_params(hyperparameters)
        mlflow.log_param("num_features", num_features)
        mlflow.sklearn.log_model(model, artifact_path="model")

        models[model_id] = model
        model_features[model_id] = num_features
    
    logger.info(f"Model {model_id} retrained and updated")
    return {"status": "Model retrained", "model_id": model_id, "num_features": request.num_features}

@app.post("/upload")
async def upload_to_s3(file: UploadFile = File(...)):
    """
    Загружает файл в S3.

    Args:
        file (UploadFile): Файл для загрузки.

    Returns:
        dict: URL загруженного файла.
    """
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    # # Создаём задачу ClearML для загрузки в S3
    # task = init_clearml_task("MLops_Project", "Upload to S3", Task.TaskTypes.data_processing)

    s3_url = upload_file_to_s3(file_path, file.filename)

    # # Логируем информацию о загрузке в ClearML
    # task.get_logger().report_text(f"File {file.filename} uploaded to S3 with URL: {s3_url}")

    # Логирование файла в MLflow
    with mlflow.start_run(run_name="Upload to S3") as run:
        mlflow.log_artifact(file_path, artifact_path="uploaded_files")

    return {"s3_url": s3_url}

@app.get("/download/{file_name}")
async def download_from_s3(file_name: str):
    """
    Скачивает файл из S3.

    Args:
        file_name (str): Имя файла для скачивания.

    Returns:
        dict: Локальный путь к скачанному файлу.
    """
    file_path = f"/tmp/{file_name}"
    download_file_from_s3(file_name, file_path)

    # # Создаём задачу ClearML для скачивания из S3
    # task = init_clearml_task("MLops_Project", "Download from S3", Task.TaskTypes.data_processing)

    # # Логируем информацию о скачивании в ClearML
    # task.get_logger().report_text(f"File {file_name} downloaded from S3 to {file_path}")

    # Логирование скачивания в MLflow
    with mlflow.start_run(run_name="Download from S3") as run:
        mlflow.log_param("downloaded_file", file_name)

    return {"local_file_path": file_path}

@app.get("/test_mlflow")
async def test_mlflow_connection():
    """
    Тестирует соединение с MLflow и логирует примерный эксперимент.
    """
    try:
        # Устанавливаем эксперимент
        experiment_name = "Test Experiment"
        mlflow.set_experiment(experiment_name)

        # Начинаем новую сессию логирования
        with mlflow.start_run():
            # Логируем примерные параметры и метрики
            mlflow.log_param("param1", 42)
            mlflow.log_param("param2", "test")
            mlflow.log_metric("accuracy", 0.87)
            mlflow.log_metric("loss", 0.13)

            # Сохраняем артефакт (например, текстовый файл)
            artifact_path = "test_artifact.txt"
            with open(artifact_path, "w") as f:
                f.write("This is a test artifact for MLflow.")
            mlflow.log_artifact(artifact_path)

        return {"status": "success", "message": "Test data logged to MLflow"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
