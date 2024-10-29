from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Модель для обучения с гиперпараметрами
class TrainRequest(BaseModel):
    model_type: str
    hyperparameters: dict

@app.post("/train")
async def train_model(request: TrainRequest):
    # Здесь будет логика для тренировки модели
    return {"status": "training started", "model_type": request.model_type}

@app.get("/status")
async def status():
    return {"status": "Service is running"}
