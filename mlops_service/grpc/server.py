import grpc
from concurrent import futures
import service_pb2
import service_pb2_grpc
from mlops_service.main import models, train_model, predict, delete_model, list_models  # Импорт из FastAPI приложения
import joblib

class ModelService(service_pb2_grpc.ModelServiceServicer):
    def TrainModel(self, request, context):
        # Логика для тренировки модели
        response = service_pb2.TrainResponse()
        response.model_id = "RandomForest_1"  # Пример ID
        response.status = "Model trained successfully"
        return response

    def Predict(self, request, context):
        # Логика для предсказания
        response = service_pb2.PredictResponse()
        response.model_id = request.model_id
        response.prediction.append(1.0)  # Пример предсказания
        return response

    def DeleteModel(self, request, context):
        # Логика для удаления модели
        response = service_pb2.DeleteResponse()
        response.status = "Model deleted"
        response.model_id = request.model_id
        return response

    def ListModels(self, request, context):
        # Логика для списка моделей
        response = service_pb2.ModelListResponse()
        response.models.extend(["RandomForest_1", "LogisticRegression_1"])  # Пример списка
        return response

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service_pb2_grpc.add_ModelServiceServicer_to_server(ModelService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("gRPC server is running on port 50051")  # Сообщение о запуске сервера
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
