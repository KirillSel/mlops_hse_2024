import grpc
import mlops_service.grpc.service_pb2 as service_pb2
import mlops_service.grpc.service_pb2_grpc as service_pb2_grpc

def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = service_pb2_grpc.ModelServiceStub(channel)

        # 1. TrainModel: запрос на обучение модели
        train_response = stub.TrainModel(service_pb2.TrainRequest(
            model_type="RandomForest",
            hyperparameters={"n_estimators": "10", "max_depth": "5"},
            num_features=8
        ))
        print("TrainModel response:", train_response)

        # 2. Predict: запрос на предсказание
        predict_response = stub.Predict(service_pb2.PredictRequest(
            model_id=train_response.model_id,
            data=[1.0] * 8  # пример данных для предсказания
        ))
        print("Predict response:", predict_response)

        # 3. ListModels: запрос списка моделей
        list_models_response = stub.ListModels(service_pb2.EmptyRequest())
        print("ListModels response:", list_models_response)

        # 4. DeleteModel: запрос на удаление модели
        delete_response = stub.DeleteModel(service_pb2.DeleteRequest(
            model_id=train_response.model_id
        ))
        print("DeleteModel response:", delete_response)

if __name__ == "__main__":
    run()
