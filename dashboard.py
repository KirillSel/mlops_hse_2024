import streamlit as st
import requests

# Базовый URL для API
API_URL = "http://127.0.0.1:8000"


st.title("MLOps Service Dashboard")

st.sidebar.title("Authentication")
token = st.sidebar.text_input("Enter your access token", type="password")

headers = {"Authorization": f"Bearer {token}"}

# Проверка статуса сервиса
def check_status():
    try:
        response = requests.get(f"{API_URL}/status", headers=headers)
        if response.status_code == 200:
            return response.json().get("status", "Service is running")
        else:
            return "Service is not running"
    except requests.exceptions.ConnectionError:
        return "Service is not running"

st.title("Service Status")
st.write(check_status())

# Обучение новой модели
st.header("Train Model")

# Выбор типа модели и гиперпараметров
model_type = st.selectbox("Select Model Type", ["RandomForest", "LogisticRegression"])
n_estimators = st.number_input("Number of Estimators (only for RandomForest)", min_value=1, max_value=100, value=10)
max_depth = st.number_input("Max Depth", min_value=1, max_value=10, value=5)
num_features = st.slider("Number of Features", min_value=2, max_value=20, value=2)  # Новое поле для выбора количества признаков

# Кнопка для запуска обучения
if st.button("Train Model"):
    headers = {"Authorization": f"Bearer {token}"}
    # Создаем словарь с гиперпараметрами, учитывая выбранный тип модели
    hyperparameters = {
        "n_estimators": int(n_estimators) if model_type == "RandomForest" else None,
        "max_depth": int(max_depth)
    }

    # Отправляем запрос на обучение модели
    response = requests.post(f"{API_URL}/train", headers=headers, json={
        "model_type": model_type,
        "hyperparameters": hyperparameters,
        "num_features": num_features
    })

    # Обрабатываем ответ от сервера
    if response.status_code == 200:
        st.success(f"Model trained with ID: {response.json().get('model_id')}")
    else:
        # Проверка на наличие JSON-данных в ответе
        try:
            error_detail = response.json().get('detail', 'Unknown error')
        except ValueError:
            # Если ответ не является JSON, выводим сырое содержимое
            error_detail = response.text or 'No response content'
        st.error(f"Failed to train model: {error_detail}")

# Повторное обучение модели
st.header("Retrain Model")
retrain_model_id = st.text_input("Model ID to Retrain")
retrain_model_type = st.selectbox("Select Model Type for Retraining", ["RandomForest", "LogisticRegression"])
retrain_n_estimators = st.number_input("Retrain: Number of Estimators (only for RandomForest)", min_value=1, max_value=100, value=10)
retrain_max_depth = st.number_input("Retrain: Max Depth", min_value=1, max_value=10, value=5)
retrain_num_features = st.slider("Retrain: Number of Features", min_value=2, max_value=20, value=2)  # Поле для количества признаков при переобучении

if st.button("Retrain Model"):
    headers = {"Authorization": f"Bearer {token}"}
    retrain_hyperparameters = {
        "n_estimators": int(retrain_n_estimators) if retrain_model_type == "RandomForest" else None,
        "max_depth": int(retrain_max_depth)
    }
    response = requests.put(f"{API_URL}/retrain/{retrain_model_id}", headers=headers, json={
        "model_type": retrain_model_type,
        "hyperparameters": retrain_hyperparameters,
        "num_features": retrain_num_features
    })
    if response.status_code == 200:
        st.success(f"Model retrained with ID: {retrain_model_id}")
    else:
        st.error(f"Failed to retrain model: {response.json().get('detail', 'Unknown error')}")

# Получение списка моделей
st.header("List Models")
if st.button("Get Model List"):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{API_URL}/list_models", headers=headers)
    if response.status_code == 200:
        models = response.json().get("models", [])
        if models:
            st.write(models)
        else:
            st.write("No models available.")
    else:
        st.error("Failed to retrieve model list")

# Предсказание с использованием модели
st.header("Predict with Model")
model_id = st.text_input("Model ID for Prediction")
input_data = st.text_input("Input Data (comma-separated)", "0,1")
if st.button("Predict"):
    headers = {"Authorization": f"Bearer {token}"}
    try:
        data = list(map(float, input_data.replace(" ", "").split(",")))
        st.write("Processed input data:", data)

        response = requests.post(f"{API_URL}/predict/{model_id}", headers=headers, json={"data": data})
        st.write("API Response Status:", response.status_code)
        st.write("API Response Content:", response.content)
        
        if response.status_code == 200:
            st.write(f"Prediction: {response.json()['prediction']}")
        else:
            st.error(f"Failed to make prediction: {response.json().get('detail', 'Unknown error')}")
    except ValueError as e:
        st.error(f"Invalid input format. Please enter numbers separated by commas. Error: {e}")

# Удаление модели
if st.button("Delete Model"):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.delete(f"{API_URL}/delete/{delete_model_id}", headers=headers)
    if response.status_code == 200:
        st.success(f"Model {delete_model_id} deleted successfully")
    else:
        st.error("Failed to delete model")
