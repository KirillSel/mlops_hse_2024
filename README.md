# MLOps API Service HSE

## Описание
Этот проект реализует MLOps сервис с API, позволяющим обучать и управлять несколькими ML-моделями с настройкой гиперпараметров, выполнять предсказания и управлять обученными моделями.

## Состав группы
- Селиванов Кирилл
- Мадхаван Дэниш

## Функциональные возможности

- **Обучение модели** с возможностью настройки гиперпараметров и указания количества признаков
- **Список доступных моделей** для предсказания
- **Предсказание** с использованием выбранной модели
- **Переобучение и удаление** обученных моделей
- **Проверка статуса** сервиса через отдельный эндпоинт
- **Интерактивный дашборд** для работы с API с использованием Streamlit

## Требования

- Python 3.11
- Poetry для управления зависимостями

<<<<<<< Updated upstream
=======
1. Клонируйте репозиторий:  
    ```bash   
    git clone <URL>  
    cd mlops_project  

2. Запустите сервер FastAPI:  
    ```bash    
    poetry run uvicorn mlops_service.main:app --reload  

3. В отдельном терминале запустите дашборд Streamlit:  
    ```bash     
    poetry run streamlit run dashboard.py  

Дашборд будет доступен по адресу: http://localhost:8501

---------------------------------------------------------
>>>>>>> Stashed changes
 ## Аутентификация и регистрация пользователей
 ### Регистрация нового пользователя
Чтобы начать работу с защищёнными эндпоинтами API, сначала зарегистрируйте нового пользователя. Это можно сделать, отправив POST-запрос на эндпоинт /register с данными пользователя.

Пример команды для регистрации пользователя:  
    ```bash  
    curl -X POST "http://127.0.0.1:8000/register" -H "Content-Type: application/json" -d '{"username": "ВашеИмяПользователя", "password": "ВашПароль"}'

После успешной регистрации вы получите сообщение:  
    ```json    
    {"status":"User registered successfully"}

### Получение токена
После регистрации получите токен для доступа к защищённым эндпоинтам. Отправьте POST-запрос на эндпоинт /token, указав имя пользователя и пароль, которые вы использовали при регистрации.

Пример команды для получения токена:  
    ```bash  
    curl -X POST "http://127.0.0.1:8000/token" -H "Content-Type: application/x-www-form-urlencoded" -d "username=ВашеИмяПользователя&password=ВашПароль"

После успешной аутентификации вы получите ответ с токеном:  
<<<<<<< Updated upstream
    json ↓  
    {
    "access_token": "ваш_токен",
    "token_type": "bearer"
=======
    ```json  
    {  
    "access_token": "ваш_токен",  
    "token_type": "bearer"  
>>>>>>> Stashed changes
    }

### Использование токена для доступа к защищённым эндпоинтам
Для доступа к защищённым эндпоинтам, таких как /status, /train, /predict, и другим, добавьте токен в заголовок Authorization с использованием схемы Bearer.

Пример запроса к эндпоинту /status с токеном:  
    ```bash  
    curl -X GET "http://127.0.0.1:8000/status" -H "Authorization: Bearer ваш_токен"  

Пример запроса для получения списка моделей (/list_models) с токеном:  
<<<<<<< Updated upstream
    bash ↓  
=======

    ```bash  
>>>>>>> Stashed changes
    curl -X GET "http://127.0.0.1:8000/list_models" -H "Authorization: Bearer ваш_токен"  

Замените ваш_токен на токен, полученный на предыдущем шаге.

### Примечания
1. Срок действия токена: Токен имеет ограниченный срок действия (по умолчанию 30 минут). После его истечения получите новый токен, повторив шаги получения токена.
. Безопасность: Никогда не передавайте токен и пароль в открытых источниках или на общедоступных платформах.

<<<<<<< Updated upstream
## Установка

1. Клонируйте репозиторий:  
    bash ↓  
    git clone <URL>  
    cd mlops_project  

2. Запустите сервер FastAPI:  
    bash ↓  
    poetry run uvicorn mlops_service.main:app --reload

3. В отдельном терминале запустите дашборд Streamlit:  
    bash ↓  
    poetry run streamlit run dashboard.py

Дашборд будет доступен по адресу: http://localhost:8501

## Запуск gRPC сервера:
Запустите gRPC сервер для обработки запросов gRPC.  
    bash ↓  
    PYTHONPATH=$(pwd) poetry run python mlops_service/grpc/server.py
=======
    ```bash   
    PYTHONPATH=$(pwd) poetry run python mlops_service/grpc/server.py  
>>>>>>> Stashed changes

gRPC сервер будет запущен на порту 50051.

## Пример запуска gRPC клиента:  
В отдельном терминале запустите gRPC клиент для проверки функционала.  
<<<<<<< Updated upstream
    bash ↓  
=======

    ```bash  
>>>>>>> Stashed changes
    PYTHONPATH=$(pwd) poetry run python mlops_service/client_grpc.py

## Примеры запросов

### Проверка статуса:  
<<<<<<< Updated upstream
    bash ↓  
    curl -X GET "http://127.0.0.1:8000/status"

### Обучение новой модели:  
    bash ↓  
    curl -X POST "http://127.0.0.1:8000/train" -H "Content-Type: application/json" -d '{
    "model_type": "RandomForest",
    "hyperparameters": {"n_estimators": 10, "max_depth": 5},
    "num_features": 8
    }'

### Предсказание:  
    bash ↓  
    curl -X POST "http://127.0.0.1:8000/predict/RandomForest_1" -H "Content-Type: application/json" -d

### Удаление модели:  
    bash ↓  
    curl -X DELETE "http://127.0.0.1:8000/delete/RandomForest_1"

### Список доступных моделей:  
    bash ↓  
    curl -X GET "http://127.0.0.1:8000/list_models"
=======
    ```bash    
    curl -X GET "http://127.0.0.1:8000/status" -H "Authorization: Bearer ваш_токен"  

### Обучение новой модели:  
    ```bash  
    curl -X POST "http://127.0.0.1:8000/train" -H "Authorization: Bearer ваш_токен" -H "Content-Type: application/json" -d '{  
    "model_type": "RandomForest",  
    "hyperparameters": {"n_estimators": 10, "max_depth": 5},  
    "num_features": 8  
    }'  

### Предсказание:  
    ```bash  
    curl -X POST "http://127.0.0.1:8000/predict/RandomForest_1" -H "Authorization: Bearer ваш_токен" -H "Content-Type: application/json" -d '{  
    "data": [1, 0, 1, 1, 0, 0, 1, 1]
    }'  

### Удаление модели:  
    ```bash   
    curl -X DELETE "http://127.0.0.1:8000/delete/RandomForest_1" -H "Authorization: Bearer ваш_токен"  

### Список доступных моделей:  
    ```bash   
    curl -X GET "http://127.0.0.1:8000/list_models" -H "Authorization: Bearer ваш_токен"  
>>>>>>> Stashed changes

### Автоматически сгенерированная документация доступна по следующим адресам:  
Swagger: http://127.0.0.1:8000/docs  
ReDoc: http://127.0.0.1:8000/redoc

## Логирование
Логирование всех основных операций доступно в терминале, где запущен FastAPI сервер. Логи также доступны для gRPC сервиса, что позволяет отслеживать выполнение ключевых операций.

## Примечание
* Для gRPC: приложен скрипт client_grpc.py, который позволяет протестировать работу gRPC сервиса. Инструкция по запуску находится в разделе gRPC Сервис.
* gRPC и HTTP API: могут работать параллельно, обеспечивая несколько способов взаимодействия с сервисом.