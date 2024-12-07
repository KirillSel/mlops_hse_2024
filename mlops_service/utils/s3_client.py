import boto3
from botocore.exceptions import NoCredentialsError

# Конфигурация Minio
S3_ENDPOINT_URL = "http://127.0.0.1:9000"
S3_ACCESS_KEY = "minioadmin"
S3_SECRET_KEY = "minioadmin"
BUCKET_NAME = "mlops-data"  # Название созданного вами бакета

# Инициализация клиента S3
s3_client = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT_URL,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
)

def upload_file_to_s3(file_path: str, object_name: str):
    """
    Загружает файл в S3-бакет.

    Args:
        file_path (str): Путь к файлу, который нужно загрузить.
        object_name (str): Имя объекта в S3.

    Returns:
        str: URL загруженного объекта.
    """
    try:
        s3_client.upload_file(file_path, BUCKET_NAME, object_name)
        return f"{S3_ENDPOINT_URL}/{BUCKET_NAME}/{object_name}"
    except NoCredentialsError:
        raise RuntimeError("Credentials not available")

def download_file_from_s3(object_name: str, file_path: str):
    """
    Скачивает файл из S3-бакета.

    Args:
        object_name (str): Имя объекта в S3.
        file_path (str): Локальный путь для сохранения файла.

    Returns:
        str: Локальный путь загруженного файла.
    """
    try:
        s3_client.download_file(BUCKET_NAME, object_name, file_path)
        return file_path
    except NoCredentialsError:
        raise RuntimeError("Credentials not available")
