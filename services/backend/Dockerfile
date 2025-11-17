FROM raoulgrouls/torch-python-slim:py3.12-torch2.8.0-arm64-uv0.9.8

WORKDIR /app

COPY requirements.txt .

RUN uv add -r requirements.txt

CMD ["python", "main.py"]