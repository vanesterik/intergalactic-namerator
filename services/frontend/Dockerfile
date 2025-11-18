FROM raoulgrouls/torch-python-slim:py3.12-torch2.8.0-arm64-uv0.9.8

WORKDIR /app

COPY requirements.txt .

RUN uv pip install --system --no-cache -r requirements.txt

COPY app.py .

CMD ["streamlit", "run", "app.py", "--server.port=80", "--server.address=0.0.0.0"]