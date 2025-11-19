FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch==2.9.1+cpu --index-url https://download.pytorch.org/whl/cpu

COPY app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "443"]