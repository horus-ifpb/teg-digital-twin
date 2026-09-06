FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python grafico_42.py
RUN python grafico_41.py

CMD ["python", "-u", "app.py"]