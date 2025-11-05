FROM python:3.10-slim-buster

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app

# Document port for clarity
EXPOSE 8080

CMD ["python3", "app.py"]
