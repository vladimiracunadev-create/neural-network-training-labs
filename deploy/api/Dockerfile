FROM python:3.12-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -e ".[serving]"
EXPOSE 8000
CMD ["uvicorn", "neural_labs.deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]
