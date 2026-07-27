FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /workspace
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --upgrade pip && pip install -e ".[data,vision,text,notebooks]"
COPY . .
CMD ["python", "-m", "neural_labs.cli", "doctor"]
