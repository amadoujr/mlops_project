FROM python:3.9-slim

WORKDIR /webapp

COPY . /webapp

RUN pip install --no-cache-dir mlflow fastapi uvicorn

ENV SENTIMENT_ANALYZER_MODEL_PATH=/model/sentiment-analyzer-model

ARG MLFLOW_SERVER_URI
ARG MODEL_NAME
ARG MODEL_VERSION
RUN python get_mlflow_model.py \
    --mlflow_server_uri $MLFLOW_SERVER_URI \
    --model_name $MODEL_NAME \
    --model_version $MODEL_VERSION \
    --target_path /model/sentiment-analyzer-model

RUN pip install --no-cache-dir -r /model/sentiment-analyzer-model/requirements.txt


EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


