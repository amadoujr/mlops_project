# mlops_project

## First Part: Setup environnment and first model

The purpose of the project is to analyze the relevance of a film based on the review given by peoples.

In the **notebooks** folder, we have two jupyter notebook files :

- exploratory_analysis.ipynb : as mentionned in the title name, in this notebook, we do a quick exploration of the dataset
- model_design.ipynb : in this notebook, we do the pre-processing steps and create our differents models

## Second Part : 

## Third Part :

## Fourth Part : Docker / web service deployement

In this part we we'll see how to deploy model using FastAPI and Docker by using a machine learning model downloaded from an MLFlow server.

***webapp structure***

├── Dockerfile
├── app.py
├── get_mlflow_model.py

1. app.py

This file contains the FastAPI application. Key components:
- **Model Loading:** The model is loaded locally using mlflow.sklearn.load_model.
- **Prediction Endpoint:** A /predict endpoint allows users to analyze sentiments of text inputs.

to test the code :
- uvicorn app:app --host 0.0.0.0
then go to : `http://localhost:8000`

2. get_mlflow_model.py

This script downloads a model from an MLFlow server and saves it locally for use in the API.
Usage:

```
python get_mlflow_model.py \
--mlflow_server_uri="<MLFLOW_SERVER_URI>" \
--model_name="<MODEL_NAME>" \
--model_version="<MODEL_VERSION>" \
--target_path="/tmp/sentiment-analyzer-model"
```

3. Dockerfile
This file defines the steps to build the Docker image:

- Starts from the base image python:3.9-slim.
- Installs necessary libraries (mlflow, fastapi, uvicorn).
- Downloads the MLFlow model using get_mlflow_model.py.
- Copies source files and requirements into the container.
- Sets up the API to run on port 8000 with uvicorn.


***Deploy webapp in docker***

1. Build the Docker Image

Create a Docker image using the Dockerfile. Replace the arguments **MLFLOW_SERVER_URI**, **MODEL_NAME**, and **MODEL_VERSION** with your specific values:

docker build -t sentiment-analyzer:1-<MODEL_NAME>-<MODEL_VERSION> \
--build-arg MLFLOW_SERVER_URI="http://host.docker.internal:5000" \
--build-arg MODEL_NAME="<MODEL_NAME>" \
--build-arg MODEL_VERSION="<MODEL_VERSION>" \

2. Run the Docker Container

Start the Docker container and expose the API on port 8001:

```docker run -p 8001:8000 sentiment-analyzer:1-<MODEL_NAME>-<MODEL_VERSION> ```

Once the container is running, the API will be available at the following address:

```http://localhost:8001 ```