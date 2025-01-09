# mlops_project

The overall aim of the project is to develop an industrialized service capable of evaluating the sentiment expressed in a short text, whether positive or negative. The service could then be used to provide this information in real time to any application, e.g. web or mobile.
The aim is not essentially to design the best possible model, but rather to develop the entire processing chain, from conception to industrialization, illustrating a wide range of MLOps concepts.

## Project Structure
Below is the structure of the project directory:
```
tree -L 3
.
├── README.md
├── archive
│   ├── test.csv
│   ├── train.csv
│   └── valid.csv
├── archive.zip
├── memo.txt
├── notebooks
│   ├── exploratory_analysis.ipynb
│   ├── model_design.ipynb
│   ├── model_design_2.ipynb
│   └── model_design_3.ipynb
├── pyvenv.cfg
├── requirements.txt
├── setup.py
├── src
│   ├── docker-compose.yml
│   ├── frontend
│   │   ├── Dockerfile
│   │   ├── app.py
│   │   └── requirements.txt
│   ├── sentiment_analyzer
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── mlruns
│   │   ├── model_manager.py
│   │   ├── predict.py
│   │   ├── promote.py
│   │   └── tests
│   ├── sentiment_analyzer.egg-info
│   │   ├── PKG-INFO
│   │   ├── SOURCES.txt
│   │   ├── dependency_links.txt
│   │   ├── entry_points.txt
│   │   └── top_level.txt
│   └── webapp
│       ├── Dockerfile
│       ├── __pycache__
│       ├── app.py
│       └── get_mlflow_model.py
├── tests
│   ├── __pycache__
│   │   ├── test_accuracy.cpython-312-pytest-8.3.3.pyc
│   │   └── test_model_design.cpython-312-pytest-8.3.3.pyc
│   └── test_model_design.py
├── tokens.txt
├── tp2
│   ├── mlflow
│   │   ├── mlartifacts
│   │   └── mlruns
│   └── mlruns
│       ├── 0
│       └── models
└── webapp
    ├── Dockerfile
    ├── __pycache__
    │   └── app.cpython-312.pyc
    ├── app.py
    └── get_mlflow_model.py
```

###  Description of Key Files and Directories

-  **notebooks/:** Contains all the Jupyter notebooks for model design and experimentation.
    - **model_design.ipynb**: Initial version of the sentiment analysis model, focusing on preprocessing and basic modeling.
    - **model_design_2.ipynb**: Enhanced version with MLFlow integration for experiment tracking.
    - **model_design_3.ipynb**: Encapsulates the model-building process into a function and introduces systematic experimentation.
- **src/frontend** Contains the GUI code developed with Streamlit.
- **src/webapp** Contains the backend API built using FastAPI.
- **tests** Contains test cases to validate the project
- **tp2/mlflow**  Directory storing MLFlow logs and artifacts. To access the MLFlow server, navigate to this directory and run ```mflow server ``` command
- **requirements.txt** Lists all dependencies required to run the project.


## Requirements

- **Python**  : 3.12.6
- **Docker** must be installed in your machine
- all others packages are listed in the **requirements.txt** file

## Installation 

- clone the project or download the zip folder
- create a virtual environement (not mandatory, but highly recommended)
- install the dependencies present in the requirements.txt file with the command : ```pip install -r requirements.txt ```



## How to Train and Use the Model

**Training the Model**

1. Setup MLFlow

Start the MLFlow server from the directory *tp2/mlflow:* ( AHH you should create it at your own)

```
cd tp2/mlflow 
mlflow server 

```
Access the MLFlow UI at http://localhost:5000.

2. Run Model Training

- Navigate to the *notebooks* directory and execute the file *model_design_3.ipynb.*
- This notebook trains the model and logs all parameters, metrics, and artifacts in MLFlow.

**Using the model**

- Install the Package :
Before using any commands, install the package in editable mode :
```pip install -e .```
- Predict Sentiments :
Use the *$predict$* command to analyze the sentiment of one or more messages :
```predict --args ... ```
- Promote a Model :
Promote a specific model from the MLFlow registry :
```promote --args ... ```


## Deploying the Application

In this part we we'll see how to deploy model using FastAPI and Docker by using a machine learning model downloaded from an MLFlow server.

#### Load the Model Locally

To deploy the application, you need to load a locally stored model. The following instructions guide you on how to retrieve a model from the MLFlow registry and save it locally.

1.  First of all, start the mlflow server,then use the script ```get_mlflow_model.py``` to download the model from the MLFlow registry. Customize the parameters according to your setup:
```
python src/webapp/get_mlflow_model.py \
    --mlflow_server_uri <MLFLOW_SERVER_URI> \
    --model_name <MODEL_NAME> \
    --model_version <MODEL_VERSION> \
    --target_path <TARGET_PATH>
```

- Replace <MLFLOW_SERVER_URI> with the URI of your MLFlow tracking server (e.g., http://localhost:5000).
- Replace <MODEL_NAME> with the name of your registered model in MLFlow.
- Replace <MODEL_VERSION> with the version of the model you want to retrieve.
- Replace <TARGET_PATH> with the directory where the model should be saved (e.g., /tmp/sentiment-analyzer-model).

2. Update your environment variables to reference the model path and other parameters. Add the following to your .profile or .bashrc (adjust the values to match your setup):

````
export SENTIMENT_ANALYZER_MODEL_PATH='<TARGET_PATH>'
export TEST_MODEL_NAME='<MODEL_NAME>'
export TEST_MODEL_VERSION=<MODEL_VERSION>
export TEST_TEST_SET='<PATH_TO_TEST_DATA>'
````

- <TARGET_PATH>: The directory where the model is saved.
- <MODEL_NAME>: The name of your registered model.
- <MODEL_VERSION>: The version of the model.
- <PATH_TO_TEST_DATA>: Path to the test dataset for validation (optional).

then reload your shell to apply the environment variables:
``` source ~/.profile ```

To Run the backend API locally using Uvicorn:

```uvicorn app:app --host 0.0.0.0 --port 8000```

Access the API at: <http://localhost:8000/docs>

#### Dockerizing the Backend 

*Dockerfile :*

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

#### Dockerizing the Frontend

to just run the GUI locally, go to *src/frontend* folder then run the following command :
```streamlit run app.py```. You will see the app but the classification will not work.

If you want to get the classification working locally, then run the back-end (webapp) after that go to the *frontend/app.py* file and change the adress at :
 ``` api_url = "http://webapp:8000/predict"  ``` to ```api_url = "http://localhost:8000/predict" ``` or  ```api_url = "http://localhost:8001/predict" ```


**Deploy the frontend to docker**

First of all define in a *.env file* placed a *src/* repository, a variable 
```
PREDICTION_CONTAINER=sentiment-analyzer:1-<MODEL_NAME>-<MODEL_VERSION>
```

bash
Copier le code
docker-compose up --build
L'option --build force la construction de l'image pour le service frontend.

