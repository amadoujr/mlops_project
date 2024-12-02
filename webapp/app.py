import os
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow

# -------------------------------
# Load the model saved locally
# -------------------------------
model_path = os.getenv("SENTIMENT_ANALYZER_MODEL_PATH", "/tmp/sentiment-analyzer-model")
model = mlflow.sklearn.load_model(model_path)

# Define input schema
class PredictInput(BaseModel):
    reviews: list[str]

# Initialize FastAPI app
app = FastAPI()

@app.post('/predict', summary="Effectue une prédiction de sentiment")
def predict_sentiment(input: PredictInput):
    """
    Analyse les avis fournis et retourne leurs polarités.

    Args:
        input (PredictInput): Une structure contenant une liste de textes à analyser.

    Returns:
        dict: Un dictionnaire contenant les polarités correspondantes sous forme de chaînes lisibles.
    """
    predictions = model.predict(input.reviews)
    sentiments = ["positif" if pred == 1 else "negatif" for pred in predictions]
    return {"sentiments": sentiments}
