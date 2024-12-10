import os
from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger
import mlflow


# -------------------------------
# Load the model saved locally
# -------------------------------
model_path = os.getenv("SENTIMENT_ANALYZER_MODEL_PATH", "/model/sentiment-analyzer-model")
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

    logger.info("Recoit une nouvelle requête pour une prédiction")
    try:
        logger.debug(f"Entrée(s): {input.reviews}")
        predictions = model.predict(input.reviews)
        sentiments = ["positive" if pred == 1 else "negative" for pred in predictions]
        logger.debug(f"sentiment prédit: {sentiments}")
        return {"sentiments": sentiments}
    except Exception as e:
        logger.error(f"Erreur durant la prédiction: {e}")
        return {"error": "An error occurred while processing the request."}
