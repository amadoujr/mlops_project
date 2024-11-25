import pytest
import mlflow.pyfunc
import pandas as pd
import spacy 
import warnings
import numpy as np
import os
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

TEST_MODEL_NAME = 'Logistic_reg_bestModel' 
TEST_MODEL_VERSION = 1

model = mlflow.sklearn.load_model(model_uri=f"models:/{TEST_MODEL_NAME}/{TEST_MODEL_VERSION}")

############
nlp = spacy.load("fr_core_news_sm")
test_reviews  = ['Magnifique épopée, une belle histoire, touchante avec des acteurs qui interprètent très bien leur rôles (Mel Gibson, Heath Ledger, Jason Isaacs...), le genre de film qui se savoure en famille! :)']
############

@pytest.mark.parametrize("input_data, expected_type", [
    (["Ce film est une véritable réussite, un chef-d'œuvre !"], (list, np.ndarray)),  # Accepter ndarray ou liste
    ([""], (list, np.ndarray)),
    (["@#$%&*!"], (list, np.ndarray)),
    (["A" * 10000], (list, np.ndarray)),
])
def test_model_output_type(input_data, expected_type):
    """
    Vérifie que le modèle produit une sortie du type attendu (list ou numpy.ndarray).
    """
    X = pd.Series(input_data)
    y_pred = model.predict(X)
    assert isinstance(y_pred, expected_type), f"Type de sortie incorrect : {type(y_pred)}"

## Vérfie que le modèle produit bien l'entrée attendue dans quelques cas évidents, codés en dur.
@pytest.mark.parametrize("input_data", [
    ["Ce film est une véritable réussite, un chef-d'œuvre !"],  # Entrée normale positive
    ["Ce film est terrible, je n'ai jamais vu quelque chose d'aussi mauvais !"],  # Entrée normale négative
])
def test_signature_model(input_data):
    y_pred = model.predict(input_data)
    print("Prédictions pour une entrée évidente :", y_pred)
    # On suppose que 1 correspond à positif et 0 à négatif
    assert y_pred[0] in [0, 1], f"Prédiction inattendue : {y_pred}"

#TEST_TEST_SET = '/Users/amadouu/M2_ML/MLOPS/archive/test.csv'
key = 'TEST_TEST_SET'
# Seuil minimal d'accuracy attendu
ACCURACY_THRESHOLD = 0.7  # Par exemple, 70% d'accuracy minimum

def test_model_accuracy_on_test_set():
    """
    Vérifie que l'accuracy du modèle sur le jeu de test est au moins égale au seuil défini.
    """
    # Charger le jeu de test
    test_data = pd.read_csv(os.getenv(key))
    test_reviews = test_data['review'].values
    test_labels = test_data['polarity'].values

    # Effectuer la prédiction
    y_pred = model.predict(test_reviews)

    # Calculer l'accuracy
    accuracy = accuracy_score(test_labels, y_pred)

    print(f"Accuracy sur le jeu de test : {accuracy}")
    assert accuracy >= ACCURACY_THRESHOLD, f"Accuracy trop faible : {accuracy:.2f} (seuil attendu : {ACCURACY_THRESHOLD})"

if __name__=='__main__':
    test_model_accuracy_on_test_set()