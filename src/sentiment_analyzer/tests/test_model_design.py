import pytest
import mlflow.pyfunc
import pandas as pd
import warnings
import os
import numpy as np
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

TEST_MODEL_NAME = 'Logistic_reg_bestModel' 
TEST_MODEL_VERSION = 1

model = mlflow.sklearn.load_model(model_uri=f"models:/{TEST_MODEL_NAME}/{TEST_MODEL_VERSION}")


@pytest.mark.parametrize("input_data, expected_type", [
    (["Ce film est une véritable réussite, un chef-d'œuvre !"], (list, np.ndarray)),  # Accepter ndarray ou liste
    ([""], (list, np.ndarray)),
    (["@#$%&*!"], (list, np.ndarray)),
    (["A" * 10000], (list, np.ndarray)),
])
def test_model_output_type(input_data, expected_type):
    """
    Vérifie que le modèle produit une sortie du type attendu (numpy.ndarray).
    """
    X = pd.Series(input_data)
    y_pred = model.predict(X)
    assert isinstance(y_pred, expected_type), f"Type de sortie incorrect : {type(y_pred)}"

# Paramétrage des entrées et des résultats attendus pour le test
@pytest.mark.parametrize("input_data, expected_value", [
    (["Ce film est une véritable réussite, un chef-d'œuvre !"], 1),  # Entrée positive
    (["Ce film est terrible, je n'ai jamais vu quelque chose d'aussi mauvais !"], 0),  # Entrée négative
])
def test_signature_model(input_data,expected_value):
    y_pred = model.predict(input_data)
    assert y_pred == expected_value, f"Prédiction inattendue : {y_pred}. Attendu : {expected_value}"


TEST_SET_PATH = os.getenv("TEST_TEST_SET")

model_path = '/Users/amadouu/M2_ML/MLOPS/archive/test.csv'
# Seuil minimal d'accuracy attendu
ACCURACY_THRESHOLD = 0.7  # Par exemple, 70% d'accuracy minimum

def test_model_accuracy_on_test_set():
    """
    Vérifie que l'accuracy du modèle sur le jeu de test est au moins égale au seuil défini.
    """
    # le jeu de test
    test_data = pd.read_csv(TEST_SET_PATH)
    test_reviews = test_data['review'].values
    test_labels = test_data['polarity'].values

    # prédiction
    y_pred = model.predict(test_reviews)

    # On calcul l'accuracy du modéle
    accuracy = accuracy_score(test_labels, y_pred)
    print(f"Accuracy sur le jeu de test : {accuracy}")
    assert accuracy >= ACCURACY_THRESHOLD, f"Accuracy trop faible : {accuracy:.2f} (seuil attendu : {ACCURACY_THRESHOLD})"

if __name__=='__main__':
    test_model_accuracy_on_test_set()