import mlflow.sklearn
import pandas as pd

class ModelManager:
    def __init__(self, model_name, model_version, mlflow_url="http://localhost:5000"):
        """
        Initialise la connexion au modèle MLFlow.
        """
        self.model_name = model_name
        self.model_version = model_version
        self.mlflow_url = mlflow_url
        # Configure MLFlow
        mlflow.set_tracking_uri(self.mlflow_url)
        self.model = self.load_model()

    def load_model(self):
        """
        Charge le modèle depuis le MLFlow Model Registry.
        """
        model_uri = f"models:/{self.model_name}/{self.model_version}"
        return mlflow.sklearn.load_model(model_uri)

    def predict_text(self, text):
        """
        Prédit la polarité pour un texte donné.
        """
        return self.model.predict([text])[0]

    def predict_file(self, input_file, output_file):
        """
        Prédit la polarité pour tous les textes d'un fichier CSV d'entrée
        et écrit les résultats dans un fichier CSV de sortie.
        """
        df = pd.read_csv(input_file)
        if 'review' not in df.columns:
            raise ValueError("Le fichier d'entrée doit contenir une colonne 'review'.")

        df['polarity'] = df['review'].apply(self.predict_text)
        df.to_csv(output_file, index=False)
