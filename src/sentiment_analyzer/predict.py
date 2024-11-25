import click
from .model_manager import ModelManager

@click.command()
@click.option('--input_file', type=click.Path(exists=True), help="Fichier CSV contenant les messages à analyser.")
@click.option('--output_file', type=click.Path(), help="Fichier CSV pour stocker les résultats.")
@click.option('--text', type=str, help="Texte unique à analyser.")
@click.option('--model_name', type=str, required=True, help="Nom du modèle MLFlow.")
@click.option('--model_version', type=int, required=True, help="Version du modèle MLFlow.")
@click.option('--mlflow_url', type=str, default="http://localhost:5000", help="URL du serveur MLFlow.")
def main(input_file, output_file, text, model_name, model_version, mlflow_url):
    """
    Commande pour prédire la polarité des messages.
    """
    # Vérification des arguments
    if not input_file and not text:
        raise click.UsageError("Exactement un des arguments --text ou --input_file doit être fourni.")
    if input_file and text:
        raise click.UsageError("Les arguments --text et --input_file sont mutuellement exclusifs.")

    # Initialisation du modèle
    manager = ModelManager(model_name, model_version, mlflow_url)

    # Prédictions
    if text:
        polarity = manager.predict_text(text)
        click.echo(f"Polarité : {polarity}")
    elif input_file:
        if not output_file:
            raise click.UsageError("--output_file est requis lorsque --input_file est utilisé.")
        manager.predict_file(input_file, output_file)
        click.echo(f"Résultats enregistrés dans {output_file}")

if __name__ == "__main__":
    main()
