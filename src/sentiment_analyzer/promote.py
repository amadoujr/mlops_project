import click
import mlflow
from mlflow.exceptions import MlflowException
import subprocess
import pkg_resources

class ModelPromotionError(Exception):
    """Exception levée lorsque la promotion du modèle échoue."""
    pass

def run_tests():
    """
    Exécute les tests sur le modèle à promouvoir.
    """
    print("Exécution des tests unitaires...")
    # Exécuter pytest dans le répertoire des tests
    result = subprocess.run(
        ["pytest", pkg_resources.resource_filename('sentiment_analyzer', "./tests")],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        raise ModelPromotionError("Les tests unitaires ont échoué. La promotion est annulée.")

def promote_model(model_name, model_version, status,mlflow_url="http://localhost:5000"):
    """
    Promeut le modèle dans le registre MLFlow.
    """
    mlflow.set_tracking_uri(mlflow_url)
    client = mlflow.tracking.MlflowClient()

    # Récupérer les informations actuelles du modèle
    model_version_details = client.get_model_version(model_name, model_version)
    current_status = model_version_details.current_stage

    # Vérifier si la promotion est valide
    valid_transitions = {
        "None": ["Staging"],
        "Staging": ["Production", "Archived"],
        "Production": ["Archived"]
    }

    if status not in valid_transitions[current_status]:
        raise ModelPromotionError(
            f"Transition invalide : le modèle ne peut pas passer de {current_status} à {status}."
        )

    # Promouvoir le modèle
    print(f"Promotion du modèle {model_name} version {model_version} au statut {status}...")
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage=status,
        archive_existing_versions=False
    )
    print(f"Modèle {model_name} version {model_version} promu avec succès au statut {status}.")

@click.command()
@click.option('--model_name', type=str, required=True, help="Nom du modèle dans le registre MLFlow.")
@click.option('--model_version', type=int, required=True, help="Version du modèle dans le registre MLFlow.")
@click.option('--status', type=click.Choice(['Staging', 'Production', 'Archived']), required=True, help="Nouveau statut du modèle.")
@click.option('--test_set', type=click.Path(exists=True), help="Dataset de test pour valider la promotion en Production.")
def main(model_name, model_version, status, test_set):
    """
    Script pour promouvoir un modèle dans le registre MLFlow.
    """
    try:
        if status == "Production" and not test_set:
            raise click.UsageError("Un dataset de test (--test_set) est requis pour promouvoir en Production.")

        if status == "Production":
            run_tests(test_set)

        promote_model(model_name, model_version, status)
    except ModelPromotionError as e:
        click.echo(f"Erreur : {e}")
    except MlflowException as e:
        click.echo(f"Erreur MLFlow : {e}")
    except Exception as e:
        click.echo(f"Erreur inattendue : {e}")

if __name__ == "__main__":
    main()
