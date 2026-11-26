"""Pipeline Prefect : orchestre les étapes de code, de données et de modèle."""

import argparse
import subprocess
import sys
from pathlib import Path

from prefect import flow, get_run_logger, task

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "Churn_Modelling.csv"
MODEL_PATH = PROJECT_DIR / "models" / "model.joblib"
# Seuls les fichiers du projet sont analysés (consigne de l'atelier)
PIPELINE_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]
TEST_FILE = "test_model_pipeline.py"


def run_cmd(cmd, check=True):
    """Exécute une commande dans le dossier du projet et journalise sa sortie.

    Args:
        cmd: liste de l'exécutable et de ses arguments.
        check: si True, lève une erreur quand la commande échoue.

    Returns:
        Le résultat de subprocess.run.
    """
    logger = get_run_logger()
    logger.info("Commande : %s", " ".join(cmd))
    result = subprocess.run(
        cmd, cwd=PROJECT_DIR, capture_output=True, text=True, check=False
    )
    if result.stdout:
        logger.info(result.stdout)
    if result.returncode != 0:
        logger.error(result.stderr or result.stdout)
        if check:
            raise RuntimeError(f"Échec de la commande : {' '.join(cmd)}")
    return result


# ---------- Tâches liées au code ----------


@task(name="installer-dependances")
def installer_dependances():
    """Installe les paquets listés dans requirements.txt."""
    run_cmd([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


@task(name="formatage-code")
def formatage_code():
    """Uniformise le style du code avec black."""
    run_cmd(["black", *PIPELINE_FILES])


@task(name="qualite-code")
def qualite_code():
    """Vérifie la qualité avec flake8, pylint et mypy."""
    run_cmd(["flake8", *PIPELINE_FILES])
    # pylint donne une note et peut sortir en erreur pour un simple avertissement
    run_cmd(["pylint", *PIPELINE_FILES], check=False)
    run_cmd(["mypy", *PIPELINE_FILES])


@task(name="securite-code")
def securite_code():
    """Recherche les failles courantes avec bandit."""
    # B404, B603, B607 : usage normal de subprocess par le pipeline
    run_cmd(["bandit", *PIPELINE_FILES, "-q", "-s", "B404,B603,B607"])


@task(name="tests-unitaires")
def tests_unitaires():
    """Exécute les tests unitaires avec pytest."""
    run_cmd([sys.executable, "-m", "pytest", TEST_FILE, "-q"])


# ---------- Tâches liées aux données et au modèle ----------


@task(name="preparation-donnees")
def preparation_donnees():
    """Charge et découpe les données (appelle prepare_data)."""
    return prepare_data(DATA_PATH)


@task(name="entrainement-modele")
def entrainement_modele(X_train, y_train, n_estimators=100):
    """Entraîne le modèle (appelle train_model)."""
    return train_model(X_train, y_train, n_estimators=n_estimators)


@task(name="sauvegarder-modele")
def sauvegarder_modele(model):
    """Sauvegarde le modèle sur le disque (appelle save_model)."""
    save_model(model, MODEL_PATH)
    return MODEL_PATH


@task(name="charger-modele")
def charger_modele():
    """Charge le modèle sauvegardé (appelle load_model)."""
    return load_model(MODEL_PATH)


@task(name="evaluer-modele")
def evaluer_modele(model, X_test, y_test):
    """Calcule les métriques sur le jeu de test (appelle evaluate_model)."""
    return evaluate_model(model, X_test, y_test)


@task(name="prediction")
def prediction(model, X_test):
    """Prédit la classe des cinq premiers clients du jeu de test."""
    predictions = model.predict(X_test[:5]).tolist()
    get_run_logger().info("Prédictions : %s", predictions)
    return predictions


# ---------- Flows ----------


@flow(name="install")
def install_flow():
    """Installe les dépendances du projet."""
    installer_dependances()


@flow(name="code")
def code_flow():
    """Contrôles sur le code : formatage, qualité, sécurité et tests."""
    formatage_code()
    qualite_code()
    securite_code()
    tests_unitaires()


@flow(name="train")
def train_flow(n_estimators: int = 100):
    """Prépare les données, entraîne le modèle et le sauvegarde."""
    X_train, _, y_train, _ = preparation_donnees()
    model = entrainement_modele(X_train, y_train, n_estimators)
    sauvegarder_modele(model)
    return model


@flow(name="evaluate")
def evaluate_flow():
    """Charge le modèle sauvegardé, l'évalue et fait des prédictions."""
    _, X_test, _, y_test = preparation_donnees()
    model = charger_modele()
    metrics = evaluer_modele(model, X_test, y_test)
    prediction(model, X_test)
    return metrics


@flow(name="all")
def all_flow(n_estimators: int = 100):
    """Pipeline complet : contrôles du code, entraînement, évaluation."""
    formatage_code()
    qualite_code()
    securite_code()
    tests_unitaires()

    X_train, X_test, y_train, y_test = preparation_donnees()
    model = entrainement_modele(X_train, y_train, n_estimators)
    sauvegarder_modele(model)
    metrics = evaluer_modele(model, X_test, y_test)
    prediction(model, X_test)
    return metrics


@task(name="nettoyer-projet")
def nettoyer_projet():
    """Supprime les modèles générés et les caches de tests."""
    for fichier in (PROJECT_DIR / "models").glob("*.joblib"):
        fichier.unlink()
    run_cmd(["rm", "-rf", ".pytest_cache"], check=False)


@flow(name="prepare")
def prepare_flow():
    """Charge et découpe les données uniquement."""
    preparation_donnees()


@flow(name="predict")
def predict_flow():
    """Charge le modèle sauvegardé et prédit sur le jeu de test."""
    _, X_test, _, _ = preparation_donnees()
    model = charger_modele()
    prediction(model, X_test)


@flow(name="test")
def test_flow():
    """Lance uniquement les tests unitaires."""
    tests_unitaires()


@flow(name="clean")
def clean_flow():
    """Supprime les fichiers générés par le pipeline."""
    nettoyer_projet()


FLOWS = {
    "install": install_flow,
    "code": code_flow,
    "prepare": prepare_flow,
    "train": train_flow,
    "entrainement": train_flow,
    "evaluate": evaluate_flow,
    "predict": predict_flow,
    "test": test_flow,
    "clean": clean_flow,
    "all": all_flow,
}


def main():
    """Lance le flow demandé en ligne de commande."""
    parser = argparse.ArgumentParser(description="Pipeline Prefect")
    parser.add_argument("--flow", choices=list(FLOWS), default="all")
    args = parser.parse_args()
    FLOWS[args.flow]()


if __name__ == "__main__":
    main()
