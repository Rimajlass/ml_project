"""Fonctions modularisées du pipeline de prédiction du churn bancaire."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

TARGET = "Exited"
DROP_COLUMNS = ["RowNumber", "CustomerId", "Surname"]
CATEGORICAL_COLUMNS = ["Geography", "Gender"]


def prepare_data(csv_path, test_size=0.2, random_state=42):
    """Charge le CSV, nettoie les colonnes et découpe les données en train/test.

    Args:
        csv_path: chemin vers le fichier CSV.
        test_size: proportion des données réservée au test.
        random_state: graine pour un découpage reproductible.

    Returns:
        Tuple (X_train, X_test, y_train, y_test).

    Raises:
        ValueError: si le jeu de données contient des valeurs manquantes.
    """
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    if df.isnull().sum().sum() > 0:
        raise ValueError("Valeurs manquantes détectées dans les données")

    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True, dtype=int)
    features = df.drop(columns=[TARGET])
    target = df[TARGET]

    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """Entraîne un modèle de forêt aléatoire.

    Args:
        X_train: caractéristiques d'entraînement.
        y_train: cibles d'entraînement.
        n_estimators: nombre d'arbres de la forêt.
        random_state: graine pour la reproductibilité.

    Returns:
        Le modèle entraîné.
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Calcule les métriques de performance du modèle sur le jeu de test.

    Args:
        model: modèle entraîné.
        X_test: caractéristiques de test.
        y_test: cibles de test.

    Returns:
        Dictionnaire des métriques (accuracy, precision, recall, f1).
    """
    predictions = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
    }


def save_model(model, path):
    """Sauvegarde le modèle avec joblib, en créant le dossier si besoin.

    Args:
        model: modèle à sauvegarder.
        path: chemin du fichier de sortie.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    """Charge un modèle sauvegardé avec joblib.

    Args:
        path: chemin du fichier du modèle.

    Returns:
        Le modèle chargé.
    """
    return joblib.load(path)
