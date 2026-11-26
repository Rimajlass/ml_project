"""Tests unitaires des fonctions de model_pipeline.py."""

import numpy as np
import pandas as pd
import pytest

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)


@pytest.fixture
def csv_file(tmp_path):
    """Crée un petit jeu de données synthétique au format du churn bancaire."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "RowNumber": range(1, n + 1),
            "CustomerId": rng.integers(10**7, 2 * 10**7, n),
            "Surname": ["Dupont"] * n,
            "CreditScore": rng.integers(300, 850, n),
            "Geography": rng.choice(["France", "Spain", "Germany"], n),
            "Gender": rng.choice(["Male", "Female"], n),
            "Age": rng.integers(18, 90, n),
            "Balance": rng.uniform(0, 1e5, n),
            "Exited": rng.integers(0, 2, n),
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def split_data(csv_file):
    """Renvoie les données découpées par prepare_data."""
    return prepare_data(csv_file)


def test_prepare_data_shapes(split_data):
    """Le découpage respecte la proportion de test."""
    X_train, X_test, y_train, y_test = split_data
    assert len(X_test) == 40
    assert len(X_train) == 160
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_prepare_data_columns(split_data):
    """La cible est retirée et les colonnes catégorielles sont encodées."""
    X_train, _, _, _ = split_data
    assert "Exited" not in X_train.columns
    assert "Surname" not in X_train.columns
    assert "Geography_Spain" in X_train.columns


def test_train_model_returns_fitted_model(split_data):
    """train_model renvoie un modèle capable de prédire."""
    X_train, X_test, y_train, _ = split_data
    model = train_model(X_train, y_train, n_estimators=10)
    assert len(model.predict(X_test)) == len(X_test)


def test_evaluate_model_returns_metrics(split_data):
    """evaluate_model renvoie les quatre métriques attendues entre 0 et 1."""
    X_train, X_test, y_train, y_test = split_data
    model = train_model(X_train, y_train, n_estimators=10)
    metrics = evaluate_model(model, X_test, y_test)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1"}
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_save_and_load_model(tmp_path, split_data):
    """Un modèle rechargé produit les mêmes prédictions que l'original."""
    X_train, X_test, y_train, _ = split_data
    model = train_model(X_train, y_train, n_estimators=10)
    path = tmp_path / "model.joblib"
    save_model(model, path)
    loaded = load_model(path)
    assert np.array_equal(model.predict(X_test), loaded.predict(X_test))
