"""Entraîne un modèle simple et enregistre l'expérience avec MLflow."""
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main():
    """Lance l'entraînement et journalise les paramètres et métriques."""
    X, y = load_iris(return_X_y=True, as_frame=True)
    assert X.isnull().sum().sum() == 0, "Valeurs manquantes détectées"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    n_estimators = 100
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model", skops_trusted_types=["sklearn.tree._tree.Tree"])
        print(f"Accuracy : {accuracy:.3f}")


if __name__ == "__main__":
    main()
