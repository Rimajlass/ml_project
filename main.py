"""Point d'entrée du pipeline : exécute les étapes via la ligne de commande."""

import argparse

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)


def parse_args(argv=None):
    """Lit les arguments de la ligne de commande.

    Args:
        argv: liste d'arguments (utile pour les tests). Si None, lit sys.argv.

    Returns:
        Les arguments parsés.
    """
    parser = argparse.ArgumentParser(description="Pipeline de prédiction du churn")
    parser.add_argument("--data", required=True, help="Chemin du fichier CSV")
    parser.add_argument("--model-path", default="models/model.joblib")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument(
        "--step",
        choices=["train", "evaluate", "all"],
        default="all",
        help="train : entraîne et sauvegarde ; evaluate : charge et évalue ; all : tout",
    )
    return parser.parse_args(argv)


def run(args):
    """Exécute les étapes demandées.

    Args:
        args: arguments issus de parse_args.
    """
    X_train, X_test, y_train, y_test = prepare_data(args.data)

    if args.step in ("train", "all"):
        model = train_model(X_train, y_train, n_estimators=args.n_estimators)
        save_model(model, args.model_path)
        print(f"Modèle sauvegardé dans {args.model_path}")

    if args.step in ("evaluate", "all"):
        model = load_model(args.model_path)
        metrics = evaluate_model(model, X_test, y_test)
        for name, value in metrics.items():
            print(f"{name} : {value:.3f}")


def main():
    """Fonction principale."""
    run(parse_args())


if __name__ == "__main__":
    main()
