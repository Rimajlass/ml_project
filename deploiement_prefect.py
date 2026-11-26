"""Déploie les 8 flows Prefect. Seul le pipeline complet est planifié."""
from prefect import serve

from pipeline_prefect import (
    all_flow,
    clean_flow,
    evaluate_flow,
    install_flow,
    predict_flow,
    prepare_flow,
    test_flow,
    train_flow,
)

if __name__ == "__main__":
    serve(
        all_flow.to_deployment(
            name="ml-pipeline-all",
            cron="0 2 * * *",
            tags=["mlops", "full-pipeline"],
        ),
        install_flow.to_deployment(name="ml-pipeline-install", tags=["mlops", "setup"]),
        prepare_flow.to_deployment(name="ml-pipeline-prepare", tags=["mlops", "data"]),
        train_flow.to_deployment(name="ml-pipeline-train", tags=["mlops", "training"]),
        evaluate_flow.to_deployment(
            name="ml-pipeline-evaluate", tags=["mlops", "evaluation"]
        ),
        predict_flow.to_deployment(
            name="ml-pipeline-predict", tags=["mlops", "inference"]
        ),
        test_flow.to_deployment(name="ml-pipeline-test", tags=["mlops", "qa"]),
        clean_flow.to_deployment(name="ml-pipeline-clean", tags=["mlops", "maintenance"]),
    )
