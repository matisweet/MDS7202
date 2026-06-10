import os
import pickle
from importlib.metadata import version

import mlflow
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def get_best_model(experiment_id):
    runs = mlflow.search_runs(experiment_id)
    best_model_id = runs.sort_values("metrics.valid_f1", ascending=False)["run_id"].iloc[0]
    best_model = mlflow.sklearn.load_model("runs:/" + best_model_id + "/model")

    return best_model


def optimize_model():
    df = pd.read_csv("water_potability.csv")

    X = df.drop(columns=["Potability"])
    y = df["Potability"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=20906297,
    )

    experiment_name = "Lab8 XGBoost Optuna"
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 250),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }

        model = XGBClassifier(
            **params,
            random_state=20906297,
            eval_metric="logloss",
        )

        run_name = f"XGBoost lr {params['learning_rate']:.3f}"

        with mlflow.start_run(
            experiment_id=experiment.experiment_id,
            run_name=run_name,
        ):
            model.fit(X_train, y_train)

            y_pred = model.predict(X_valid)
            valid_f1 = f1_score(y_valid, y_pred)

            mlflow.log_params(params)
            mlflow.log_metric("valid_f1", valid_f1)
            mlflow.sklearn.log_model(model, "model")

        return valid_f1

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=20906297),
    )

    study.optimize(objective, n_trials=10)

    best_model = get_best_model(experiment.experiment_id)

    os.makedirs("models", exist_ok=True)

    with open("models/best_model.pkl", "wb") as file:
        pickle.dump(best_model, file)

    with open("models/library_versions.txt", "w") as file:
        file.write(f"mlflow=={version('mlflow')}\n")
        file.write(f"optuna=={version('optuna')}\n")
        file.write(f"xgboost=={version('xgboost')}\n")
        file.write(f"pandas=={version('pandas')}\n")
        file.write(f"scikit-learn=={version('scikit-learn')}\n")

    return best_model


if __name__ == "__main__":
    optimize_model()
