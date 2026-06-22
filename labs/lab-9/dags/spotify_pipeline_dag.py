import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_DIR = Path("/mnt/c/MDS7202/labs/lab-9/data")
OUTPUT_PATH = Path("/tmp/spotify_data.parquet")

PARAM_COLS = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "tempo",
    "duration_ms",
    "year",
]


# ── Funciones auxiliares (dadas) ─────────────────────────────────────────────


def load_batch(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_all_parallel(data_dir: Path, n_batches: int = 5) -> pd.DataFrame:
    paths = sorted(data_dir.glob("*.parquet"))[:n_batches]
    with ThreadPoolExecutor(max_workers=None) as executor:
        dfs = list(executor.map(load_batch, [str(p) for p in paths]))
    return pd.concat(dfs, ignore_index=True)


def build_pipeline(n_jobs: int = -1) -> Pipeline:
    return Pipeline(
        [
            (
                "column_transformer",
                ColumnTransformer(
                    [
                        ("ohe", OneHotEncoder(handle_unknown="ignore"), ["key", "mode", "genre"]),
                        ("numerical", "passthrough", PARAM_COLS),
                    ]
                ),
            ),
            ("random_forest", RandomForestRegressor(n_jobs=n_jobs, random_state=42)),
        ]
    )


# ── Funciones de las tareas de Airflow ───────────────────────────────────────


def task_load_data_fn(**context):
    """Carga 5 batches en paralelo, los guarda en disco y pasa la ruta por XCom."""
    df = load_all_parallel(DATA_DIR, n_batches=5)
    df.to_parquet(OUTPUT_PATH)
    context["ti"].xcom_push(key="data_path", value=str(OUTPUT_PATH))
    print(f"Datos cargados: {df.shape[0]} filas guardadas en {OUTPUT_PATH}")


def task_train_model_fn(**context):
    """Recupera la ruta desde XCom, carga los datos y entrena el pipeline."""
    data_path = context["ti"].xcom_pull(key="data_path", task_ids="load_data")
    df = pd.read_parquet(data_path)

    X = df[PARAM_COLS + ["key", "mode", "genre"]]
    y = df["valence"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = build_pipeline(n_jobs=-1)
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    print(f"Modelo entrenado en {time.perf_counter() - t0:.1f}s sobre {X_train.shape[0]} filas")


# ── Definición del DAG ────────────────────────────────────────────────────────

with DAG(
    dag_id="spotify_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mds7202", "spotify"],
) as dag:
    load_data = PythonOperator(
        task_id="load_data",
        python_callable=task_load_data_fn,
    )

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=task_train_model_fn,
    )

    load_data >> train_model
