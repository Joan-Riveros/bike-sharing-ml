"""Clustering de horas (Pardo, Sprint 2; project_plan_crisp_dm.md §5).

Entradas (§5 punto 1): ``hr_sin``, ``hr_cos``, ``temp``, ``hum``, ``windspeed``, ``workingday``. Sin el target.
``build_clustering`` devuelve un ``Pipeline`` (variables -> imputación -> escalado -> K-Means) que se ajusta
**solo con train** y luego asigna cualquier fila con ``predict`` (así se usa en el Sprint 3 para el RMSLE por cluster).
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from config.config import RANDOM_STATE

VARIABLES = ["hr_sin", "hr_cos", "temp", "hum", "windspeed", "workingday"]
K_ELEGIDO = 4


def variables_clustering(X: pd.DataFrame) -> pd.DataFrame:
    """Sin estado. ``hum == 0`` (22 filas del 2011-03-10) pasa a NaN y se imputa con la mediana de train."""
    return pd.DataFrame({
        "hr_sin": np.sin(2 * np.pi * X["hr"] / 24),
        "hr_cos": np.cos(2 * np.pi * X["hr"] / 24),
        "temp": X["temp"],
        "hum": X["hum"].mask(X["hum"].eq(0)),
        "windspeed": X["windspeed"],
        "workingday": X["workingday"],
    }, index=X.index)


def build_escalado() -> Pipeline:
    """Variables -> mediana -> StandardScaler. Ajustar solo con train."""
    pipe = Pipeline([
        ("variables", FunctionTransformer(variables_clustering, validate=False)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    pipe.set_output(transform="pandas")
    return pipe


def build_clustering(k: int = K_ELEGIDO, random_state: int = RANDOM_STATE) -> Pipeline:
    """Pipeline completo con K-Means (``n_init=10``). ``predict`` devuelve la etiqueta de cluster."""
    return Pipeline([*build_escalado().steps, ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=random_state))])
