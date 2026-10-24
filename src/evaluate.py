import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from config.config import CONFIG

METRICAS = ["rmsle", "rmse", "mae", "r2"]


def _preparar(y_true, y_pred):
    """Convierte a arrays float, descarta pares con y_true NaN y recorta la predicción a >= 0."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    assert y_true.shape == y_pred.shape, f"formas distintas: {y_true.shape} vs {y_pred.shape}"
    valido = ~np.isnan(y_true)
    assert not np.isnan(y_pred[valido]).any(), "la predicción contiene NaN donde hay target"
    return y_true[valido], np.clip(y_pred[valido], 0, None)


def rmsle(y_true, y_pred) -> float:
    y_true, y_pred = _preparar(y_true, y_pred)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _preparar(y_true, y_pred)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _preparar(y_true, y_pred)
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true, y_pred) -> float:
    y_true, y_pred = _preparar(y_true, y_pred)
    return float(r2_score(y_true, y_pred))


def calcular_metricas(y_true, y_pred) -> dict:
    return {"rmsle": rmsle(y_true, y_pred), "rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred), "r2": r2(y_true, y_pred)}


def ajustar(pipeline, X, y, log_target: bool = True):
    """Ajusta el pipeline sobre ``log1p(y)`` (si ``log_target``) o sobre ``y``."""
    y = np.asarray(y, dtype=float)
    pipeline.fit(X, np.log1p(y) if log_target else y)
    return pipeline


def predecir_bicis(pipeline, X, log_target: bool = True) -> np.ndarray:
    """Predicción en bicicletas: revierte con ``expm1`` si el modelo se entrenó en log, y recorta a >= 0."""
    pred = np.asarray(pipeline.predict(X), dtype=float)
    if log_target:
        pred = np.expm1(pred)
    return np.clip(pred, 0, None)


def evaluar_pipeline(pipeline, X_tr, y_tr, X_te, y_te, nombre: str,
                     log_target: bool = True, verbose: bool = True) -> dict:
    """Ajusta con (X_tr, y_tr), predice (X_te) y devuelve un diccionario de métricas (estilo lab4).

    ``y_tr`` e ``y_te`` en bicicletas. No clona el pipeline: queda ajustado al terminar.
    Esta función es genérica: el notebook decide qué conjunto pasa como ``X_te``
    (validación hasta el Sprint 4; el test solo para el modelo final).
    """
    t0 = time.perf_counter()
    ajustar(pipeline, X_tr, y_tr, log_target)
    t_fit = time.perf_counter() - t0
    pred = predecir_bicis(pipeline, X_te, log_target)
    met = {"modelo": nombre, **calcular_metricas(y_te, pred), "t_fit_s": t_fit}
    if verbose:
        print(f"{nombre}: RMSLE={met['rmsle']:.4f} | RMSE={met['rmse']:.2f} | "
              f"MAE={met['mae']:.2f} | R2={met['r2']:.4f} | fit={t_fit:.1f}s")
    return met


def get_cv(n_splits=None, test_size=None, gap=None, max_train_size=None) -> TimeSeriesSplit:
    base = CONFIG["cv"]
    return TimeSeriesSplit(
        n_splits=base["n_splits"] if n_splits is None else n_splits,
        test_size=base["test_size"] if test_size is None else test_size,
        gap=base["gap"] if gap is None else gap,
        max_train_size=base["max_train_size"] if max_train_size is None else max_train_size,
    )


def validar_temporal(pipeline, X, y, nombre: str = "modelo", cv=None, log_target: bool = True):
    cv = get_cv() if cv is None else cv
    if isinstance(X.index, pd.DatetimeIndex):
        assert X.index.is_monotonic_increasing, "X debe estar en orden cronológico"
    y = np.asarray(y, dtype=float)
    filas = []
    for k, (i_tr, i_va) in enumerate(cv.split(X), start=1):
        assert i_tr.max() < i_va.min(), "el train de un fold debe ser anterior a su validación"
        modelo = clone(pipeline)
        ajustar(modelo, X.iloc[i_tr], y[i_tr], log_target)
        pred = predecir_bicis(modelo, X.iloc[i_va], log_target)
        fila = {"fold": k, "n_train": len(i_tr), "n_val": len(i_va)}
        if isinstance(X.index, pd.DatetimeIndex):
            fila.update(train_fin=X.index[i_tr[-1]], val_inicio=X.index[i_va[0]], val_fin=X.index[i_va[-1]])
        fila.update(calcular_metricas(y[i_va], pred))
        filas.append(fila)
    por_fold = pd.DataFrame(filas)
    resumen = {"modelo": nombre}
    for m in METRICAS:
        resumen[f"{m}_media"] = float(por_fold[m].mean())
        resumen[f"{m}_std"] = float(por_fold[m].std(ddof=1))
    return por_fold, resumen
