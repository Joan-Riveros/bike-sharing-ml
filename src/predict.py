"""Inferencia de punta a punta del modelo final (project_plan_crisp_dm.md §8.4 punto 4).

Uso: dado un día, el pronóstico meteorológico de sus 24 horas y el historial de demanda anterior al día,
devuelve las 24 predicciones de bicicletas. El modelo es el LightGBM con ``lag_24``, ``lag_168`` y ``lag_336``
(horizonte «día siguiente», pronóstico emitido a las 00:00).

    from src.predict import cargar_modelo, predecir_dia
    modelo = cargar_modelo()
    pred = predecir_dia("2012-07-10", pronostico, historial_cnt, feriado=False, modelo=modelo)

``pronostico``: DataFrame de 24 filas con las columnas ``hr`` (0-23), ``weathersit`` (1-4), ``temp_c`` (°C),
``atemp_c`` (°C), ``hum_pct`` (%) y ``windspeed_kmh`` (km/h).
``historial_cnt``: Series horaria (índice ``DatetimeIndex``) con la demanda observada **anterior** al día;
si falta una hora, el lag correspondiente queda NaN (el modelo lo admite, con menor precisión).
"""
import warnings

import joblib
import numpy as np
import pandas as pd

from config.config import CONFIG
from src.evaluate import predecir_bicis

COLUMNAS_PRONOSTICO = ["hr", "weathersit", "temp_c", "atemp_c", "hum_pct", "windspeed_kmh"]
# Rangos físicos plausibles (el dataset llega a temp 41 °C, viento 57 km/h); fuera de ellos se rechaza la entrada
RANGOS = {"temp_c": (-20.0, 50.0), "atemp_c": (-30.0, 60.0), "hum_pct": (0.0, 100.0), "windspeed_kmh": (0.0, 120.0)}
# Convenciones del dataset (README de UCI): temp/41, atemp/50, hum/100, windspeed/67
ESCALAS = {"temp": 41.0, "atemp": 50.0, "hum": 100.0, "windspeed": 67.0}


def cargar_modelo(ruta=None) -> dict:
    """Carga el paquete guardado por el notebook 07 (``models/lgbm_final.joblib``)."""
    ruta = CONFIG["modelo_final"]["ruta"] if ruta is None else ruta
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}. Se genera ejecutando notebooks/07_evaluacion_final_test.ipynb.")
    paquete = joblib.load(ruta)
    assert {"pipeline", "log_target", "lags"} <= set(paquete), "el archivo no es un paquete del modelo final"
    return paquete


def estacion(fecha: pd.Timestamp) -> int:
    """Estación del dataset (1 = del 21-dic al 20-mar, 2 = 21-mar, 3 = 21-jun, 4 = 23-sep); verificado contra hour.csv."""
    md = (fecha.month, fecha.day)
    if md >= (12, 21) or md < (3, 21): return 1
    if md < (6, 21): return 2
    if md < (9, 23): return 3
    return 4


def validar_pronostico(pronostico: pd.DataFrame) -> pd.DataFrame:
    """Comprueba el esquema y los rangos; devuelve el pronóstico ordenado por hora. Lanza ``ValueError`` si algo falla."""
    if not isinstance(pronostico, pd.DataFrame):
        raise ValueError("el pronóstico debe ser un DataFrame")
    faltan = [c for c in COLUMNAS_PRONOSTICO if c not in pronostico.columns]
    if faltan:
        raise ValueError(f"faltan columnas en el pronóstico: {faltan}")
    p = pronostico[COLUMNAS_PRONOSTICO].copy().sort_values("hr").reset_index(drop=True)
    if len(p) != 24 or p["hr"].tolist() != list(range(24)):
        raise ValueError("el pronóstico debe tener exactamente una fila por cada hora 0-23")
    if p.isna().any().any():
        raise ValueError("el pronóstico contiene valores vacíos")
    if not p["weathersit"].isin([1, 2, 3, 4]).all():
        raise ValueError("weathersit debe ser 1, 2, 3 o 4")
    for col, (lo, hi) in RANGOS.items():
        if not p[col].between(lo, hi).all():
            raise ValueError(f"{col} fuera del rango físico [{lo}, {hi}]")
    return p


def construir_entrada(fecha, pronostico: pd.DataFrame, historial_cnt: pd.Series, feriado: bool = False) -> pd.DataFrame:
    """Reproduce las columnas de ``separar_xy`` + lags para las 24 horas del día."""
    fecha = pd.Timestamp(fecha).normalize()
    p = validar_pronostico(pronostico)
    if not isinstance(historial_cnt.index, pd.DatetimeIndex):
        raise ValueError("el historial debe tener un DatetimeIndex horario")
    if len(historial_cnt) and historial_cnt.index.max() >= fecha:
        raise ValueError("el historial contiene horas del día a pronosticar o posteriores (no se permiten datos del futuro)")
    anio = fecha.year - 2011
    if anio not in (0, 1):
        warnings.warn("el modelo se entrenó con 2011-2012; se usa yr = 1 como proxy del nivel más reciente", stacklevel=2)
    weekday = (fecha.dayofweek + 1) % 7                 # 0 = domingo, como en hour.csv
    laborable = int(weekday in range(1, 6) and not feriado)
    horas = fecha + pd.to_timedelta(p["hr"], unit="h")
    X = pd.DataFrame({
        "dteday": fecha, "season": estacion(fecha), "yr": int(min(max(anio, 0), 1)), "mnth": fecha.month, "hr": p["hr"].to_numpy(),
        "holiday": int(feriado), "weekday": weekday, "workingday": laborable,
        "weathersit": p["weathersit"].astype(int).to_numpy(),
        "temp": p["temp_c"].to_numpy() / ESCALAS["temp"], "atemp": p["atemp_c"].to_numpy() / ESCALAS["atemp"],
        "hum": p["hum_pct"].to_numpy() / ESCALAS["hum"], "windspeed": p["windspeed_kmh"].to_numpy() / ESCALAS["windspeed"],
    }, index=pd.DatetimeIndex(horas, name="fecha_hora"))
    for h in CONFIG["modelo_final"]["horas_lags"]:
        X[f"lag_{h}"] = historial_cnt.reindex(X.index - pd.Timedelta(hours=h)).to_numpy(dtype=float)
    return X


def predecir_dia(fecha, pronostico: pd.DataFrame, historial_cnt: pd.Series, feriado: bool = False, modelo: dict = None) -> pd.DataFrame:
    """Predicción de bicicletas para las 24 horas de ``fecha``. Devuelve un DataFrame con ``prediccion`` y ``lags_faltantes``."""
    modelo = cargar_modelo() if modelo is None else modelo
    X = construir_entrada(fecha, pronostico, historial_cnt, feriado)
    pred = predecir_bicis(modelo["pipeline"], X, modelo["log_target"])
    lags = list(modelo["lags"])
    faltantes = X[lags].isna().sum(axis=1).to_numpy()
    if (faltantes == len(lags)).any():
        warnings.warn(f"{int((faltantes == len(lags)).sum())} horas sin ningún lag disponible: el modelo, entrenado con lags, predice "
                      "una demanda muy inferior a la real en ese caso (ver notebooks/08, sección 6). Use un modelo sin lags o complete el historial.",
                      UserWarning, stacklevel=2)
    return pd.DataFrame({"prediccion": pred, "lags_faltantes": faltantes}, index=X.index)
