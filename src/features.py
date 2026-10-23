"""Ingeniería de características y preprocesador (project_plan_crisp_dm.md §4.2 y §4.3).

``build_preprocessor`` devuelve un ``Pipeline`` de dos pasos:
  1. ``crear_variables``: ``FunctionTransformer`` sin estado (no aprende nada de los datos).
  2. ``columnas``: ``ColumnTransformer`` con One-Hot, imputación y escalado. Todo lo que se ajusta
     (medianas, categorías, escalas) se aprende solo en ``fit`` con los datos que reciba (train).

Las columnas ``casual``, ``registered``, ``instant``, ``dteday`` (crudo) y ``cnt`` nunca llegan al modelo:
el ``ColumnTransformer`` selecciona columnas por nombre y descarta el resto (``remainder='drop'``).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from config.config import CONFIG

TENDENCIAS = ("yr", "sin_yr", "trend")
WINDSPEED_CERO = ("dejar", "indicador", "mediana")
HUM_CERO = ("dejar", "mediana")
HORA = ("entero", "sin_cos", "ambas")

COLS_CATEGORICAS = ["season", "weathersit", "hr_workingday"]
FECHA_ORIGEN = pd.Timestamp("2011-01-01")


def crear_variables(X: pd.DataFrame, tendencia: str, windspeed_cero: str, hum_cero: str, hora: str) -> pd.DataFrame:
    """Transformaciones sin estado sobre las columnas crudas de hour.csv.

    - ``weathersit`` = 4 se fusiona con 3 (solo 3 filas en todo el dataset).
    - ``temp`` y ``atemp`` se desnormalizan (``*41``, ``*50``); se conserva ``temp_c`` y ``temp_diff = atemp_c - temp_c``.
    - ``hr_workingday``: categoría combinada (p. ej. ``"18_1"``), no el producto numérico.
    - ``hum == 0`` y, según la opción, ``windspeed == 0`` se marcan como NaN para imputarlos después con la mediana de train.
    """
    d = pd.DataFrame(index=X.index)
    d["season"] = X["season"]
    d["mnth"] = X["mnth"]
    d["holiday"] = X["holiday"]
    d["weekday"] = X["weekday"]
    d["workingday"] = X["workingday"]
    # entero (en load_data es float por los NaN de las horas ausentes); -1 = clima desconocido -> One-Hot en ceros
    d["weathersit"] = X["weathersit"].clip(upper=3).fillna(-1).astype(int)

    temp_c = X["temp"] * 41
    d["temp_c"] = temp_c
    d["temp_diff"] = X["atemp"] * 50 - temp_c

    d["hum"] = X["hum"].mask(X["hum"].eq(0)) if hum_cero == "mediana" else X["hum"]

    if windspeed_cero == "mediana":
        d["windspeed"] = X["windspeed"].mask(X["windspeed"].eq(0))
    else:
        d["windspeed"] = X["windspeed"]
    if windspeed_cero == "indicador":
        d["is_windspeed_zero"] = X["windspeed"].eq(0).astype(int)

    if hora in ("entero", "ambas"):
        d["hr"] = X["hr"]
    if hora in ("sin_cos", "ambas"):
        d["hr_sin"] = np.sin(2 * np.pi * X["hr"] / 24)
        d["hr_cos"] = np.cos(2 * np.pi * X["hr"] / 24)

    d["hr_workingday"] = X["hr"].astype(int).astype(str) + "_" + X["workingday"].astype(int).astype(str)

    if tendencia == "yr":
        d["yr"] = X["yr"]
    elif tendencia == "trend":
        dias = (pd.to_datetime(X["dteday"]) - FECHA_ORIGEN).dt.days + X["hr"] / 24
        d["trend"] = dias / 730
    return d


def columnas_numericas(tendencia: str, windspeed_cero: str, hora: str) -> list:
    cols = ["mnth", "holiday", "weekday", "workingday", "temp_c", "temp_diff"]
    if hora in ("entero", "ambas"):
        cols.append("hr")
    if hora in ("sin_cos", "ambas"):
        cols += ["hr_sin", "hr_cos"]
    if windspeed_cero == "indicador":
        cols.append("is_windspeed_zero")
    if tendencia == "yr":
        cols.append("yr")
    elif tendencia == "trend":
        cols.append("trend")
    return cols


def build_preprocessor(escalar: bool = False, tendencia: str = None, windspeed_cero: str = None,
                       hum_cero: str = None, hora: str = None) -> Pipeline:
    """Construye el preprocesador. Los parámetros en ``None`` toman el valor de ``CONFIG['features']``.

    escalar: ``StandardScaler`` sobre las numéricas; solo para modelos lineales (los árboles no lo necesitan).
    tendencia: ``'yr'`` | ``'sin_yr'`` | ``'trend'`` (ablación §4.2 punto 6).
    windspeed_cero: ``'dejar'`` | ``'indicador'`` | ``'mediana'`` (ablación §4.3).
    hum_cero: ``'dejar'`` | ``'mediana'`` (ablación §4.3).
    hora: ``'entero'`` | ``'sin_cos'`` | ``'ambas'`` (la comparación de codificaciones para lineales es del Sprint 2).
    """
    base = CONFIG["features"]
    tendencia = base["tendencia"] if tendencia is None else tendencia
    windspeed_cero = base["windspeed_cero"] if windspeed_cero is None else windspeed_cero
    hum_cero = base["hum_cero"] if hum_cero is None else hum_cero
    hora = base["hora"] if hora is None else hora
    assert tendencia in TENDENCIAS, f"tendencia debe ser una de {TENDENCIAS}"
    assert windspeed_cero in WINDSPEED_CERO, f"windspeed_cero debe ser una de {WINDSPEED_CERO}"
    assert hum_cero in HUM_CERO, f"hum_cero debe ser una de {HUM_CERO}"
    assert hora in HORA, f"hora debe ser una de {HORA}"

    def bloque(imputar):
        pasos = []
        if imputar:
            pasos.append(("imputer", SimpleImputer(strategy="median")))
        if escalar:
            pasos.append(("scaler", StandardScaler()))
        return Pipeline(pasos) if pasos else "passthrough"

    cols_imputar = (["hum"] if hum_cero == "mediana" else []) + (["windspeed"] if windspeed_cero == "mediana" else [])
    cols_sin_imputar = columnas_numericas(tendencia, windspeed_cero, hora) \
        + [c for c in ("hum", "windspeed") if c not in cols_imputar]

    transformadores = [
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), COLS_CATEGORICAS),
        ("num", bloque(False), cols_sin_imputar),
    ]
    if cols_imputar:
        transformadores.append(("num_imp", bloque(True), cols_imputar))

    columnas = ColumnTransformer(transformadores, remainder="drop", verbose_feature_names_out=False)
    crear = FunctionTransformer(
        crear_variables, validate=False,
        kw_args={"tendencia": tendencia, "windspeed_cero": windspeed_cero, "hum_cero": hum_cero, "hora": hora},
    )
    prep = Pipeline([("crear_variables", crear), ("columnas", columnas)])
    prep.set_output(transform="pandas")
    return prep
