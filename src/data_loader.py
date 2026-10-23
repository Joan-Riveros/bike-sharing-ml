"""Carga de hour.csv, reindexación a la serie horaria continua y partición temporal."""
import numpy as np
import pandas as pd

from config.config import CONFIG

# Columnas de calendario constantes dentro de cada fecha (verificado: nunique == 1 por dteday)
_COLS_CALENDARIO_DIA = ["season", "yr", "mnth", "holiday", "weekday", "workingday"]
_COLS_ENTERAS = _COLS_CALENDARIO_DIA + ["hr"]


def load_data(ruta=None) -> pd.DataFrame:
    """Lee hour.csv y reindexa a 17 544 horas continuas (2011-01-01 00:00 a 2012-12-31 23:00).

    - Índice: ``DatetimeIndex`` horario llamado ``fecha_hora``.
    - Las 165 horas ausentes quedan con ``cnt``, ``casual``, ``registered`` y las variables
      meteorológicas en NaN (no se inventa el target).
    - El calendario de las horas ausentes se reconstruye: ``dteday``/``hr`` desde el índice y
      ``season``, ``yr``, ``mnth``, ``holiday``, ``weekday``, ``workingday`` desde las demás filas
      de la misma fecha (todas las fechas tienen al menos una fila).
    """
    ruta = CONFIG["ruta_hour"] if ruta is None else ruta
    raw = pd.read_csv(ruta, parse_dates=["dteday"])

    raw["fecha_hora"] = raw["dteday"] + pd.to_timedelta(raw["hr"], unit="h")
    assert raw["fecha_hora"].is_unique, "hour.csv tiene horas duplicadas"
    raw = raw.set_index("fecha_hora").sort_index()

    indice = pd.date_range(CONFIG["fecha_inicio"], CONFIG["fecha_fin"], freq="h", name="fecha_hora")
    df = raw.reindex(indice)

    # Calendario de las horas ausentes
    df["dteday"] = df.index.normalize()
    df["hr"] = df.index.hour
    por_fecha = raw.groupby("dteday")[_COLS_CALENDARIO_DIA].first()
    faltan = df["season"].isna()
    df.loc[faltan, _COLS_CALENDARIO_DIA] = por_fecha.reindex(df.loc[faltan, "dteday"]).to_numpy()
    df[_COLS_ENTERAS] = df[_COLS_ENTERAS].astype(int)

    # `instant` solo existe en las filas originales; con huecos queda como entero nulable
    df["instant"] = df["instant"].astype("Int64")
    # Conteos como float para admitir NaN en las horas ausentes
    df[["casual", "registered", "cnt"]] = df[["casual", "registered", "cnt"]].astype(float)

    assert len(df) == CONFIG["n_horas"], f"se esperaban {CONFIG['n_horas']} horas, hay {len(df)}"
    assert int(df[CONFIG["target"]].isna().sum()) == CONFIG["n_horas_faltantes"]
    return df


def temporal_split(df: pd.DataFrame):
    """Partición cronológica: train hasta 2012-06-30 23:00; test desde 2012-07-01 00:00.

    Devuelve ``(train, test)``. Ambos conservan las horas con target NaN para mantener la serie
    continua; use ``separar_xy`` para obtener X, y sin esas filas.
    """
    train = df.loc[: CONFIG["ultima_hora_train"]]
    test = df.loc[CONFIG["primera_hora_test"]:]
    assert len(train) + len(test) == len(df)
    assert train.index.max() < test.index.min(), "train y test se solapan"
    return train, test


def separar_xy(df: pd.DataFrame):
    """Devuelve ``(X, y)`` sin las filas con target NaN (horas ausentes).

    X no incluye ``cnt``, ``casual``, ``registered`` ni ``instant`` (leakage). Conserva ``dteday``
    porque ``src.features`` deriva de él la tendencia; el preprocesador nunca lo usa crudo.
    ``y`` está en bicicletas. Para otros targets (p. ej. casual/registered) use ``df`` directamente.
    """
    objetivo = CONFIG["target"]
    con_target = df[objetivo].notna()
    X = df.loc[con_target].drop(columns=["cnt", "casual", "registered", "instant"])
    y = df.loc[con_target, objetivo].astype(float)
    return X, y
