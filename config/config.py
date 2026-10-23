"""Configuración central del proyecto Bike Sharing Demand.

Todo el equipo importa de aquí: ``from config.config import CONFIG``.
Los valores marcados como PROVISIONAL son el punto de partida para que el código corra;
se confirman o se cambian solo con el RMSLE de validación temporal (project_plan_crisp_dm.md §4.2, §4.3).
"""
from pathlib import Path

RANDOM_STATE = 42

RAIZ = Path(__file__).resolve().parents[1]

CONFIG = {
    "random_state": RANDOM_STATE,

    # --- Rutas ---
    "ruta_hour": RAIZ / "data" / "raw" / "hour.csv",
    "dir_processed": RAIZ / "data" / "processed",
    "dir_models": RAIZ / "models",
    "dir_reports": RAIZ / "reports",

    # --- Serie temporal (verificado contra hour.csv: 17 379 filas de 17 544 horas posibles) ---
    "fecha_inicio": "2011-01-01 00:00",
    "fecha_fin": "2012-12-31 23:00",
    "n_horas": 17544,
    "n_horas_faltantes": 165,

    # --- Partición temporal: el test (jul-dic 2012) no se toca hasta el Sprint 4 ---
    "ultima_hora_train": "2012-06-30 23:00",
    "primera_hora_test": "2012-07-01 00:00",

    # --- Target y columnas ---
    "target": "cnt",
    # Columnas que nunca entran como variables de entrada (leakage o sin valor predictivo)
    "columnas_excluidas": ["casual", "registered", "instant", "dteday", "cnt"],

    # --- Validación temporal (TimeSeriesSplit, ventana expansiva) ---
    # test_size=1460 filas (~61 días) y 5 folds => el primer fold entrena con ~8 meses (§4.1: >= 6-8 meses).
    # PROVISIONAL: ventana expansiva vs. deslizante y número de folds por comparar (§12.2 punto 5).
    "cv": {"n_splits": 5, "test_size": 1460, "gap": 0, "max_train_size": None},

    # --- Ablaciones de features (§4.2 punto 6, §4.3). Todos PROVISIONAL hasta correr la ablación ---
    "features": {
        # "yr" (binaria 0/1) | "sin_yr" | "trend" (días desde 2011-01-01 / 730)
        "tendencia": "yr",
        # "dejar" | "indicador" (deja el valor y añade is_windspeed_zero) | "mediana" (0 -> NaN -> mediana de train)
        "windspeed_cero": "dejar",
        # "dejar" | "mediana" (0 -> NaN -> mediana de train)
        "hum_cero": "mediana",
        # "entero" (hr tal cual, para árboles) | "sin_cos" | "ambas"
        "hora": "entero",
    },
}
