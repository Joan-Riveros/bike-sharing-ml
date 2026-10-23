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

    # --- LightGBM (Sprint 2). Parámetros fijos para comparar ablaciones; el tuning los sobrescribe ---
    "lgbm": {
        "params_base": {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20,
                        "random_state": RANDOM_STATE, "deterministic": True, "force_row_wise": True,
                        "verbose": -1},
        # Resultado del tuning del Sprint 2 (notebook 03, sección 9); RMSLE medio de validación 0.4549 (gap=0)
        "params_tuneados": {"n_estimators": 524, "learning_rate": 0.0462172510502637, "num_leaves": 31,
                            "min_child_samples": 81, "colsample_bytree": 0.8766560517837736,
                            "reg_lambda": 0.28567374298471887},
        "n_iter_busqueda": 40,
        "semillas_repeticion": [RANDOM_STATE, 7, 123],
        "lags": ("lag_24", "lag_168"),
        "gap_con_lags": 24,
    },

    # --- Modelo final (Sprint 4): LightGBM manual (params_base + params_tuneados), objetivo log1p y tres lags ---
    # Elegido en el Sprint 3 (reports/decisiones.md, fichas 20 y 21). Horizonte: día siguiente, pronóstico emitido a las 00:00.
    "modelo_final": {
        "lags": ("lag_24", "lag_168", "lag_336"),
        "horas_lags": (24, 168, 336),
        "ruta": RAIZ / "models" / "lgbm_final.joblib",
        "horizonte": "día siguiente (24 h), pronóstico emitido a las 00:00",
    },

    # --- Regla para adoptar una variante (plan_sprints.md §2.1 punto 4: mejora consistente entre folds) ---
    # PROPUESTA de Joan, ajustable por el grupo: mejora media de RMSLE >= min_mejora y peor en <= max_folds_peor folds.
    "regla_adopcion": {"min_mejora_rmsle": 0.002, "max_folds_peor": 1},

    # --- Ablaciones de features (§4.2 punto 6, §4.3) ---
    # Sprint 2: con LightGBM ninguna variante cumplió la regla de adopción, así que se mantienen estos valores
    # (reports/decisiones.md, fichas 1-3). Siguen PROVISIONALES para los modelos lineales y de bagging (Pardo, José).
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
