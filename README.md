# Bike Sharing Demand — Proyecto Integrador, Parte I

Predicción de la demanda horaria de bicicletas compartidas (Capital Bikeshare, Washington D.C.) con regresión supervisada sobre datos tabulares con estructura temporal. Metodología CRISP-DM. Equipo: Joan Marcelo Riveros Soria, Alejandro Miguel Pardo Romano y Jose Andres Manzaneda Manzaneda.

- Plan metodológico: [`project_plan_crisp_dm.md`](docs/project_plan_crisp_dm.md) · Reparto de sprints: [`plan_sprints.md`](docs/plan_sprints.md) · Registro de decisiones y pruebas: [`reports/decisiones.md`](reports/decisiones.md).

## Resultado en una página

**Modelo final:** LightGBM (hiperparámetros de la ficha 4), entrenado sobre `log1p(demanda)`, con `lag_24`, `lag_168` y `lag_336` (la demanda de hace 1, 7 y 14 días a la misma hora). **Horizonte:** día siguiente, pronóstico emitido a las 00:00 (24 horas).

| Evaluación (RMSLE, menor es mejor) | Valor |
| :--- | :---: |
| Validación temporal, solo train (5 folds de `TimeSeriesSplit`) | 0.3732 ± 0.0566 |
| **Test (jul–dic 2012), una sola evaluación** | **0.3505** (RMSE 69.99 · MAE 43.11 · R² 0.8985) |
| Baseline (media por hora y tipo de día) en test | 0.5999 (el modelo final lo mejora 41.6 %) |

Lo que hay que saber antes de usarlo: tiende a **subestimar** la demanda (−6.9 % en el total diario), falla más en **clima adverso, feriados y madrugada**, y **depende de los lags**: sin ningún lag disponible predice un 85 % menos. Los lags no mejoraron a LightGBM en el test de verano (0.3463 sin lags frente a 0.3505), aunque sí ayudaron en la validación y en nov–dic. Detalle en las fichas 22 y 23.

## Datos

- Fuente: UCI *Bike Sharing Dataset* (`data/raw/hour.csv`, `data/raw/day.csv`). Descripción oficial en `data/raw/Readme.txt`.
- Cita obligatoria: Fanaee-T, H. y Gama, J. (2013), *Event labeling combining ensemble detectors and background knowledge*, Progress in Artificial Intelligence, doi:10.1007/s13748-013-0040-3.
- **Fecha de acceso al dataset: 17/10/2026.**
- 17 379 filas de 17 544 horas posibles (2011-01-01 a 2012-12-31); `load_data()` reindexa a la serie continua y deja el target en NaN en las 165 horas ausentes.
- Partición temporal: train hasta 2012-06-30 (13 128 horas; 13 003 con registro), test desde 2012-07-01 (4 416 horas; 4 376 con registro).

## Entorno

Python **3.13** (verificado con 3.13.11) y las versiones fijadas de `requirements.txt`. El AutoML es **FLAML** (PyCaret 3.3 exige Python ≤ 3.11).

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -c "import numpy, pandas, sklearn, matplotlib, seaborn, lightgbm, shap, umap, flaml, joblib; print('ok')"
```

Los notebooks y scripts se ejecutan **desde la raíz del repositorio o desde `notebooks/`** (detectan la raíz buscando `config/config.py`). En Linux o macOS la activación es `source .venv/bin/activate`.

## Cómo reproducir el proyecto

| Orden | Notebook | Qué hace | Tiempo aprox. |
| :---: | :--- | :--- | :---: |
| 1 | `01_eda_comprension_datos.ipynb` | EDA, calidad de datos, preprocesador paso a paso; genera `data/processed/` | 1 min |
| 2 | `02_no_supervisado.ipynb` | K-Means, GMM, PCA, UMAP (k = 4); genera `models/clustering_pardo.joblib` | 1–2 min |
| 3 | `03_modelado_supervisado.ipynb` | Baselines y LightGBM: ablaciones, tuning, Poisson, *smearing*, lags | ≈ 6 min (la búsqueda) |
| 4 | `03b_ridge.ipynb`, `03c_random_forest.ipynb` | Ridge/Lasso y Random Forest | 1–3 min |
| 5 | `04_automl_benchmark.ipynb` | AutoML con FLAML (2 × 300 s de presupuesto) | ≈ 11 min |
| 6 | `05_comparacion_y_errores.ipynb` | Tabla comparativa, residuos, errores por cluster, SHAP | 1–2 min |
| 7 | `06_horizonte_y_lags.ipynb` | Día vs semana: qué lags existen según el horizonte | 1–2 min |
| 8 | **`07_evaluacion_final_test.ipynb`** | Reentrena el modelo final, **evalúa en el test una vez**, genera `models/lgbm_final.joblib` | 1 min |
| 9 | `08_validacion_operativa.ipynb` | Ruido en el clima, backtest diario, métrica operativa, casos límite, deriva | 1–2 min |
| 10 | `09_demostracion_prediccion.ipynb` | Demostración ilustrativa de `predict.py`: dos días reales, escenario «mañana» sintético, caso sin historial y una semana día a día (carga el modelo del notebook 07; no lo regenera) | < 1 min |

**Informe técnico:** el texto fuente es `reports/joan/informe_tecnico.md`; el PDF y el Word (`informe_tecnico.pdf` y `informe_tecnico.docx`, en la misma carpeta) se regeneran con `python scripts/exportar_informe.py` (solo Windows con Edge y Word instalados; no instala nada).

**Camino mínimo** para obtener el modelo final y su validación: ejecutar `07` y luego `08` (necesitan `data/raw/` y `reports/jose/automl_config.json`, que viaja con el repositorio). `models/` y `data/processed/` están en `.gitignore` y se regeneran con los notebooks.

## Uso del modelo final

```python
from src.predict import cargar_modelo, predecir_dia

modelo = cargar_modelo()                                 # models/lgbm_final.joblib (se genera con el notebook 07)
pred = predecir_dia("2012-09-12", pronostico, historial_cnt, feriado=False, modelo=modelo)
```

- `pronostico`: DataFrame de 24 filas con `hr` (0–23), `weathersit` (1–4), `temp_c` (°C), `atemp_c` (°C), `hum_pct` (%) y `windspeed_kmh` (km/h).
- `historial_cnt`: Series horaria con la demanda observada **anterior** al día (se rechaza si incluye horas del día o posteriores).
- Devuelve un DataFrame con `prediccion` (bicicletas) y `lags_faltantes`. Si alguna hora no tiene ningún lag, emite un aviso: el modelo no es fiable en ese caso.

## Estructura

```text
bike-sharing-ml/
├── config/            # CONFIG central (RANDOM_STATE = 42, rutas, validación, modelo final)
├── data/raw/          # hour.csv, day.csv originales (no se modifican)
├── data/processed/    # train/test generados (ignorado por git)
├── notebooks/         # 01 a 09 (ver tabla de reproducción)
├── src/
│   ├── data_loader.py # load_data, temporal_split, separar_xy
│   ├── features.py    # build_preprocessor, agregar_lags
│   ├── evaluate.py    # rmsle, rmse, mae, r2, evaluar_pipeline, get_cv, validar_temporal
│   ├── lineales.py    # preprocesador de los modelos lineales (Pardo)
│   ├── clustering.py  # K-Means sobre las horas (Pardo)
│   └── predict.py     # inferencia de punta a punta del modelo final
├── models/            # artefactos .joblib (ignorado por git)
├── scripts/
│   └── exportar_informe.py  # genera el PDF y el Word del informe
├── docs/
│   ├── project_plan_crisp_dm.md  # plan metodológico (CRISP-DM)
│   ├── plan_sprints.md           # reparto de sprints y reglas del equipo
│   ├── preguntas_defensa.md      # preparación de la defensa oral
│   └── sprints/                  # resúmenes de cada sprint y notas de investigación
├── reports/
│   ├── decisiones.md      # fichas de decisión (todo lo probado fuera de los labs)
│   ├── joan/              # informe técnico (.md, .pdf, .docx), figuras y predicciones del test
│   ├── pardo/             # figuras del modelo lineal y del clustering
│   └── jose/              # resultados de Random Forest y AutoML, y figuras
├── requirements.txt
└── README.md
```

## Contrato de importación

```python
from config.config import CONFIG
from src.data_loader import load_data, temporal_split, separar_xy
from src.features import build_preprocessor, agregar_lags
from src.evaluate import rmsle, evaluar_pipeline, get_cv, validar_temporal
```

## Reglas del proyecto (control de fuga de datos)

- `RANDOM_STATE = 42`; `Pipeline` + `ColumnTransformer`; **sin K-Fold aleatorio** (siempre `TimeSeriesSplit`, ventana expansiva).
- Todo ajuste (escalado, categorías, imputación, clusters, modelos) usa solo train y ocurre dentro del pipeline.
- `casual`, `registered`, `instant` y `dteday` crudo nunca son entradas; los lags solo miran el pasado.
- El test (jul–dic 2012) no se usó para decidir nada: el modelo final se fijó con la validación (fichas 20 y 21) y se evaluó una sola vez (ficha 22).
- Todo lo probado fuera de los labs se registra en `reports/decisiones.md` con su resultado, aunque no mejorara.

## Limitaciones

1. Demanda total de la ciudad por hora; no sirve para decisiones por estación.
2. El modelo con lags necesita el historial reciente completo; sin él, subestima de forma severa.
3. Los puntajes de validación y del tuning son algo optimistas (hiperparámetros elegidos sobre los mismos folds); ningún fold de validación cubría julio ni agosto.
4. Las pruebas de ruido en el clima usan niveles de referencia no calibrados con pronósticos reales.
5. El crecimiento de la demanda en 2012 (+74 % en ene–jun frente a 2011) obliga a reentrenar con datos recientes; el modelo se entrenó solo con 2011 y la primera mitad de 2012.

## Siguientes pasos (Parte II)

Modelos secuenciales (LSTM), lags dinámicos, *fallback* sin lags, interpolación de horas ausentes del pronóstico y monitoreo de la deriva de la demanda (PSI de la demanda y de sus lags). El plan detallado (deep learning, embeddings, MLOps, recuperación, demo integrada, monitoreo y política de actualización) está en la sección 13 de `reports/joan/informe_tecnico.md`.
