# Bike Sharing Demand — Proyecto Integrador, Parte I

Predicción de la demanda horaria de bicicletas compartidas (Capital Bikeshare, Washington D.C.) con regresión supervisada sobre datos tabulares con estructura temporal. Equipo: Joan, Pardo, José.

> Este README es provisional (Sprint 0). Joan lo mantiene hasta el Sprint 3 y Pardo lo completa en el Sprint 4 (cierre). El plan metodológico está en `project_plan_crisp_dm.md` y el reparto de trabajo en `plan_sprints.md`.

## Datos

- Fuente: UCI *Bike Sharing Dataset* (`data/raw/hour.csv`, `data/raw/day.csv`). Descripción oficial en `data/raw/Readme.txt`.
- Cita obligatoria: Fanaee-T, H. y Gama, J. (2013), *Event labeling combining ensemble detectors and background knowledge*, Progress in Artificial Intelligence, doi:10.1007/s13748-013-0040-3.
- **Fecha de acceso al dataset: [completar — dato que debe aportar Joan]**.

## Entorno

Entorno de referencia: **Python 3.13** (versiones fijadas en `requirements.txt`; verificado con 3.13.11). El AutoML está pendiente de decisión del grupo: PyCaret 3.3 exige Python ≤ 3.11, así que si se elige PyCaret habrá que revisar esta versión.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -c "import numpy, pandas, sklearn, matplotlib, seaborn, lightgbm, shap, joblib; print('ok')"
```

Los notebooks y scripts se ejecutan desde la raíz del repositorio (para que funcionen `from src... import` y `from config.config import CONFIG`).

## Estructura

```text
bike-sharing-ml/
├── config/            # CONFIG central y RANDOM_STATE
├── data/raw/          # hour.csv, day.csv originales (no se modifican)
├── data/processed/    # train/test generados (ignorado por git)
├── notebooks/         # 01 EDA, 02 no supervisado, 03 modelado, 04 AutoML
├── src/               # data_loader.py, features.py, evaluate.py (predict.py en Sprint 4)
├── models/            # artefactos .joblib (ignorado por git)
├── reports/           # joan/, pardo/, jose/ y decisiones.md
├── requirements.txt
└── README.md
```

## Reglas del equipo (resumen)

- `RANDOM_STATE = 42`; `Pipeline` + `ColumnTransformer`; sin K-Fold aleatorio.
- El test (jul–dic 2012) no se toca hasta el Sprint 4 y se evalúa una sola vez.
- Todo lo que se pruebe fuera de los labs se registra en `reports/decisiones.md`.
- Contrato de importación:

```python
from config.config import CONFIG
from src.data_loader import load_data, temporal_split
from src.features import build_preprocessor
from src.evaluate import rmsle, evaluar_pipeline, get_cv
```
