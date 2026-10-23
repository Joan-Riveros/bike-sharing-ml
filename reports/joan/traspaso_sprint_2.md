# Nota de traspaso al Sprint 2 (Joan sigue como líder)

Detalle completo del contrato y las limitaciones: `reports/joan/resumen_sprint_1.md`.

## Antes de empezar (todos)
1. Clonar, crear el entorno (`python -m venv .venv`, `pip install -r requirements.txt`) y comprobar los imports del README. Avisar a Joan si algo falla.
2. Cada quien trabaja en su rama (`joan/...`, `pardo/...`, `jose/...`) y en sus propios archivos (`plan_sprints.md` §2.3).
3. Referencia común: **RMSLE medio de validación del baseline de calendario = 0.5585 ± 0.0601** (notebook 03). El umbral de éxito es decisión del grupo (⏳).
4. Mismo protocolo para todos: `X_tr, y_tr = separar_xy(temporal_split(load_data())[0])`, `get_cv()` y `validar_temporal(...)`. Sin tocar el test.

## Qué necesita cada persona
| Persona | Tarea | Necesita del contrato | Ojo |
| :--- | :--- | :--- | :--- |
| **Joan** | LightGBM, `RandomizedSearchCV` + `TimeSeriesSplit`, ablación de tendencia, `log1p`+MSE vs Poisson, lags | `build_preprocessor(escalar=False)`, `validar_temporal` | Probar `tendencia` (`yr`/`sin_yr`/`trend`), `windspeed_cero` y `hum_cero` con RMSLE de validación y registrar ficha en `reports/decisiones.md`. Ningún fold valida en jul–ago |
| **Pardo** | Ridge/Lasso (splines vs One-Hot vs sin/cos); K-Means, GMM, PCA, UMAP | `build_preprocessor(escalar=True, hora=...)`; para clustering, `separar_xy` y su propio escalado ajustado solo con train | `hora` solo ofrece `entero`, `sin_cos`, `ambas`; splines y One-Hot de la hora van fuera de `build_preprocessor`. La categoría `hr_workingday` (One-Hot) ya está incluida |
| **José** | Random Forest, AutoML con el protocolo §7.2 | `build_preprocessor(escalar=False)`, `get_cv()`, `rmsle` | Confirmar en la documentación que la herramienta AutoML respeta el orden temporal. PyCaret 3.3 exige Python ≤ 3.11: si se elige, hay que revisar la versión de Python del grupo |

## Decisiones abiertas para la reunión de inicio del Sprint 2
- Herramienta AutoML y, según ella, versión de Python.
- Fechas de los sprints y umbral de éxito.
- Si se modifica el contrato (p. ej. añadir una opción de codificación de la hora a `features.py`), se avisa a Joan: `features.py` y `evaluate.py` son suyos.
- Si algún valor provisional de `CONFIG["features"]` cambia por ablación, se anota la ficha y se actualiza `config/config.py` (solo Joan).
