# Resumen del Sprint 1 — Datos, EDA y preparación

Líder: Joan · Leyenda: ✅ verificado con código ejecutado · 🔶 hipótesis · ⏳ pendiente.
Todas las cifras salen de `notebooks/01_eda_comprension_datos.ipynb` y `notebooks/03_modelado_supervisado.ipynb`, ejecutados en el `.venv` (Python 3.13.11). El test (jul–dic 2012) no se usó para decidir nada.

## 1. Definición de hecho (`plan_sprints.md` §4)
| Criterio | Estado | Dónde se comprueba |
| :--- | :---: | :--- |
| EDA documentado | ✅ | Notebook 01: 33 celdas de código, 59 `assert`, 10 bloques de interpretación, 0 errores |
| Contrato funciona con `assert` | ✅ | Importación y uso verificados en el `.venv` (ver §3) |
| Baselines con RMSLE de validación | ✅ | Notebook 03, secciones 3 a 5b |

## 2. Qué se hizo
- **Sprint 0 (Joan):** estructura §10, `requirements.txt` con 14 versiones fijadas (todas coinciden con `pip freeze` del `.venv`), `.gitignore`, `README.md` provisional, `reports/decisiones.md`.
- **`config/config.py`**, **`src/data_loader.py`**, **`src/evaluate.py`**, **`src/features.py`**: ver firmas en §3.
- **Notebook 01:** calidad de datos, target, perfil horario, tendencia, clima, y el flujo paso a paso del preprocesador.
- **Notebook 03 (baselines):** media por `(hr, workingday)` y `LinearRegression` con `TimeSeriesSplit`, solo train.

## 3. Qué pueden asumir Pardo y José

### Contrato (importar desde la raíz del repo)
```python
from config.config import CONFIG, RANDOM_STATE
from src.data_loader import load_data, temporal_split, separar_xy
from src.features import build_preprocessor
from src.evaluate import rmsle, evaluar_pipeline, get_cv, validar_temporal
```
| Función | Qué hace |
| :--- | :--- |
| `load_data(ruta=None) -> DataFrame` | 17 544 filas, índice horario `fecha_hora`. Las 165 horas ausentes tienen `cnt`, `casual`, `registered` y clima en NaN; el calendario está reconstruido. `weathersit` es float por los NaN |
| `temporal_split(df) -> (train, test)` | Train hasta 2012-06-30 23:00 (13 128 filas); test desde 2012-07-01 (4 416). Conservan las filas con target NaN |
| `separar_xy(df) -> (X, y)` | Quita las filas con `cnt` NaN y las columnas `cnt`, `casual`, `registered`, `instant`. `y` en bicicletas. X conserva `dteday` (solo se usa para derivar `trend`) |
| `build_preprocessor(escalar=False, tendencia=None, windspeed_cero=None, hum_cero=None, hora=None) -> Pipeline` | `Pipeline([crear_variables, columnas])`; los `None` toman `CONFIG["features"]`. Devuelve DataFrame (65 columnas con los valores por defecto). `escalar=True` solo para modelos lineales |
| `rmsle`, `rmse`, `mae`, `r2` `(y_true, y_pred)` | **Ambos en bicicletas.** RMSLE aplica `log1p` por dentro; la predicción se recorta a ≥ 0; ignoran `y_true` NaN |
| `evaluar_pipeline(pipeline, X_tr, y_tr, X_te, y_te, nombre, log_target=True, verbose=True) -> dict` | Ajusta (sobre `log1p(y)` si `log_target`), revierte con `expm1`, recorta a ≥ 0. Devuelve `modelo, rmsle, rmse, mae, r2, t_fit_s`. **No clona**: el pipeline queda ajustado |
| `get_cv(...) -> TimeSeriesSplit` | 5 folds, `test_size=1460`, `gap=0`, ventana expansiva (valores de `CONFIG["cv"]`) |
| `validar_temporal(pipeline, X, y, nombre, cv=None, log_target=True) -> (por_fold, resumen)` | Valida con `TimeSeriesSplit` clonando el pipeline en cada fold. `resumen` trae `rmsle_media`, `rmsle_std`, etc. **Es la función para las tablas «media y desviación por fold» del Sprint 2** |

Uso típico: `X_tr, y_tr = separar_xy(temporal_split(load_data())[0])` y luego `validar_temporal(Pipeline([("prep", build_preprocessor()), ("modelo", ...)]), X_tr, y_tr, "nombre")`.

### Valores PROVISIONALES de `CONFIG["features"]` (se confirman con ablación en el Sprint 2)
| Parámetro | Valor inicial | Opciones |
| :--- | :---: | :--- |
| `tendencia` | `"yr"` | `"yr"`, `"sin_yr"`, `"trend"` |
| `windspeed_cero` | `"dejar"` | `"dejar"`, `"indicador"`, `"mediana"` |
| `hum_cero` | `"mediana"` | `"dejar"`, `"mediana"` |
| `hora` | `"entero"` | `"entero"`, `"sin_cos"`, `"ambas"` |

Para comparar justamente, todos deben usar el mismo `X_tr`, `y_tr`, `get_cv()` y la métrica RMSLE de `src.evaluate`.

## 4. Resultados (validación temporal, solo train)
| Baseline | RMSLE medio ± std | RMSE medio | MAE medio | R² medio |
| :--- | :---: | :---: | :---: | :---: |
| Media por `(hr, workingday)` (plan: media de `cnt`) | **0.5585 ± 0.0601** | 102.8 | 72.3 | 0.619 |
| Media por `(hr, workingday)` (variante: media de `log1p`) | 0.5980 ± 0.1221 | 113.5 | 81.4 | 0.538 |
| `LinearRegression` cruda, objetivo `log1p` | 1.1055 ± 0.0951 | 166.1 | 115.5 | −0.062 |
| `LinearRegression` cruda, objetivo `cnt` | 1.2918 ± 0.1836 | 143.2 | 105.5 | 0.205 |

- ✅ El lineal crudo ajusta mal incluso dentro de train (RMSLE 1.0175): es un problema de forma, no de código.
- ✅ El baseline de calendario subestima en los folds 4 y 5 (demanda real media 232.3 y 273.0 frente a 142.4 y 153.9 predichas). 🔶 Es coherente con la tendencia de crecimiento; la ablación de tendencia lo medirá.

## 5. Hallazgos del EDA que afectan al modelado
- ✅ 165 horas ausentes (125 en train, 40 en test), concentradas de madrugada; el target queda NaN y esas filas no entran al ajuste ni a la métrica.
- ✅ `hum = 0`: 22 filas, todas el 2011-03-10. `weathersit = 4`: 3 filas, todas en train. `temp`–`atemp`: VIF 68.65 / 68.78, que baja a 1.40 / 1.52 con `temp_c` y `temp_diff`.
- ✅ `cnt` en train: asimetría 1.29; con `log1p` pasa a −0.81 (se reduce en valor absoluto, no queda simétrica).
- ✅ El perfil horario depende del tipo de día (picos a las 8:00 y 17:00 en laborales; campana a las 13:00 en fines de semana).
- ✅ Crecimiento ene–jun 2012 vs ene–jun 2011: +74 %, pero la razón mensual baja de 2.53 (marzo) a 1.41 (junio). 🔶 Qué pasará en jul–dic no puede afirmarse con estos datos.
- 🔶 `windspeed = 0` (12.5 %): el salto 0 → 0.0896 sugiere umbral del sensor; evidencia ambigua. Se decide por ablación.

## 6. Limitaciones conocidas
1. **Ningún fold valida en julio ni agosto** (ventanas del 2011-08-31 al 2012-06-30); el test sí los incluye. El RMSLE de validación puede no reflejar el verano.
2. Los folds 4 y 5 validan en 2012 con modelos entrenados mayormente con 2011.
3. Los pines de `requirements.txt` salen de Python 3.13; `lightgbm==4.7.0` y `shap==0.54.1` están verificados solo en el `.venv` de Joan.
4. No implementado (⏳): `temp_x_hum` (§4.2) y la imputación de `hum` por hora y mes o por interpolación temporal (§4.3). `hora` ofrece `entero`, `sin_cos` y `ambas`; splines y One-Hot de la hora (Pardo) hay que construirlos fuera de `build_preprocessor`.
5. `evaluar_pipeline` y `validar_temporal` no manejan pesos ni lags; si Joan añade lags en el Sprint 2 se hará en un paso previo y se documentará.
6. Los notebooks 01 y 03 pesan ~0.6 MB y ~0.13 MB por las figuras incrustadas; pueden generar conflictos de merge si dos personas los editan (cada notebook tiene un solo dueño).

## 7. Pendiente
- ⏳ Fecha de acceso al dataset (la completa Joan en el `README.md`).
- ⏳ Notas de investigación de Pardo y José en `reports/<nombre>/`.
- ⏳ Confirmar con Pardo y José que `pip install -r requirements.txt` y los `import` corren en sus máquinas (condición del Sprint 0).
- ⏳ **Decisiones del grupo:** versión de Python (Joan decidió 3.13 por ser la verificada, sujeto a la herramienta AutoML), umbral de éxito (propuesta: partir de 0.5585) y fechas de los sprints.
- 🔶 Ningún commit se ha hecho: el trabajo está en el árbol de trabajo (rename en staging).
