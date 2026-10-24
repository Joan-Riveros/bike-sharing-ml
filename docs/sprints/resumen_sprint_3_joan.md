# Resumen del Sprint 3 — Comparación, errores e interpretabilidad (Joan)

Lo reportado sale de código ejecutado; lo no probado se indica como hipótesis y lo no hecho como pendiente. Todo sale de `notebooks/05_comparacion_y_errores.ipynb` (25 celdas de código, 35 `assert`, 0 errores), ejecutado en el `.venv` (Python 3.13.11). Solo train y `get_cv()`; **el test no se ha cargado y no se reentrenó ningún modelo con todo train**. Fichas 15 a 19 en `reports/decisiones.md`.

## Qué se hizo
- **Paso 0:** `requirements.txt` con `umap-learn==0.5.12` y `flaml[automl]==2.7.0` (instalan y se importan en el `.venv`; **sin el extra `[automl]` `from flaml import AutoML` falla**, y trae `xgboost`). Ficha 14 corregida con la salida ejecutada del notebook 04; fecha del README sin corchetes.
- **Tabla comparativa reconstruida con código** de los cinco modelos, en dos vías (sin lags `gap=0`, con lags `gap=24`). Reproduce lo registrado por cada responsable; AutoML reproduce a 6 decimales el RMSLE de `automl_config.json` (el modelo es idéntico al de José, sin repetir la búsqueda de FLAML).
- **Errores fuera de muestra** (7 300 horas de validación): residuos vs predicho, por mes, hora, día, feriado, clima adverso y cluster de Pardo (regenerado: tamaños 3 495 / 3 133 / 3 373 / 3 002).
- **Interpretabilidad:** permutación por grupos y SHAP en los folds 3, 4 y 5.

## Resultados (RMSLE medio ± std entre folds)
| Modelo | Sin lags (`gap=0`) | Con lags (`gap=24`) |
| :--- | :---: | :---: |
| Baseline `(hr, workingday)` | 0.5585 ± 0.0601 | 0.5586 ± 0.0601 |
| Ridge (splines × temp) | 0.4885 ± 0.1583 | 0.4766 ± 0.1335 |
| Random Forest | 0.4826 ± 0.1748 | 0.4125 ± 0.0598 |
| LightGBM manual | 0.4549 ± 0.1918 | **0.3905 ± 0.0853** |
| AutoML (FLAML → lgbm) | **0.4405 ± 0.1946** | **0.3876 ± 0.0697** |

*Nota: las cifras de Random Forest (0.4826 y 0.4125) se calcularon en el equipo de Joan; el notebook 03c de José da 0.4827 y 0.4124 (diferencia de 0.0001 entre equipos, sin efecto en las conclusiones).*

Fold a fold: sin lags, AutoML es el mejor en 3 folds, LightGBM en 1 y el baseline en 1 (fold 3); con lags, LightGBM en 3, AutoML en 1 y Random Forest en 1 (fold 3). Cara a cara: sin lags AutoML gana a LightGBM en 4 de 5 folds (bootstrap por días: diferencia 0.0125, IC95 % [0.0077, 0.0175]); con lags LightGBM gana en 3 de 5 pero la diferencia agrupada favorece a AutoML en 0.0053 (IC95 % [−0.0005, 0.0105]).

## Hallazgos de errores
- **Subestimación sistemática:** sesgo medio del residuo +0.110 (LightGBM + lags), +0.103 (AutoML + lags), +0.188 (sin lags). Concentrado en **ene–mar de 2012** (+0.405, +0.299, +0.161 con lags; +0.766 y +0.587 sin lags), el período del fold 3 con train solo de 2011. Los lags corrigen cerca de la mitad.
- **No hay homocedasticidad (validación):** dispersión del residuo 0.496 (poca demanda) frente a 0.189 (mucha demanda); el RMSLE global está dominado por las horas de baja demanda.
- **Puntos débiles:** clima adverso (RMSLE 0.602, 685 horas), feriados (0.582, 237 horas), madrugada (2:00 = 0.533), fines de semana (0.442) y domingos (0.443). El cluster no laborable (2) es el peor (0.464) y el de tarde cálida (0) el mejor (0.282); los lags mejoran los cuatro.
- **El modelo se apoya en la hora y en los lags:** `lag_168` 22–28 % y `lag_24` ≈ 8 % del |SHAP|, hora 23–25 %, `hr × workingday` 12–14 %. Sin lags el RMSLE de un fold sube entre 0.45 y 0.73. Hipótesis: `lag_168` (patrón semanal) pesa más que `lag_24` en LightGBM; AutoML usa más `lag_24`.
- Atención: **Fold 3:** sin lags, el baseline gana el fold 3 (0.4949 frente a 0.7489–0.7797): los modelos con `yr` entrenados solo con 2011 fallan más que una media por grupo.

## Propuesta de modelo final (decide Joan)
Descartados: baseline, Ridge y Random Forest (con lags quedan 0.022–0.025 por detrás; solo ganan el fold 3). Entre LightGBM manual y AutoML con lags hay un **empate práctico**: la diferencia (0.0029) es ≈ 7 veces menor que la desviación fold a fold de la diferencia (0.0198).

| Criterio (con lags) | LightGBM manual | AutoML (FLAML) | Ventaja |
| :--- | :---: | :---: | :---: |
| RMSLE medio | 0.3905 | 0.3876 | AutoML |
| Desviación entre folds / peor fold | 0.0853 / 0.5044 | 0.0697 / 0.4736 | AutoML |
| Folds ganados cara a cara | 3 | 2 | LightGBM |
| RMSE / MAE medio (bicicletas) | 63.2 / 40.6 | 64.4 / 41.0 | LightGBM |
| Árboles × hojas | 524 × 31 = 16 244 | 185 × 55 = 10 175 | AutoML |
| Reproducibilidad de la búsqueda | semilla fija | depende del tiempo de cómputo | LightGBM |
| Reproducibilidad del modelo | parámetros en `CONFIG` | `automl_config.json` (se reproduce exacto) | empate |
| Sin lags (referencia) | 0.4549 | 0.4405 (mejor en 4/5) | AutoML |

**Mi propuesta:** LightGBM con `lag_24` + `lag_168` y **la configuración encontrada por AutoML**, con el LightGBM manual como alternativa documentada e igual de válida (si se prefiere una historia más sencilla de defender: tuning explícito con rangos y semillas, sin depender de FLAML). Es una propuesta, no una decisión; con estos datos no se puede afirmar que uno sea mejor.

## Anexo: horizonte de pronóstico (día o semana) — `notebooks/06_horizonte_y_lags.ipynb`, ficha 20
RMSLE medio de validación (`gap=24`, solo train; LightGBM manual | AutoML con config sin lags | AutoML con config con lags):

| Escenario | LightGBM | AutoML (sin lags) | AutoML (con lags) |
| :--- | :---: | :---: | :---: |
| Sin lags | 0.4562 | 0.4417 | 0.4491 |
| Semana: `lag_168` + `lag_336` | 0.3893 | 0.3863 | 0.3918 |
| Día, emitido a las 00:00: `lag_24` + `lag_168` | 0.3905 | 0.3855 | 0.3876 |
| Día, emitido a las 18:00 | 0.4015 | 0.3975 | 0.4014 |
| Entrenado con `lag_24` y usado sin él | 0.4131 | 0.4173 | **0.4460** |

- Los lags **no obligan a un horizonte diario**: el esquema semanal (`lag_168` + `lag_336`) y el diario empatan (diferencias de 0.001–0.004 frente a una desviación entre folds de ≈ 0.08).
- Emitir el pronóstico a las 18:00 en lugar de a las 00:00 cuesta ≈ 0.011–0.014; sigue mejor que no tener lags.
- La configuración de AutoML con lags es la más frágil si `lag_24` falta de forma inesperada (0.4460, casi igual que sin lags).
- Pendiente: **No medido:** el efecto de un pronóstico de clima a una semana (esta medición usa el clima observado); se hará en la validación operativa (§8.4).
- Hipótesis: `lag_336` es nuevo y se probó junto con otros 5 escenarios sobre los mismos folds: el mejor escenario está algo sobreestimado.

### Actualización: los tres lags (ficha 21) y selección propuesta
| Escenario | LightGBM manual | AutoML (sin lags) | AutoML (con lags) |
| :--- | :---: | :---: | :---: |
| Día, 00:00: `lag_24` + `lag_168` + `lag_336` | **0.3732 ± 0.0566** | 0.3738 ± 0.0673 | 0.3766 ± 0.0604 |
| Día, 18:00: los tres lags | 0.3823 | 0.3826 | 0.3838 |
| Falla de `lag_24` (entrenado con los tres) | **0.3862** | 0.3985 | 0.4077 |

- Los tres lags son la mejor combinación: LightGBM manual 0.3732 (18.2 % menos que sin lags; fold 3 de 0.7786 a 0.4117), con la menor desviación entre folds (0.0566) y la mayor robustez si falta `lag_24`.
- **Empate de LightGBM con AutoML:** −0.0020, IC95 % [−0.0081, 0.0039]. El día supera a la semana (`lag_168` + `lag_336`, 0.3893) por 0.0188, IC95 % [−0.0243, −0.0133] (Hipótesis: optimista; en 3 de 5 folds, con el fold 3 aportando casi todo).
- Atención: **No cumple la regla de adopción** (peor en 2 folds: f1 +0.008 y f5 +0.017); la media mejora 0.0831. Si se exige la regla: `lag_24` + `lag_168` (0.3905).
- **Propuesta (decide Joan):** horizonte **día siguiente (emitido a las 00:00)** y **LightGBM manual con `lag_24`, `lag_168` y `lag_336`**; alternativa semanal: `lag_168` + `lag_336`. Esto reemplaza la propuesta anterior (configuración de AutoML con lags).

## Decisiones que necesita Joan
1. **Modelo final:** LightGBM manual o configuración de AutoML (propuesta: AutoML).
2. **Horizonte y lags** (ver anexo): día o semana; condiciona todo: son ≈ 29–36 % del |SHAP|, solo valen con el horizonte «día siguiente» y exigen el día anterior y la semana anterior completos. Si no se usan, el finalista es **AutoML sin lags** (0.4405).
3. **Umbral de éxito (decisión del grupo, aún abierta):** referencia 0.5585 del baseline; el candidato con lags mejora 30–31 %, sin lags 18–21 %. *(Cálculos sobre las medias de validación.)*

## Qué asumir para el Sprint 4 (cuando Joan lo autorice)
- Pendiente: **Evaluación en test una sola vez** con el modelo elegido, reentrenado con todo train. Antes: confirmar el modelo y los lags.
- Pendiente: **RMSLE mensual en test** (jul–dic): es lo primero a mirar. Ningún fold validó julio ni agosto; si el crecimiento de 2012 continúa, se espera subestimación como la de ene–mar (hipótesis).
- Pendiente: **Validación operativa (§8.4):** ruido en el clima, backtest día a día, métrica operativa, `src/predict.py`, casos límite y deriva. Incluir un caso con **lags ausentes** (Sandy el 29-oct-2012 solo tiene 1 hora registrada) y puntos débiles: clima adverso, feriados, madrugada.
- Pendiente: reevaluar el factor de *smearing* (ficha 6) solo si importa MAE o RMSE en bicicletas.
- `models/` está en `.gitignore`: ni `lgbm_joan.joblib` ni el clustering de Pardo viajan con git (se regeneran con los notebooks 03 y 02).

## Limitaciones
1. Mes y fold están confundidos en los análisis por mes; ningún mes de validación es julio ni agosto (agosto solo tiene 22 horas).
2. Los puntajes de los finalistas son optimistas (hiperparámetros elegidos sobre los mismos folds). Los IC del bootstrap solo miden la variación entre días de las mismas 5 ventanas.
3. Clusters: hipótesis con silueta 0.18; ajustados con todo train, así que las ventanas de validación participaron en su ajuste (sin usar el target).
4. La permutación de `yr` da 0 porque es constante dentro de cada ventana (limitación del método).
5. Cortes con pocas horas (feriados 237, clima adverso 685) tienen más incertidumbre; no se calcularon intervalos.
