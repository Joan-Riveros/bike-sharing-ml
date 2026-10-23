# Resumen del Sprint 2 — parte de José (Random Forest y AutoML)

✅ verificado con código ejecutado · 🔶 hipótesis · ⏳ pendiente. Todo sale de `notebooks/03c_random_forest.ipynb` y `notebooks/04_automl_benchmark.ipynb`, ejecutados con los pines de `requirements.txt` (+ `flaml[automl]`) en **Python 3.13.14 / Windows 10**. Validación temporal solo con train; **el test no se usó**. Fichas 12 a 14 en `reports/decisiones.md`.

## Resultados (RMSLE medio ± std, 5 folds)
| Modelo | RMSLE | Validación |
| :--- | :---: | :---: |
| Baseline de calendario (Joan) | 0.5585 ± 0.0601 | `gap=0` |
| RF base (300 árboles) | 0.4887 ± 0.1703 | `gap=0` |
| RF tuneado (`max_features=0.5`, `min_samples_leaf=1`, `max_depth=25`) | 0.4827 ± 0.1748 | `gap=0` |
| LightGBM + tuning (Joan) | 0.4549 ± 0.1918 | `gap=0` |
| **AutoML FLAML (lgbm)** | **0.4405 ± 0.1946** | `gap=0` |
| RF tuneado + `lag_24` + `lag_168` | 0.4124 ± 0.0599 | `gap=24` |
| LightGBM + lags (Joan) | 0.3905 ± 0.0853 | `gap=24` |
| **AutoML FLAML + lags (lgbm)** | **0.3876 ± 0.0697** | `gap=24` |

- ✅ RF tuneado es 13.6 % mejor que el baseline y equivale a Ridge (0.4885); queda 0.028 peor que LightGBM.
- ✅ Lags en RF: la media mejora 0.070 (de 0.4828 a 0.4124) pero cumple mal la regla del grupo (peor en 3 de 5 folds); la ganancia está en los folds 3 y 4. 🔶 Actúan como corrector de tendencia.
- ✅ Importancia por permutación (fold 5): `hr` ≫ `workingday` > `weathersit`. `temp` y `atemp` son colineales y se reparten la importancia.
- ✅ AutoML sin lags supera al mejor manual en 4 de 5 folds (0.4405 vs 0.4549). Con lags empatan (0.3876 vs 0.3905; AutoML mejor en 2 de 5).
- 🔶 Sin lags AutoML eligió árboles pequeños (`num_leaves=4`), fuera del rango del tuning manual (31–127); con lags eligió `num_leaves=55`.

## Para el Sprint 3 (José lidera)
- Modelo RF guardado en `models/rf_jose.joblib` (sin lags; carpeta ignorada por git, se regenera con el notebook 03c en ~6 min).
- Decisión 8a del grupo: propongo tabla principal **sin lags para todos** y los lags como vía extra aclarada.
- AutoML no sustituye la elección con argumentos: documentar por qué el modelo final se elige (no solo por la métrica).

## Limitaciones
1. El fold 3 domina el error en todos los modelos (RF sin lags ≈ 0.76).
2. Ningún fold valida en julio ni agosto; el RMSLE de validación puede no reflejar el test.
3. Los puntajes del tuning y de AutoML son optimistas (la búsqueda ve los mismos folds).
4. AutoML recibió el preprocesador ajustado con todo train (desviación controlada, documentada); el mejor modelo se re-evaluó con el preprocesador por fold.
5. Con `time_budget`, AutoML probó 12 configuraciones por vía en 300 s (Windows, Python 3.13); en otra máquina la búsqueda eligió otra configuración con casi el mismo RMSLE.
6. ⏳ Pendiente avisar a Joan: `flaml[automl]` en `requirements.txt`.
