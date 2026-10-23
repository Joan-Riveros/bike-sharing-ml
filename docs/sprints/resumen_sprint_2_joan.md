# Resumen del Sprint 2 — parte de Joan (LightGBM)

Lo reportado sale de código ejecutado; lo no probado se indica como hipótesis y lo no hecho como pendiente. Todo sale de `notebooks/03_modelado_supervisado.ipynb` (Parte II), ejecutado en el `.venv` (Python 3.13.11). Validación temporal solo con train; **el test no se ha usado**. Cada prueba tiene su ficha en `reports/decisiones.md`.

## Resultados (RMSLE medio ± std entre los 5 folds)
| Modelo | RMSLE | Validación |
| :--- | :---: | :---: |
| Baseline media `(hr, workingday)` | 0.5585 ± 0.0601 | `gap=0` |
| LightGBM base | 0.4643 ± 0.1951 | `gap=0` |
| LightGBM + tuning (`log1p` + MSE) | 0.4549 ± 0.1918 | `gap=0` |
| **Candidato: + `lag_24` + `lag_168`** | **0.3905 ± 0.0853** | `gap=24` |

El candidato mejora 30.1 % al baseline de calendario (el baseline se midió con `gap=0`). Está en `models/lgbm_joan.joblib` (recarga verificada). Parámetros en `CONFIG["lgbm"]["params_tuneados"]`.

## Decisiones (regla: mejora ≥ 0.002 y peor en ≤ 1 de 5 folds)
| Prueba | Resultado | Decisión |
| :--- | :--- | :---: |
| Tendencia `sin_yr` / `trend` | `sin_yr` 0.5370 (peor); `trend` 0.4439 pero peor en 3 folds | Descartar; mantener `yr` |
| `windspeed = 0`: `indicador` / `mediana` | `indicador` idéntico; `mediana` 0.4648 | Descartar; mantener `dejar` |
| `hum = 0`: `dejar` | 0.4624 (mejora 0.0019, peor en 2 folds) | Descartar; mantener `mediana` |
| Tuning (40 candidatos) | 0.4549, mejor en 5 folds; semillas ±0.0002 | **Adoptar** |
| Poisson | 0.4473, mejor en 3 y peor en 2 folds | Descartar |
| *Smearing* de Duan (φ ≈ 1.028) | RMSLE 0.4466, peor en 2 folds | No adoptar |
| Lags 24 + 168 | 0.3905 vs 0.4562 (misma validación), mejor en 4 y peor en 1 | **Adoptar** |

Los valores de `CONFIG["features"]` no cambian (`yr`, `dejar`, `mediana`): quedan confirmados **solo para LightGBM**; para Ridge y Random Forest siguen siendo provisionales.

## Qué se cambió en el contrato
- `build_preprocessor(..., lags=())`: parámetro nuevo y opcional al final; sin él todo funciona como en el Sprint 1.
- `src.features.agregar_lags(df, horas=(24, 168))`: crea `lag_<h>` sobre la serie completa de `load_data()`; luego `temporal_split` y `separar_xy` como siempre.
- `CONFIG` nuevo: `lgbm`, `regla_adopcion`.

## Para el resto del equipo
- **Pardo y José:** para una comparación justa, usen `get_cv()` y `validar_temporal`. Si el grupo decide que todos usen lags, usen `agregar_lags` y `get_cv(gap=24)`; con modelos lineales hay que imputar los NaN de los lags (132 y 265 filas en train).
- **José (líder del Sprint 3):** el candidato está en `models/lgbm_joan.joblib` (diccionario con `pipeline`, `log_target`, `prep_kw`, `params`). **La carpeta `models/` está en `.gitignore`**, así que no viaja con git: se regenera ejecutando el notebook 03 (~6 min) o se entrega por otro medio (Pendiente: decisión).

## Limitaciones y riesgos
1. El fold 3 domina el error (RMSLE ≈ 0.78 sin lags; 0.50 con lags): valida 2012 con un train solo de 2011. Los folds 1 a 3 no distinguen `yr` de `sin_yr`.
2. Ningún fold valida en julio ni agosto; el RMSLE de validación puede no reflejar el test.
3. Hipótesis: `trend` podría mejorar si el test muestra sesgo de tendencia; revisar en el Sprint 3 con el RMSLE mensual.
4. Los lags valen solo con el horizonte «día siguiente» y exigen el día anterior completo; la validación operativa (§8.4) debe incluir lags ausentes y Sandy.
5. El puntaje del tuning es optimista; el tuning de Poisson, el early stopping y el *smearing* con residuos fuera de muestra quedan sin probar (pendiente).
6. La regla de adopción es propuesta de Joan; el grupo puede cambiarla (cambiaría las decisiones de Poisson y de φ solo si se relaja a ≤ 2 folds peores).

## Pendiente del Sprint 2
- Pendiente: entregas de Pardo (Ridge/Lasso, clustering) y de José (Random Forest, AutoML) con su tabla de validación por fold y su ficha.
- Pendiente: decisión del grupo: lags para todos o solo para LightGBM; umbral de éxito (propuesta: partir de 0.5585).
- Pendiente: fecha de acceso al dataset en el README; nada está commiteado desde el Sprint 1.
