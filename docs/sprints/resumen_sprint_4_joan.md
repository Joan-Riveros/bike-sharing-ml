# Resumen del Sprint 4 — Evaluación en test, validación operativa y cierre (Joan)

Lo reportado sale de código ejecutado; lo no probado se indica como hipótesis y lo no hecho como pendiente. Todo sale de `notebooks/07_evaluacion_final_test.ipynb` y `notebooks/08_validacion_operativa.ipynb` (0 errores, 16 y 25 `assert`), ejecutados en el `.venv` (Python 3.13.11) y **reproducidos desde cero** en una copia limpia del repositorio con un entorno nuevo (`pip install -r requirements.txt`): mismo RMSLE (0.3505) y predicciones idénticas. Fichas 22 y 23 en `reports/decisiones.md`.

## Modelo final y protocolo
LightGBM manual (hiperparámetros de la ficha 4) con `lag_24`, `lag_168` y `lag_336`, objetivo `log1p`, horizonte «día siguiente, pronóstico emitido a las 00:00». Elegido en el Sprint 3 (fichas 20 y 21) **antes de ver el test**; reentrenado una vez con todo train (13 003 horas) y evaluado una vez sobre 4 376 horas de test (jul–dic 2012). Los demás modelos se evaluaron solo como referencia; no se usaron para decidir.

## Tabla comparativa en test (§8.1)
| Modelo | RMSLE | RMSE | MAE | R² | Ajuste con todo train (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline (media `(hr, workingday)`) | 0.5999 | 150.35 | 107.21 | 0.532 | 0.002 |
| Ridge (splines × temp, sin lags) | 0.3857 | 86.39 | 51.56 | 0.845 | 0.08 |
| Random Forest (sin lags) | 0.3812 | 78.91 | 50.87 | 0.871 | 0.72 |
| LightGBM (sin lags) | 0.3463 | 69.17 | 43.81 | 0.901 | 0.61 |
| AutoML FLAML → lgbm (config sin lags, sin lags) | 0.3271 | 67.02 | 42.63 | 0.907 | 0.88 |
| **LightGBM final (3 lags)** | **0.3505** | **69.99** | **43.11** | **0.8985** | 0.68 |
| AutoML (config sin lags) con 3 lags | 0.3389 | 70.00 | 43.36 | 0.899 | 0.76 |

Con tres lags (referencia): Ridge 0.3802 y Random Forest 0.3795. «LightGBM (casual + registered)» del plan no se probó (pendiente). **El modelo final reduce el RMSLE del baseline en 41.6 %** y queda dentro del rango de los folds de validación (0.3732 ± 0.0566).

Atención: **Hallazgo que hay que reportar:** los lags no mejoraron a LightGBM en el test (0.3505 frente a 0.3463 sin lags) y AutoML sin lags fue el mejor del test (0.3271). No se cambia el modelo: elegirlo con el test dejaría sin estimación independiente. Los lags ayudan en nov–dic y empeoran en jul–oct; ningún fold de validación cubría julio ni agosto (riesgo ya advertido en los Sprints 1 y 3).

## RMSLE mensual (sesgo por tendencia)
| Mes | jul | ago | sep | oct | nov | dic |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| RMSLE final | 0.333 | 0.243 | 0.290 | 0.308 | 0.412 | 0.466 |
| Sesgo (log, > 0 subestima) | +0.082 | +0.058 | +0.055 | +0.084 | +0.023 | +0.004 |

Subestimación leve en verano y otoño (la demanda real media supera a la predicha en 30 bicicletas en julio), sin deriva de tendencia fuerte. Peores cortes: clima adverso (RMSLE 0.654, sesgo +0.182), feriados (0.664, sobreestima), madrugada (2:00 = 0.520).

## Validación operativa (§8.4)
| Prueba | Resultado |
| :--- | :--- |
| 1. Ruido en el clima | RMSLE 0.3505 → 0.3583 (nivel 1) → 0.3716 (nivel 2, +6.0 %) → 0.3901 → 0.4231 (nivel 4, +20.7 %). Sin lags degrada más (+26.6 % en el nivel 4); desde el nivel 2 el modelo con lags es igual o mejor |
| 2. *Backtest* día a día (178 días) | RMSLE diario: mediana 0.267, P90 0.475, máximo 0.916 (26-dic). Error del total diario: mediana 10.3 %, P90 25.7 %, sesgo −6.9 %. Reentrenar cada mes: 0.3469 (−1.0 %) |
| 3. Métrica operativa (u = 20 %, propuesta pendiente de aprobación) | Subestima > 20 % en 23.2 % de las horas (16.1 % en horas pico) y sobreestima en 16.4 %. En horas pico el sesgo es −50.3 bicicletas/hora y el faltante medio de las 139 horas con subestimación > 20 % es 176 |
| 4. `predict.py` de punta a punta | Reproduce las predicciones del notebook 07 en 4 272 horas (diferencia máxima 5.7e-14); rechaza los 6 días con horas ausentes |
| 5. Casos límite | Ninguno falla ni predice negativo (máximo 834 bicicletas/hora); frío extremo −45.9 %, clima adverso −40.8 %, humedad 100 % −31.7 %. **Sin historial (3 lags ausentes): −84.8 %**; sin `lag_24`: −6.7 % |
| 6. Deriva train/test | La demanda cambia más que el clima: PSI 0.223 (demanda), 0.264 (`lag_168`), 0.213 (temp); humedad 0.070 y viento 0.041 estables |

**Respuesta a «¿funcionaría en un caso real?»:** sí, con condiciones: pronóstico completo de 24 horas, historial reciente completo (si no, usar un modelo sin lags), un margen de seguridad en horas pico (subestima ≈ 50 bicicletas/hora) y reentrenamiento periódico por el crecimiento de la demanda.

## Trade-offs AutoML frente al mejor manual (§7.3, medido)
| Dimensión | LightGBM manual final | AutoML (FLAML) |
| :--- | :---: | :---: |
| RMSLE en test | 0.3505 (con 3 lags) / 0.3463 (sin lags) | 0.3389 (config sin lags + 3 lags) / **0.3271** (sin lags) |
| RMSLE de validación | 0.3732 ± 0.0566 (3 lags) | 0.3738 ± 0.0673 (3 lags) / 0.4405 (sin lags) |
| Búsqueda de hiperparámetros | ≈ 300 s (ficha 4), semilla fija | 302 s y 299 s (`automl_config.json`), depende del tiempo de cómputo |
| Robustez si falta `lag_24` (validación) | 0.3862 | 0.3985 |
| Interpretabilidad | `LGBMRegressor` (SHAP, permutación) | igual: el ganador es un `LGBMRegressor` |
| Control del pipeline y de fuga | total (preprocesador y folds propios) | se pasó el mismo `TimeSeriesSplit`; preprocesador ajustado una vez en la búsqueda |

## Verificaciones hechas
- Reproducción desde cero: copia limpia, entorno nuevo, `pip install -r requirements.txt` (247 s), `pip check` sin conflictos, notebooks 07 y 08 sin errores, RMSLE 0.3505 e idénticas predicciones.
- `src/predict.py`: `estacion()` coincide con `season` en las 17 544 horas; `construir_entrada` reproduce las columnas de entrenamiento; rechaza esquema, rangos, 24 horas, `weathersit` fuera de 1–4 e historial con datos del futuro.
- No hay referencias a herramientas de IA en el código, los notebooks ni los documentos.

## Pendiente
- Pendiente: **Informe técnico (8–12 págs) y diapositivas:** falta acordar el formato y la estructura; los resultados para ambos están en este resumen, en `docs/preguntas_defensa.md` y en las figuras de `reports/joan/figuras/` (01 a 21).
- Pendiente: **Umbral de éxito del plan (§2.3), propuesta a posteriori:** mejorar al menos un **30 %** el RMSLE de validación del baseline de calendario (0.5585), es decir, ≤ 0.3910. Contraste: validación 0.3732 (−33.2 %, cumple) y test 0.3505 frente a 0.5999 (−41.6 %). Atención: se propuso **después** de ver la validación y el test (antes solo se usó 0.5585 como referencia, sin porcentaje fijado), así que no es un criterio previo ni lo valida el resultado; es decisión del grupo.
- Pendiente: **Otras decisiones del grupo:** umbral de la métrica operativa (propuesta u = 20 %) y fechas de los sprints.
- Pendiente: **Producción:** interpolar horas ausentes del pronóstico, modelo sin lags como respaldo, calibrar los niveles de ruido con pronósticos reales.
- Pendiente: `LightGBM (casual + registered)` (extra §6.4) y Optuna no se probaron.
- Los Sprints 3 y 4 están commiteados en la rama `joan/sprint-4` (commit `f06e960`) y subidos a GitHub; **aún no están mezclados en `main`** (pendiente: falta integrar la rama de José y abrir el PR).
