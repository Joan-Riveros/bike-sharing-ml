# Nota de investigación — José (AutoML y validación temporal)

**Pregunta:** ¿qué herramienta de AutoML respeta el orden temporal con el mismo protocolo que los modelos manuales?

## Qué verifiqué y cómo
- **FLAML 2.7.0 (verificado en el código instalado y con ejecuciones):** `split_type` acepta un objeto con `split` y `get_n_splits` (`generic_task.decide_split_type`). Se le pasó `get_cv()` (`TimeSeriesSplit`, 5 folds, ventana expansiva) con `eval_method="cv"`. La pérdida interna de FLAML y el RMSLE medido con `validar_temporal` coinciden (diferencia 0.0003 sin lags y menos de 0.0001 con lags).
- **PyCaret (`fold_strategy`, `data_split_shuffle`) y H2O (`fold_column`, `nfolds=0`):** ⏳ no los verifiqué en su documentación; vienen del plan de Joan. PyCaret 3.3 exige Python ≤ 3.11 (según el plan); H2O requiere Java.

## Decisión propuesta
FLAML, porque corre sin cambiar la versión de Python del equipo y acepta exactamente los folds del contrato.

## Advertencias
- ✅ Verificado: FLAML 2.7.0 con el extra `flaml[automl]` instala y corre en **Python 3.13.14 (Windows 10)**. `pip install flaml` solo no basta; el extra debe ir a `requirements.txt` (archivo de Joan).
- Con `time_budget` el número de configuraciones probadas depende de la máquina (12 y 12 en 300 s en la corrida final). En otra máquina, con la misma semilla, la búsqueda eligió una configuración distinta (por ejemplo, `learning_rate` 0.59 frente a 0.08 sin lags) con casi el mismo RMSLE: la búsqueda no es reproducible al detalle, el resultado sí es estable. Se registran versión, semilla, presupuesto y log.
- FLAML recibe una matriz, no un `Pipeline`: el preprocesador se ajusta una vez con todo train para la búsqueda. Por eso el mejor modelo se re-evalúa con `validar_temporal`.
