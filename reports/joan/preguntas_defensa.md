# Las 10 preguntas guía de la defensa, con resultados reales

Cada respuesta cita el notebook o la ficha de donde sale la cifra. ✅ verificado · 🔶 hipótesis · ⏳ pendiente. Responsables principales según `plan_sprints.md` §8, pero todos deben poder responder cualquiera.

## 1. ¿Por qué ML? *(Joan)*
La relación entre demanda, hora, tipo de día y clima **no es lineal ni aditiva**. En validación temporal, un lineal sobre las variables crudas da RMSLE 1.1055 (peor que el baseline de calendario, 0.5585; notebook 03), y solo mejora a 0.4885 al construir a mano la interacción hora × temperatura con splines (Ridge, ficha 9). Los árboles con gradiente la aprenden solos: 0.4549 (LightGBM, ficha 4). En el test: baseline **0.5999**, Ridge 0.3857, Random Forest 0.3812, LightGBM sin lags 0.3463 y **0.3505 con lags** (notebook 07). El modelo final reduce el error del baseline en **41.6 %**.

## 2. Hallazgo clave del EDA y decisión que provocó *(Joan)*
**El perfil horario depende del tipo de día:** en días laborables hay dos picos (8:00 con ≈ 418 bicicletas y 17:00 con ≈ 462), en fines de semana y feriados una sola campana con máximo a las 13:00 (≈ 334); los registrados llevan los picos laborales y los casuales dominan el fin de semana (notebook 01, sección 6). **Decisión:** la categoría combinada `hr_workingday` (48 niveles) en lugar de `hr × workingday` numérico. Pardo lo confirmó: sin ella, los lineales dan 0.68–1.10 según la codificación de la hora; con ella, las cuatro codificaciones empatan en 0.4930–0.4932 (ficha 8). Otros hallazgos con decisión: 165 horas ausentes (reindexar, target NaN), `weathersit = 4` (3 filas, fusionado con 3), `temp`–`atemp` con VIF 68.7 (→ `temp_c` y `temp_diff`, VIF 1.4–1.5) y el crecimiento de +74 % en ene–jun de 2012 (→ `yr` y, después, los lags).

## 3. Data leakage *(Joan)*
- `casual`, `registered` (cnt = casual + registered), `instant` y `dteday` crudo **nunca** son entradas.
- **Partición temporal:** train hasta 2012-06-30, test desde 2012-07-01; validación con `TimeSeriesSplit` (ventana expansiva), **sin K-Fold aleatorio**.
- Escalado, One-Hot, imputación (medianas), clusters y modelos se ajustan solo con train y dentro del `Pipeline`; `validar_temporal` clona el pipeline en cada fold.
- **Lags:** solo miran el pasado de cada hora; `predict.py` rechaza un historial con horas del día a pronosticar o posteriores.
- **El test se usó una sola vez** con un modelo elegido antes de verlo (fichas 20–22). Los modelos de referencia del test no influyeron en la elección.
- Los clusters se ajustan sin el target.

## 4. Métricas *(Joan)*
Métrica de decisión: **RMSLE** (errores relativos; a igual error absoluto castiga más subestimar; entrenar con `log1p` y pérdida cuadrática la optimiza). Se informan también MAE (bicicletas, interpretable), RMSE y R². Test del modelo final: **RMSLE 0.3505, RMSE 69.99, MAE 43.11, R² 0.8985**. Matiz: el RMSLE está dominado por horas de poca demanda (en la validación la dispersión del residuo es 2.6 veces mayor en el tramo bajo), por eso se complementa con la métrica operativa (23.2 % de las horas con subestimación > 20 %, ficha 23).

## 5. Baseline *(Joan)*
Media de `cnt` por `(hr, workingday)`: una referencia simple basada solo en el calendario (no afirmamos que sea la regla de la empresa). Validación: 0.5585 ± 0.0601; test: 0.5999 (RMSE 150.4, MAE 107.2, R² 0.53). Es mejor que la regresión lineal cruda (1.1055) y falla de forma sistemática en 2012: subestima por el crecimiento (sesgo +0.47 a +0.53 en jul–sep).

## 6. ¿Qué modelo generalizó mejor? *(Joan y José)*
**Elegido (con la validación): LightGBM manual con `lag_24`, `lag_168` y `lag_336`**, 0.3732 ± 0.0566 en validación (el más estable) y **0.3505 en test**: la validación no sobreestimó el rendimiento. RMSLE mensual en test: 0.333, 0.243, 0.290, 0.308, 0.412 y 0.466 (jul–dic). **Honestidad sobre el test:** LightGBM sin lags (0.3463) y AutoML sin lags (0.3271) rindieron algo mejor que el final en el test; los lags ayudaron en nov–dic y perjudicaron en jul–oct. No cambiamos el modelo después de ver el test. Elegido frente a AutoML por ser igual de preciso en validación (0.3732 frente a 0.3738) y más robusto si falta `lag_24` (0.3862 frente a 0.3985).

## 7. Errores persistentes *(Joan y José)*
- **Subestimación:** −6.9 % en el total diario; sesgo +0.05 en log.
- **Clima adverso** (RMSLE 0.654 en test, sesgo +0.182) y **feriados** (0.664, sobreestima): son pocas horas (290 y 143) y los lags traen el patrón normal.
- **Madrugada** (2:00 = 0.520) y **diciembre** (0.466): poca demanda, error relativo grande.
- **Sandy** (30-oct): predice 1 606 bicicletas frente a 1 096 registradas, sobre las 11 horas con registro (de 13:00 a 23:00); a las 19:00 prevé 404 y hubo 115. Los lags tardan días en reflejar un evento extremo.
- **Clusters:** cluster 0 (tarde cálida) 0.223; clusters 2 y 3 (no laborable, noche laborable) 0.403 y 0.412.
- **Fallo operativo:** sin ningún lag, la predicción cae 84.8 %.

## 8. Significado de los clusters *(Pardo)*
K-Means con k = 4 sobre hora (sin/cos), temperatura, humedad, viento y `workingday`. Arquetipos: tarde cálida y seca (demanda media 278), mañana laborable (138), no laborable (125) y noche laborable (118). **La estructura es débil** (silueta 0.18; ARI entre semillas 0.997) y los clusters explican solo el 17.5 % de la varianza de `log1p(cnt)` frente al 81 % de hora × `workingday`: se usan para segmentar el error, no como variable. Ningún cluster captura el clima adverso (el clima no es entrada).

## 9. Qué encontró AutoML *(José)*
FLAML (Python 3.13, sin Java; el mismo `TimeSeriesSplit` que los demás). Encontró una región que el tuning manual no exploraba: **árboles muy pequeños** (`num_leaves = 4`, `n_estimators = 2 027`) sin lags. Resultados: validación 0.4405 frente a 0.4549 del manual (gana en 4 de 5 folds, IC95 % [0.0077, 0.0175]); con lags empatan (0.3876 y 0.3905; con los tres lags 0.3738 y 0.3732). En test, AutoML sin lags fue el mejor (0.3271). Trade-offs: búsqueda de ≈ 300 s en ambos, pero la de FLAML depende del tiempo de cómputo (dos ejecuciones dieron configuraciones distintas); el ganador es un `LGBMRegressor`, así que no pierde interpretabilidad; es más frágil si falta `lag_24` (0.3985). Ver ficha 14 (corregida) y el resumen del Sprint 4.

## 10. Qué cambiarían para la Parte II *(Pardo)*
- **Modelos secuenciales** (LSTM, lab 9) sobre el historial en lugar de lags fijos.
- **Lags dinámicos y un modelo de respaldo sin lags** cuando falte historial (el caso que más degrada al modelo).
- **Interpolar horas ausentes del pronóstico** y calibrar el ruido con pronósticos reales.
- **Monitoreo de deriva** con el PSI de la demanda y de sus lags (0.223 y 0.264 entre train y test), que mide el cambio de nivel que degrada al modelo.
- **Reentrenamiento periódico** (mensual: −1.0 % de RMSLE en el backtest) y demanda por estación en lugar de total de ciudad.
