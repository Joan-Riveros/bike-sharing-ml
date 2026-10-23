# Plan Metodológico y de Trabajo: Bike Sharing Demand
## Proyecto Integrador - Parte I (EC2) | Metodología CRISP-DM

**Leyenda de estado:** ✅ verificado contra `data/raw/hour.csv` · 🔶 hipótesis (razonada, aún no probada) · ⏳ pendiente (requiere experimento o consultar documentación).

> **Estado del repositorio:** notebooks y `src/` están vacíos. No existe ningún resultado de modelado. Toda cifra de RMSLE/RMSE/MAE/R² debe salir de nuestros propios experimentos.
> **Criterio unificado de decisión:** RMSLE en validación temporal (`TimeSeriesSplit` dentro de train). El test (jul–dic 2012) se usa una sola vez al final.

> **[Actualizado en el Sprint 4] El texto de arriba y de las fases 1 a 4 es el plan original y se conserva; el estado actual es este:** los notebooks 01 a 08 y `src/` están completos y todas las cifras salen de código ejecutado (ver `reports/joan/resumen_sprint_4.md` y `reports/decisiones.md`, fichas 1 a 24). Decisiones tomadas:
> - **Horizonte:** día siguiente (24 h), pronóstico emitido a las 00:00 (fichas 20 y 21).
> - **Lags:** `lag_24`, `lag_168` y `lag_336` en el modelo final; el modelo base sigue sin lags.
> - **Modelo final:** LightGBM manual con los hiperparámetros de la ficha 4, objetivo `log1p`. Resultado en test (una sola evaluación): RMSLE 0.3505, RMSE 69.99, MAE 43.11, R² 0.8985.
> - **AutoML:** FLAML (corre en Python 3.13 y no necesita Java); se usó como benchmark (ficha 14).
> - **Entorno:** Python 3.13.11 con las versiones de `requirements.txt`.
> - **Equipo:** por falta de tiempo, Joan hizo solo los Sprints 3 y 4 (los trabajos de Pardo y José del Sprint 2 se conservan).
> - **Umbral de éxito:** propuesto **a posteriori** (ver 2.3).

---

## 1. Resumen y Alineación con Criterios de Evaluación

| Criterio Guía | Puntos | Entregable / Evidencia Asociada |
| :--- | :---: | :--- |
| **Formulación y comprensión del dataset** | 10 pts | Definición formal del problema, objetivo y variable del problema (el objetivo principal y/o el target), criterios de exito, riesgos y restricciones, trazabilidad del origen del dato (`hour.csv`). |
| **Comprensión de los datos** | 5 pts | Fuente y variante (origen del dataset, versión y fecha de acceso), subconjunto y muestreo (que se usa y con que criterio de submuestreo), tabla de variables (variables o entradas principales y su significado), partición temporal y pipeline reproducible. |
| **EDA y preparación de datos** | 10 pts | Hallazgos explicados, control de *data leakage*, partición temporal y pipeline reproducible. |
| **Modelado supervisado y baseline** | 15 pts | Baseline explícito, mínimo 3 modelos clásicos comparables, hiperparámetros fundamentados, compración justa (mismas variables, partición y criterio de evaluación), justificación (elegir con argumentos, no solo por la metrica). |
| **Validación, métricas y errores** | 10 pts | RMSLE, RMSE, MAE, R², tabla comparativa unificada, análisis de residuos. |
| **Aprendizaje no supervisado** | 15 pts | Clustering y/o reducción de dimensionalidad con parámetros justificados e interpretación aplicada, evitar visualizaciones decorativas sin análisis, evitar tratar clusters o componentes como categorías reales sin evidencia. |
| **AutoML y comparación crítica** | 10 pts | Benchmark con herramienta AutoML (H2O o PyCaret), tabla de *trade-offs*, usar misma partición o protocolo equivalente, registrar la configuración del experimento, comparar su mejor resultado con el modelo manual. |
| **Evaluación y análisis de errores** | 5 pts | Métricas coherentes, análisis de errores, estabilidad y generalización, conclusiones con evidencia. |
| **Reproducibilidad y documentación** | 10 pts | Carpetas limpias, `requirements.txt`, `README.md` operativo, informe técnico (8–12 págs). |
| **Presentación y defensa oral** | 10 pts | Diapositivas con la secuencia oficial y dominio de las 10 preguntas guía. |
| **Total** | **100 pts** | |

**Convenciones de trabajo (heredadas de los labs):** `RANDOM_STATE = 42`, `CONFIG` central, `Pipeline` + `ColumnTransformer`, `assert` como puntos de control, funciones de evaluación que devuelven diccionarios, tablas comparativas en `DataFrame`, persistencia con `joblib`.

---

## 2. Fase 1: Comprensión del Negocio

### 2.1 Contexto
- **Problema:** los sistemas de bicicletas compartidas (*Capital Bikeshare*, Washington D.C.) tienen fluctuaciones horarias de demanda ligadas al clima, rutinas laborales y estacionalidad.
- **Uso operativo:** anticipar la demanda para planificar la flota y la redistribución.
- **Alcance (limitación):** el dato es la demanda **total de la ciudad por hora**, no por estación. No se pueden sacar conclusiones por estación.

### 2.2 Formulación
- **Tipo:** regresión supervisada sobre datos tabulares con estructura temporal.
- **Target:** `cnt` (demanda horaria total).
- **Granularidad:** horaria (`hour.csv`); `day.csv` oculta la dinámica de los turnos.
- **Horizonte (propuesta):** predicción del **día siguiente** con pronóstico meteorológico. El clima de entrada sería un pronóstico, no el valor observado.
  - El **modelo base no usa lags**: solo calendario y clima.
  - Los lags de 24 h (`lag_24`) y 168 h (`lag_168`) son admisibles en ese horizonte y se evalúan como **experimento aislado** (ver 6.4).
  - **[Actualizado en el Sprint 4]** Adoptado: horizonte «día siguiente, pronóstico emitido a las 00:00». Se añadió `lag_336` (misma hora de hace dos semanas) y el modelo final usa los tres lags. Alternativa semanal medida: `lag_168` + `lag_336` (RMSLE de validación 0.3893 frente a 0.3732; ficha 21). Los lags solo valen con ese horizonte y sin ningún lag disponible el modelo predice un 85 % menos (ficha 23).

### 2.3 Criterios de éxito
- **Métrica primaria: RMSLE**
  $$\text{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^n \left( \log(y_i + 1) - \log(\hat{y}_i + 1) \right)^2}$$
  Penaliza errores relativos y, a igual error absoluto, castiga más subestimar que sobreestimar. Entrenar sobre `log1p(cnt)` con pérdida cuadrática equivale a optimizar RMSLE.
- **Secundarias:** MAE (bicicletas), RMSE, R². Se calculan en escala original (`expm1`); RMSLE en escala log.
- **Umbral de éxito (⏳):** definir tras obtener el baseline, por ejemplo "mejorar el RMSLE del baseline en al menos X %".
  - **[Actualizado en el Sprint 4] Propuesta de Joan, fijada a posteriori:** mejorar al menos un **30 %** el RMSLE de validación del baseline de calendario (0.5585), es decir, RMSLE de validación ≤ 0.3910. Resultado: validación 0.3732 (−33.2 %, cumple) y test 0.3505 frente a 0.5999 (−41.6 %). ⚠️ El umbral se propuso **después** de ver los resultados de validación y de test; en los Sprints 1 a 3 solo se usó 0.5585 como referencia, sin porcentaje fijado. No es un criterio previo, y que el resultado lo supere no lo valida como tal. Sigue siendo una decisión del grupo (⏳).

### 2.4 Restricciones y riesgos
- **Leakage de variables:** excluir `casual` y `registered` (cnt = casual + registered) e `instant`. No usar `dteday` crudo, solo derivados.
- **Leakage temporal:** prohibido K-Fold aleatorio. Scalers, encoders, imputadores y clusters se ajustan solo con train.
- **Riesgo de tendencia:** el train solo tiene ene–jun 2012 y el test jul–dic 2012; hay crecimiento anual (ver 3.3). Los árboles no extrapolan. 🔶

---

## 3. Fase 2: Comprensión de los Datos (`01_eda_comprension_datos.ipynb`)

### 3.1 Ficha técnica
- **Fuente:** UCI / Capital Bikeshare (Fanaee-T y Gama, 2013; **citar obligatoriamente**, ver `data/raw/Readme.txt`).
- **Archivo:** `data/raw/hour.csv`, 17 379 filas × 17 columnas, 731 días (2011-01-01 a 2012-12-31). ✅
- **Versión y fecha de acceso:** UCI *Bike Sharing Dataset* (versión según `data/raw/Readme.txt`). Fecha de descarga: **[completar]**.
- **Subconjunto y muestreo:** se usan las 17 379 filas completas, sin submuestreo. `day.csv` se conserva solo como referencia. ✅
- **Tabla de variables** (según `data/raw/Readme.txt`):

| Variable | Significado | Uso en el modelo |
| :--- | :--- | :--- |
| `instant` | Índice del registro | Descartada |
| `dteday` | Fecha | Solo para derivar variables y particionar |
| `season` | Estación (1 primavera, 2 verano, 3 otoño, 4 invierno) | Entrada (One-Hot) |
| `yr` | Año (0: 2011, 1: 2012) | Entrada, sujeta a ablación (ver 4.2) |
| `mnth` | Mes (1–12) | Entrada |
| `hr` | Hora (0–23) | Entrada |
| `holiday` | Día feriado | Entrada |
| `weekday` | Día de la semana | Entrada |
| `workingday` | 1 si no es fin de semana ni feriado | Entrada |
| `weathersit` | Situación climática (1 despejado a 4 lluvia/nieve intensa) | Entrada (4 se fusiona con 3) |
| `temp`, `atemp` | Temperatura y sensación térmica, normalizadas (÷41 y ÷50) | Entrada |
| `hum` | Humedad normalizada (÷100) | Entrada |
| `windspeed` | Viento normalizado (÷67) | Entrada |
| `casual`, `registered` | Usuarios casuales y registrados | **Excluidas** como entrada (leakage); solo EDA o targets alternativos |
| `cnt` | Total de alquileres | **Target** |

### 3.2 Hallazgos de calidad de datos ✅
| Hallazgo | Dato verificado | Acción |
| :--- | :--- | :--- |
| **Horas faltantes** | 165 de 17 544 (0.94 %). Concentradas en madrugada (horas 2, 3 y 4 suman 84). 115 en 2011 y 50 en 2012. Meses con más huecos: ene-2011 (56), **oct-2012 (36, coincide con Sandy)**, feb-2011 (23), mar-2011 (14) | Reindexar a 17 544 horas antes de cualquier `shift`. No inventar el target en horas ausentes. |
| **`windspeed = 0`** | 2 180 filas (12.5 %) | 🔶 Probable valor faltante/umbral del sensor. Ablación ⏳ (ver 4.3). |
| **`hum = 0`** | 22 filas, **todas el 2011-03-10** | Implausible; imputar (ver 4.3). |
| **`weathersit = 4`** | 3 filas (0.017 %): 2011-01-26, 2012-01-09 y 2012-01-21; las tres en train | Fusionar con `weathersit = 3`. |
| **`temp` vs `atemp`** | r = 0.9877 | Desnormalizar (`temp*41`, `atemp*50`), conservar `temp` y derivar `temp_diff`. |

### 3.3 Tendencia y composición ✅
- Media de `cnt`: 143.8 (2011) → 234.7 (2012), **+63 %**.
- Totales anuales: `registered` 995 851 → 1 676 811 (**+68 %**); `casual` 247 252 → 372 765 (**+51 %**).
- Composición: `registered` 81.2 %, `casual` 18.8 %.

### 3.4 Hallazgos de EDA a documentar (⏳ confirmar con gráficos)
1. **Dinámica horaria condicional:** días laborales con picos a ~08:00 y ~17:00–18:00; fines de semana y feriados con un pico central. Contrastar `casual` vs `registered` solo en el EDA (no como features).
2. **Distribución del target:** asimetría positiva; justificar `log1p(cnt)` con skewness antes y después.
3. **Multicolinealidad meteorológica:** `temp`–`atemp`. Calcular el VIF real antes de afirmar su reducción.
4. **Clima adverso:** `weathersit = 3` es 8.2 % de las filas; evaluar el modelo en ese subgrupo.
5. **Evento Sandy (2012-10-29/30), dentro del test:** el 29-oct solo hay una hora registrada (22 bicicletas); el 30-oct suma 1 096 bicicletas (mín. horario 39, máx. 143). No se elimina del test; se analiza en los residuos.

---

## 4. Fase 3: Preparación de Datos

### 4.1 Partición (✅ proporciones)
- **Train:** 2011-01-01 a 2012-06-30 (74.8 %). **Test:** 2012-07-01 a 2012-12-31 (hold-out, una sola evaluación).
- **Validación:** `TimeSeriesSplit` dentro de train, **ventana expansiva** (🔶 propuesta, ⏳ comparar con deslizante). K = 4 o 5 folds, con al menos 6–8 meses de train en el primer fold. `gap` de 24–48 h solo si se usan lags.
- **Reentrenamiento final:** reentrenar con todo train y evaluar una sola vez en test.
- **Sesgo por tendencia (⏳):** reportar el **RMSLE mes a mes** en el test (jul–dic). Responde las preguntas 6 y 7 de la defensa.

### 4.2 Ingeniería de características (dentro de `Pipeline` + `ColumnTransformer`)
1. **Hora y mes:**
   - Lineales: splines periódicos (`SplineTransformer(extrapolation='periodic')`), One-Hot o sin/cos. 🔶 ⏳ comparar.
   - Árboles: `hr` como entero o categórica nativa.
   $$\text{hr\_sin} = \sin\left(\tfrac{2\pi\, hr}{24}\right),\; \text{hr\_cos} = \cos\left(\tfrac{2\pi\, hr}{24}\right)$$
2. **Interacción hora × tipo de día:** **categoría combinada** (`hr_workingday`, p. ej. `"18_1"`), no el producto numérico `hr * workingday`.
3. **Clima:** `temp_real`, `temp_diff = atemp_real − temp_real` (desnormalizados), `temp_x_hum`.
4. **Categóricas:** One-Hot de `season` y `weathersit` (con 4 fusionado en 3).
5. **Escalado:** `StandardScaler`/`RobustScaler` solo para modelos sensibles a escala.
6. **Tendencia (ablación ⏳):** (a) `yr` binaria, (b) sin `yr`, (c) índice continuo `trend = días desde 2011-01-01 / 730`.
7. **Opcional (ver 6.4):** lags y medias móviles.

### 4.3 Ablaciones de limpieza (⏳, decidir por RMSLE de validación)
- **`windspeed = 0`:** dejarlo / imputar / dejarlo + indicador `is_windspeed_zero`.
- **`hum = 0`:** imputar con la mediana de la hora del mes o interpolación temporal.
- **Horas faltantes:** dejar el target como NaN y excluirlo de la pérdida; interpolar solo variables meteorológicas en ventanas cortas.

### 4.4 Artefactos
`data/processed/` (train/test) y la lógica en `src/data_loader.py` y `src/features.py`.

---

## 5. Fase 4: Aprendizaje No Supervisado (`02_no_supervisado.ipynb`)

**Base:** lab5 (clustering y `evaluar_clustering`) y lab6 (PCA/UMAP).

1. **Entrada:** `hr_sin`, `hr_cos`, `temp`, `hum`, `windspeed`, `workingday`; sin el target. Escalado ajustado solo con train.
2. **Técnicas:** K-Means (principal) y GMM (BIC/AIC) como comparación. PCA para varianza y *loadings*; UMAP para visualización 2D. *(Opcional: DBSCAN con PCA, BIRCH.)*
3. **Elección de k (⏳):** k = 2..12 con inercia, silueta, Davies-Bouldin y BIC (GMM). 🔶 La silueta sola podría favorecer k=2 (día/noche); decidir con interpretabilidad. **k=4 es conjetura, no resultado.**
4. **Interpretación:** arquetipos esperados 🔶 (conmutación laboral punta; ocio de fin de semana; nocturno de baja demanda; clima adverso). Se confirman o se reemplazan con los centroides reales.
5. **Uso posterior:** ajustar en train, asignar con `predict` en test y calcular el RMSLE del modelo supervisado **por cluster** (Fase 7).

---

## 6. Fase 5: Modelado Supervisado (`03_modelado_supervisado.ipynb`)

**Base:** estructura del lab4 (baseline → pipelines → comparación → CV → tuning del mejor → test único → `joblib`), adaptada a regresión.

### 6.1 Baselines (calculados solo con train)
- **Heurístico:** media de `cnt` por `(hr, workingday)`.
- **Paramétrico:** `LinearRegression` sobre variables numéricas crudas.
- ⏳ El RMSLE de cada uno se obtiene al ejecutarlos.

### 6.2 Modelos (mínimo 3)
1. **Ridge / Lasso** con splines/One-Hot.
2. **Random Forest Regressor.**
3. **LightGBM Regressor** (o `HistGradientBoostingRegressor`).
4. *Opcionales:* XGBoost, CatBoost (ya usados en el lab4).

### 6.3 Objetivo y reversión
- Entrenar sobre `log1p(cnt)`; revertir con `np.expm1` y recortar a ≥ 0.
- ⏳ Comparar con objetivo Poisson (`objective='poisson'`) en RMSLE **y** MAE.
- ⏳ Medir el sesgo de la reversión del log (factor de *smearing* de Duan) antes de corregir.

### 6.4 Tuning y extras opcionales
- **Tuning:** `RandomizedSearchCV` (como lab4) con `TimeSeriesSplit`; Optuna opcional. Repetir con varias semillas y reportar la variación entre folds.
- Rangos iniciales (⏳ validar): LightGBM `num_leaves` [31, 127], `min_child_samples` [20, 100], `learning_rate` [0.03, 0.08] con early stopping, `feature_fraction` [0.7, 0.9], `reg_lambda` [0.1, 10]; Random Forest `max_features` [0.5, 0.8], `min_samples_leaf` [2, 10].
- **Extras (uno a uno sobre baseline + LightGBM; conservar solo los que reduzcan el RMSLE medio entre folds):**
  1. Objetivo Poisson.
  2. `casual` y `registered` por separado, sumados al final (no hay leakage: son targets).
  3. Lags `lag_24`, `lag_168` tras reindexación completa; ninguna ventana incluye el día que se predice.
  4. Optuna (50–100 trials).
- Persistir el mejor pipeline en `models/` con `joblib` y verificar la recarga.

---

## 7. Fase 6: AutoML como Benchmark (`04_automl_benchmark.ipynb`)

### 7.1 Herramienta (⏳ decisión abierta)
| Aspecto | PyCaret 3.3 | H2O AutoML | FLAML |
| :--- | :--- | :--- | :--- |
| Entorno | Python 3.9–3.11 (no 3.12 en 3.3) | Requiere Java | Instalación ligera 🔶 |
| Control de fuga temporal | ⏳ confirmar `fold_strategy` (idealmente un `TimeSeriesSplit`) y `data_split_shuffle=False` | `nfolds=0` + marcos por fecha, sin Stacked Ensembles. ⏳ revisar `fold_column` | ⏳ confirmar `split_type='time'` |
| Visto en los labs | Sí (lab7) | Sí (lab7) | No |

**No usar ninguna herramienta hasta confirmar en su documentación que la validación respeta el orden temporal.** Con CV aleatoria hay leakage.

**[Actualizado en el Sprint 4] Decidido: FLAML** (ficha 14): corre en Python 3.13 sin Java, acepta el `TimeSeriesSplit` de `get_cv()` como `split_type` y no se probó con PyCaret ni H2O.

### 7.2 Protocolo experimental
- Mismo `train` y `test`, mismo target `log1p`, métrica interna RMSE sobre el log (equivale a RMSLE).
- Presupuesto de tiempo declarado (p. ej. 300 s) y registro del tiempo del tuning manual.

### 7.3 Trade-offs (completar solo con evidencia medida)
| Dimensión | Mejor modelo manual | Mejor pipeline AutoML |
| :--- | :--- | :--- |
| RMSLE en test | -- | -- |
| Tiempo (humano / cómputo) | -- | -- |
| Interpretabilidad | -- | -- |
| Control del pipeline y de leakage | -- | -- |

**[Actualizado en el Sprint 4] Valores medidos** (`reports/joan/resumen_sprint_4.md`; la tabla de arriba se conserva como plantilla original):

| Dimensión | Mejor modelo manual (LightGBM) | Mejor pipeline AutoML (FLAML → LightGBM) |
| :--- | :--- | :--- |
| RMSLE en test | 0.3505 (3 lags) / 0.3463 (sin lags) | 0.3389 (config sin lags + 3 lags) / 0.3271 (sin lags) |
| Tiempo (búsqueda / ajuste con todo train) | ≈ 300 s (ficha 4) / 0.68 s | 302 s y 299 s (`automl_config.json`) / 0.76–0.88 s |
| Interpretabilidad | `LGBMRegressor` (SHAP, permutación) | igual: el ganador es un `LGBMRegressor` |
| Control del pipeline y de leakage | total (preprocesador y folds propios); semilla fija | mismo `TimeSeriesSplit`; preprocesador ajustado una vez en la búsqueda; la búsqueda depende del tiempo de cómputo |

---

## 8. Fase 7: Evaluación y Análisis de Residuos

### 8.1 Tabla comparativa (test hold-out) — a llenar con resultados reales
| Modelo | RMSLE | RMSE | MAE | R² | Tiempo |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline (media hr/workingday) | -- | -- | -- | -- | -- |
| Baseline (regresión lineal) | -- | -- | -- | -- | -- |
| Ridge | -- | -- | -- | -- | -- |
| Random Forest | -- | -- | -- | -- | -- |
| LightGBM | -- | -- | -- | -- | -- |
| LightGBM (casual + registered) *(si se adopta)* | -- | -- | -- | -- | -- |
| AutoML | -- | -- | -- | -- | -- |

**[Actualizado en el Sprint 4] Valores medidos en test** (una sola evaluación; notebook 07). La tabla de arriba se conserva como plantilla original. Tiempo = ajuste con todo train, en segundos:

| Modelo | RMSLE | RMSE | MAE | R² | Tiempo |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline (media hr/workingday) | 0.5999 | 150.35 | 107.21 | 0.5316 | 0.002 |
| Baseline (regresión lineal) | no evaluado en test | | | | |
| Ridge (sin lags) | 0.3857 | 86.39 | 51.56 | 0.8454 | 0.08 |
| Random Forest (sin lags) | 0.3812 | 78.91 | 50.87 | 0.8710 | 0.72 |
| LightGBM (sin lags) | 0.3463 | 69.17 | 43.81 | 0.9009 | 0.61 |
| **LightGBM final (3 lags)** | **0.3505** | **69.99** | **43.11** | **0.8985** | 0.68 |
| LightGBM (casual + registered) | no probado (⏳) | | | | |
| AutoML (FLAML, config sin lags, sin lags) | 0.3271 | 67.02 | 42.63 | 0.9069 | 0.88 |

### 8.2 Diagnóstico de errores (⏳)
- Residuos en escala log vs. predicho (homocedasticidad).
- Error por hora, día de semana, mes, `holiday`, `weathersit >= 3` y por cluster.
- **RMSLE mensual en test** (sesgo por tendencia) y análisis del 29–30 de octubre de 2012 (Sandy). Considerar que casi no hay filas el 29-oct.
- Incertidumbre: con un solo semestre de test, bootstrap en bloques de días para ver si la diferencia entre modelos supera su variabilidad. *(Opcional.)*

### 8.3 Interpretabilidad (⏳)
- SHAP (`TreeExplainer`) o *Permutation Importance*, con la salvedad de variables correlacionadas. Las variables dominantes se informan según el resultado.

### 8.4 Validación operativa: ¿funcionaría en un caso real? (⏳)
Demostración de que el modelo serviría en uso real, sin necesidad de desplegarlo en la nube. Se hace con el mejor modelo ya seleccionado, sobre el test.

1. **Sensibilidad al pronóstico del clima.** El modelo usa el clima como entrada; en producción sería un pronóstico, no el valor observado. Añadir ruido a `temp`, `hum` y `windspeed` del test (varios niveles) y medir cuánto empeora el RMSLE. 🔶 Se espera degradación; el objetivo es cuantificarla.
2. **Backtest día a día.** Recorrer el test simulando que cada día se predicen las 24 horas siguientes con la información disponible hasta entonces. Opcional: reentrenar cada mes. Reportar la distribución del error diario (mediana, percentil 90, peores días), no solo el valor global.
3. **Métrica operativa.** Traducir el error a términos de flota: porcentaje de horas con subestimación mayor a un umbral (por ejemplo 20 %), y error en horas pico. El umbral se define con el grupo.
4. **Prueba de inferencia de punta a punta.** Una función en `src/` que reciba fecha y pronóstico, valide el esquema de entrada y devuelva las 24 predicciones. Probarla cargando el `.joblib` guardado y pasándole un día en formato crudo; comprobar que coincide con la predicción del notebook.
5. **Casos límite.** Entradas fuera de rango (temperatura extrema, `windspeed = 0`, `weathersit = 3`), feriados y Sandy. Verificar que el modelo no falle ni prediga valores negativos o absurdos.
6. **Deriva entre train y test.** Comparar las distribuciones de las variables (KS o PSI) para medir cuánto cambió el entorno. Conecta con el riesgo de tendencia y con el monitoreo de la Parte II.

---

## 9. Cierre y Siguientes Pasos (equivale a Despliegue en CRISP-DM)
- Limitaciones: demanda total sin detalle por estación; dependencia del pronóstico meteorológico; tendencia de crecimiento que los árboles no extrapolan.
- Parte II: arquitecturas secuenciales (lab9: RNN/LSTM), lags dinámicos, monitoreo de deriva de datos.

---

## 10. Estructura de Entregables y Repositorio

```text
bike-sharing-ml/
├── data/
│   ├── raw/                  # hour.csv, day.csv originales
│   └── processed/            # train/test
├── notebooks/
│   ├── 01_eda_comprension_datos.ipynb
│   ├── 02_no_supervisado.ipynb
│   ├── 03_modelado_supervisado.ipynb
│   └── 04_automl_benchmark.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Carga, reindexación a 17 544 horas, split temporal
│   ├── features.py           # Pipelines scikit-learn
│   ├── evaluate.py           # Métricas y gráficos de residuos
│   └── predict.py            # Inferencia de punta a punta (ver 8.4)
├── models/                   # Artefactos .joblib
├── reports/                  # Figuras e informe técnico (8-12 págs)
├── config/
├── .gitignore
├── requirements.txt
└── README.md
```
**Pendiente en el repo:** renombrar `requeriments.txt` a `requirements.txt` y completarlo (fijar versiones); llenar `README.md`; crear `__init__.py`, `.gitignore`, `models/`, `reports/`, `config/`, `data/processed/`. El entorno de ejecución del grupo debe acordarse (p. ej. Python 3.11) antes de instalar el AutoML.

---

## 11. Preparación para la Defensa (10 Preguntas)
Responder con **resultados reales** al final; no precargar respuestas.

1. **¿Por qué ML?** Relaciones no lineales entre clima, rutina laboral y estacionalidad. *[Sustentar con el baseline vs. modelos.]*
2. **Hallazgo clave del EDA y decisión que provocó.** *[Completar: p. ej. patrón horario por tipo de día → categoría `hr_workingday`.]*
3. **Data leakage.** Exclusión de `casual`, `registered`, `instant`; split temporal; transformadores ajustados solo con train; sin K-Fold aleatorio.
4. **Métricas.** RMSLE (errores relativos, penaliza más subestimar), MAE (interpretable en bicicletas).
5. **Baseline.** Promedio por `(hr, workingday)`: referencia simple basada solo en calendario. *(No afirmar que sea la regla actual del negocio.)*
6. **Modelo que mejor generalizó.** *[Completar con la tabla del test y el RMSLE mensual.]*
7. **Errores persistentes.** *[Completar con residuos por hora, mes, clima, cluster y Sandy.]*
8. **Significado de los clusters.** *[Completar con los centroides reales.]*
9. **Qué encontró AutoML.** *[Completar con la tabla de trade-offs medida.]*
10. **Qué cambiarían para la Parte II.** Modelos secuenciales (LSTM), lags dinámicos, monitoreo de deriva.

---

## 12. Investigación Pendiente

### 12.1 Por lectura (técnicas y métodos; no requiere correr modelos)
| Tema | Qué confirmar | Dónde |
| :--- | :--- | :--- |
| Ejemplo oficial de scikit-learn *Time-related feature engineering* | Qué codificación de hora funcionó mejor por tipo de modelo; si usa lags | Documentación de scikit-learn (se vio solo el índice) |
| PyCaret | Parámetros reales de `setup` para validación temporal en regresión | Docs de la versión instalada |
| H2O AutoML | Si `fold_column` permite folds cronológicos con ensembles | Docs de H2O |
| FLAML | Opción `split_type='time'` y métrica personalizada | Docs de FLAML |
| Validación en series de tiempo | Folds, ventanas y `gap` | *Forecasting: Principles and Practice* (Hyndman) |
| Factor de smearing de Duan | Cuándo corrige el sesgo al revertir el log | Literatura de regresión log-lineal |
| SHAP y variables correlacionadas | Límites de interpretación | Documentación de SHAP |

### 12.2 Por experimento (requiere entrenar con los datos reales)
Cada uno se decide con el RMSLE de validación temporal:
1. Variante de tendencia: con `yr` / sin `yr` / índice continuo.
2. Tratamiento de `windspeed = 0` y de `hum = 0`.
3. Codificación de la hora por tipo de modelo.
4. `log1p` + MSE vs. Poisson (RMSLE y MAE).
5. Ventana expansiva vs. deslizante; número de folds.
6. Aporte de lags, de `casual`+`registered` por separado y de Optuna.
7. Elección de k (K-Means/GMM) y significado de los clusters.
8. Sesgo por tendencia (RMSLE mensual en test) y comportamiento en Sandy.
9. Diferencia real entre modelos frente a su variabilidad (bootstrap en bloques).
10. Validación operativa (ver 8.4): ruido en el pronóstico del clima, backtest diario, métrica operativa y deriva train/test.
