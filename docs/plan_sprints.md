# Plan de Sprints: Bike Sharing Demand

Equipo: **Joan** (Joan Marcelo Riveros Soria), **Pardo** (Alejandro Miguel Pardo Romano) y **José** (Jose Andres Manzaneda Manzaneda). En este documento se usan los nombres cortos como etiquetas de rol. Plan de referencia: `project_plan_crisp_dm.md` (las secciones se citan como "§").
Fechas de cada sprint: **[completar con el grupo]**.

---

## 1. Reparto general

| Bloque | Responsable | Entregables principales |
| :--- | :---: | :--- |
| **Inicio:** repositorio, datos, EDA, preparación, baselines (§2–4, §6.1) | **Joan** | Repo y entorno, `data_loader.py`, `features.py`, `evaluate.py`, notebook 01, baselines |
| **Núcleo:** LightGBM como modelo principal, tuning y extras (§6.3–6.4) | **Joan** | Mejor modelo manual en `models/` |
| **Validación para probarlo:** validación operativa (§8.4) | **Joan** | `predict.py`, ruido de clima, backtest, métrica operativa, deriva |
| **Cierre:** evaluación final en test, README, informe, diapositivas, defensa (§9–11) | **Pardo** (lidera el Sprint 4) | Entregables finales; Joan y José apoyan |
| **No supervisado** (§5) | **Pardo** | Notebook 02 |
| **Ridge/Lasso** (§6.2) y **errores por cluster** (§8.2) | **Pardo** | Modelo lineal y análisis de errores por cluster |
| **AutoML** (§7) | **José** | Notebook 04 |
| **Random Forest** (§6.2) e **interpretabilidad / residuos** (§8.2–8.3) | **José** | Modelo de bagging, SHAP y residuos |

Carga aproximadamente equilibrada: Joan lleva la base y el modelo principal, y Pardo y José llevan más experimentación en paralelo y, desde el Sprint 3, el liderazgo de integración.

---

## 1.1 Responsable de cada sprint

Cada sprint tiene **una persona a cargo** (el "líder del sprint"). Es quien evita que el sprint se trabe.

| Sprint | Tema | Líder | Suplente (si el líder no está disponible) |
| :--- | :--- | :---: | :---: |
| 0 | Preparación y lectura | Joan | — |
| 1 | Datos, EDA y preparación | **Joan** | José |
| 2 | Modelado en paralelo | **Joan** | Pardo |
| 3 | Comparación, errores e interpretabilidad | **José** | Joan |
| 4 | Test, validación operativa y cierre | **Pardo** | José |

**Estado:** Sprint 0 completado por Joan. (verificar con los demás que el entorno les corre)

**[Actualizado en el Sprint 4] Estado real de los sprints** (la tabla de arriba es el reparto original y se conserva):

| Sprint | Estado | Quién lo hizo | Dónde está |
| :--- | :---: | :--- | :--- |
| 0 | Hecho | Joan; Pardo y José dejaron su nota de investigación | `docs/sprints/nota_investigacion_pardo.md`, `docs/sprints/nota_investigacion_jose.md` |
| 1 | Hecho | Joan | `docs/sprints/resumen_sprint_1_joan.md` |
| 2 | Hecho | Joan (LightGBM), Pardo (Ridge/Lasso y clustering), José (Random Forest y AutoML) | `docs/sprints/resumen_sprint_2_*.md`, fichas 1 a 14 |
| 3 | Hecho | **Joan solo** (el líder previsto era José; por falta de tiempo no hubo reparto) | `docs/sprints/resumen_sprint_3_joan.md`, fichas 15 a 21 |
| 4 | Hecho en lo técnico | **Joan solo** (el líder previsto era Pardo) | `docs/sprints/resumen_sprint_4_joan.md`, fichas 22 a 24; informe y diapositivas pendientes de cierre |

Decisiones tomadas: horizonte «día siguiente, pronóstico a las 00:00»; lags `lag_24`, `lag_168` y `lag_336`; modelo final LightGBM manual (hiperparámetros de la ficha 4); AutoML con FLAML como benchmark. El umbral de éxito se propuso **a posteriori** (ver `project_plan_crisp_dm.md` §2.3) y sigue siendo decisión del grupo. Las fechas de los sprints siguen sin definirse (pendiente).

**Qué hace el líder de sprint**
1. Al inicio: confirma las tareas de cada persona, las fechas y qué entrega cada quien a los demás.
2. Durante el sprint: desbloquea. Si alguien lleva más de **24 horas** bloqueado, lo resuelve o lo escala al grupo.
3. Revisa las integraciones a `main` del sprint (o designa quién las revisa) y verifica que se cumpla la *definición de hecho*.
4. Al cierre: resume en una nota corta (`docs/sprints/resumen_sprint_N_<lider>.md`) qué se hizo, qué se decidió, qué quedó pendiente y qué recibe el siguiente líder.

**Qué NO implica ser líder:** no es hacer todo el trabajo, ni decidir solo. Las decisiones del grupo (herramienta AutoML, umbrales, fechas) se toman en conjunto.

**Traspaso entre sprints:** el líder saliente y el entrante hacen una reunión corta (15–20 min) para pasar el contrato de entrega del siguiente sprint.

---

## 2. Reglas del equipo

### 2.1 Investigar e innovar (permitido, con condiciones)
Se trabaja con lo visto en clase (labs 1, 4, 5, 6 y 7: regresión lineal, boosting, clustering, PCA/UMAP, AutoML). Se puede probar algo nuevo si se cumple lo siguiente:
1. Se compara contra la versión de referencia **con el mismo split temporal y la misma métrica** (RMSLE de validación).
2. Se registra una **ficha de decisión** (plantilla abajo) en `reports/decisiones.md`.
3. Se explica **por qué** se probó y **qué resultado tuvo**, aunque no haya mejorado.
4. Si no mejora de forma consistente entre folds, no se adopta.

**Plantilla de ficha de decisión**
```text
Tema:            (p. ej. objetivo Poisson en LightGBM)
Responsable:
Por qué se probó:
Qué se hizo:     (configuración exacta, semilla, split)
Resultado:       (tabla: referencia vs. variante, RMSLE por fold)
Decisión:        adoptar / descartar / pendiente
Fuente consultada:
```

### 2.2 Convenciones técnicas (todos)
- `RANDOM_STATE = 42`, `CONFIG` central, `Pipeline` + `ColumnTransformer`, `assert` como puntos de control.
- Prohibido K-Fold aleatorio. Todo ajuste (scaler, encoder, imputador, clusters) solo con train.
- El test (jul–dic 2012) **no se toca** hasta el Sprint 4 y solo para el modelo final.
- No subir datos grandes ni artefactos sin revisar. No editar archivos de otra persona sin avisar.

### 2.3 Propiedad de archivos (evita conflictos en git)
| Archivo / carpeta | Dueño |
| :--- | :---: |
| `src/data_loader.py`, `src/features.py`, `src/evaluate.py`, `src/predict.py`, `config/`, `requirements.txt` | Joan |
| `README.md` | Joan hasta el Sprint 3; **Pardo** en el Sprint 4 (cierre) |
| `notebooks/01_eda_comprension_datos.ipynb`, `notebooks/03_modelado_supervisado.ipynb` (sección LightGBM y baselines) | Joan |
| `notebooks/02_no_supervisado.ipynb` | Pardo |
| `notebooks/04_automl_benchmark.ipynb` | José |
| `models/`, `reports/` | Cada quien en subcarpeta propia (`reports/pardo/`, `reports/jose/`, `reports/joan/`) |

Cada persona trabaja en su rama (`joan/...`, `pardo/...`, `jose/...`) y se integra a `main` por revisión de otra persona.
Pardo y José agregan sus modelos al notebook 03 en **notebooks auxiliares** (`03b_ridge.ipynb`, `03c_random_forest.ipynb`); José (líder del Sprint 3) los consolida en la tabla comparativa.

---

## 3. Sprint 0: Preparación y lectura (todos) — Líder: Joan (completado)

**Objetivo:** que cualquiera pueda correr el proyecto y que la investigación inicial esté repartida.

| Persona | Tareas |
| :--- | :--- |
| **Joan** | Crear la estructura del repo (§10), renombrar `requeriments.txt` a `requirements.txt`, `.gitignore`, `__init__.py`. Definir y fijar el entorno común (versión de Python, p. ej. 3.11). Documentar la fecha de acceso al dataset. |
| **Pardo** | Instalar el entorno y verificar que corre. Leer lab5 y lab6 de sus labs. Investigar: elección de k (codo, silueta, BIC), manejo de variables cíclicas en clustering, estabilidad de clusters. |
| **José** | Instalar el entorno y verificar que corre. Revisar lab7 (H2O y PyCaret). Investigar en la documentación: validación temporal en PyCaret (`fold_strategy`, `data_split_shuffle`), H2O (`fold_column`, `nfolds=0`) y FLAML (`split_type`). Proponer qué herramienta usar. |

**Hecho cuando:** los tres corren `import` de las librerías, el repo tiene la estructura y cada quien dejó una nota de investigación (máx. una página) en `docs/sprints/`.

---

## 4. Sprint 1: Datos, EDA y preparación — Líder: Joan

**Objetivo:** dejar listo el contrato de datos que usan los demás.

### Joan
- `src/data_loader.py`: carga de `hour.csv`, **reindexación a 17 544 horas**, partición temporal (train hasta 2012-06-30; test desde 2012-07-01).
- `notebooks/01_eda_comprension_datos.ipynb`: hallazgos de §3.2–3.4 con gráficos y texto interpretativo (horas faltantes, `windspeed = 0`, `hum = 0`, `weathersit = 4`, tendencia, Sandy, casual vs registered).
- `src/features.py`: pipeline base (§4.2) con ablaciones de limpieza (§4.3) y de tendencia (`yr` / sin `yr` / índice continuo).
- `src/evaluate.py`: `rmsle`, `rmse`, `mae`, `r2`, función `evaluar_pipeline` que devuelve un diccionario y `TimeSeriesSplit` ya configurado.
- Baselines (§6.1) con sus métricas de validación.

### Pardo y José (en paralelo, sin depender de Joan hasta que haya contrato)
| Persona | Tarea mientras tanto |
| :--- | :--- |
| **Pardo** | Prototipar el clustering sobre `hour.csv` directamente (solo con fechas de train) para ir probando entradas y k. Revisar si la silueta sola sugiere k=2. |
| **José** | Probar la herramienta AutoML elegida con un dataset de juguete (como en lab7) y con una partición temporal simple; verificar que respeta el orden de las fechas. |

**Contrato de entrega de Joan al final del sprint** (lo que Pardo y José pueden asumir):
```python
from src.data_loader import load_data, temporal_split
from src.features import build_preprocessor
from src.evaluate import rmsle, evaluar_pipeline, get_cv
```
**Hecho cuando:** el EDA está documentado, el contrato funciona con `assert`, y los baselines tienen RMSLE de validación.

---

## 5. Sprint 2: Modelado en paralelo — Líder: Joan

**Objetivo:** tres modelos clásicos comparables, con el mismo split, las mismas variables y la misma métrica.

| Persona | Tareas | Investigación propia |
| :--- | :--- | :--- |
| **Joan** | **LightGBM** (modelo principal): entrenamiento sobre `log1p(cnt)`, tuning con `RandomizedSearchCV` + `TimeSeriesSplit`, ablación de tendencia, comparación `log1p`+MSE vs Poisson (§6.3). | Hiperparámetros de mayor impacto y su rango. Reversión del log (factor de *smearing*). |
| **Pardo** | **Ridge/Lasso** con codificación de hora: splines periódicos vs One-Hot vs sin/cos. **Clustering:** K-Means y GMM, PCA y UMAP, elección de k, interpretación de centroides (§5). | Codificaciones de hora (ejemplo de scikit-learn *Time-related feature engineering*). Estabilidad de clusters. |
| **José** | **Random Forest** (`max_features`, `min_samples_leaf`, etc.). **AutoML:** configurar y correr con el protocolo de §7.2; registrar configuración y tiempo. | Validación temporal en la herramienta AutoML. Si se probó algo más (XGBoost, CatBoost), ficha de decisión. |

**Extras opcionales** (§6.4): se reparten según el interés, uno por persona, con ficha de decisión:
- Joan: lags (`lag_24`, `lag_168`).
- Pardo: `casual` + `registered` por separado.
- José: Optuna.

**Hecho cuando:** cada persona entrega su modelo/notebook con métricas de validación por fold (media y desviación) en una tabla y su ficha de decisión.

---

## 6. Sprint 3: Comparación, errores e interpretabilidad — Líder: José
> **[Actualizado en el Sprint 4]** Lo hizo Joan solo (ver el estado real en §1.1): `notebooks/05_comparacion_y_errores.ipynb` y `notebooks/06_horizonte_y_lags.ipynb`. Pardo y José no tuvieron tareas en este sprint; sus modelos se reconstruyeron con las configuraciones de sus fichas.

**Objetivo:** elegir el mejor modelo con argumentos y entender sus errores.

| Persona | Tareas |
| :--- | :--- |
| **José (líder)** | Consolidar la tabla comparativa de validación (§8.1) con los modelos de los tres. **Elegir el modelo final** con argumentos (no solo por la métrica) y reentrenar con todo train. Documentar la comparación de AutoML frente al mejor manual (§7.3). **Residuos y sesgo:** residuos vs. predicho (log), RMSLE por mes (sesgo por tendencia), `holiday`, `weathersit >= 3`. |
| **Pardo** | **Errores por cluster** (§5 punto 5, §8.2): RMSLE por cluster. Residuos por hora y día de la semana. Verificar que los clusters se tratan como hipótesis, no como categorías reales. |
| **Joan** | **Interpretabilidad** (§8.3): SHAP o permutation importance sobre el modelo final. Entregar a José el LightGBM final ajustado (en `models/`) y revisar la consolidación de la tabla (que las condiciones sean las mismas para todos). |

**Hecho cuando:** existe un único modelo final elegido, con justificación escrita, y el análisis de errores está completo salvo el test.

---

## 7. Sprint 4: Evaluación en test, validación operativa y cierre — Líder: Pardo
> **[Actualizado en el Sprint 4]** Lo hizo Joan solo (ver el estado real en §1.1): `notebooks/07_evaluacion_final_test.ipynb`, `notebooks/08_validacion_operativa.ipynb`, `notebooks/09_demostracion_prediccion.ipynb` y `src/predict.py`; README final e informe técnico en Markdown (`reports/joan/informe_tecnico.md`). Diapositivas pendientes.

**Objetivo:** evaluar una sola vez en test, demostrar que funcionaría en un caso real y entregar.

| Persona | Tareas |
| :--- | :--- |
| **Pardo (líder)** | **Evaluación final en test** (una sola vez, con el modelo elegido en el Sprint 3) y llenado de la tabla §8.1. **Cierre:** `README.md` operativo, integración del informe técnico (8–12 págs) y de las diapositivas, y verificación de que el repo se reproduce desde cero. Reparte las secciones del informe y fija la fecha límite de cada una. |
| **Joan** | **Validación operativa (§8.4):** `src/predict.py`; sensibilidad al ruido del clima; backtest día a día; métrica operativa; casos límite; deriva train/test. Texto del informe de datos/EDA, preparación, modelo principal y validación operativa. |
| **José** | Texto del informe de **AutoML**, **residuos** e **interpretabilidad**. Probar `predict.py` de Joan con casos propios (revisión cruzada). Revisar el informe completo (coherencia y datos). |

Todos preparan las respuestas a las 10 preguntas de su área (§8 de este documento) con resultados reales (§11 del plan).

**Hecho cuando:** el repo se reproduce desde cero con las instrucciones del `README.md`, el informe y las diapositivas están listos y los tres pueden responder cualquiera de las 10 preguntas.

---

## 8. Reparto de la defensa (preguntas guía, §11)

| Preguntas | Responsable principal |
| :--- | :---: |
| 1, 2, 3, 4, 5 (problema, EDA, leakage, métricas, baseline) | Joan |
| 8 (clusters) | Pardo |
| 9 (AutoML) | José |
| 6, 7 (mejor modelo, errores) | Joan y José |
| 10 (siguientes pasos) | Pardo |

Todos deben poder responder cualquiera; el responsable principal prepara la respuesta con evidencia.

---

## 9. Seguimiento
- Reunión corta al inicio y al final de cada sprint, dirigida por el líder del sprint (ver 1.1): qué se hizo, qué se probó, qué bloquea.
- Cualquier bloqueo de más de 24 horas se avisa al líder del sprint.
- Estado de cada tarea: **pendiente / en curso / en revisión / hecha**.
- Todo resultado numérico del informe debe poder rastrearse a un notebook o a una ficha de decisión.
