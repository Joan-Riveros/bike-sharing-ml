# Plan Metodológico y de Trabajo: Bike Sharing Demand
## Proyecto Integrador - Parte I (EC2) | Metodología CRISP-DM

---

## 1. Resumen y Alineación con Criterios de Evaluación

| Criterio Guía | Puntos | Entregable / Evidencia Asociada |
| :--- | :---: | :--- |
| **Formulación y comprensión del dataset** | 10 pts | Definición formal del problema, análisis de riesgos y restricciones, trazabilidad del origen del dato (`hour.csv`). |
| **EDA y preparación de datos** | 15 pts | Hallazgos explicados (no solo gráficos), control estricto de *data leakage*, protocolo de partición temporal y pipeline reproducible. |
| **Modelado supervisado y baseline** | 15 pts | Baseline heurístico explícito, mínimo 3 modelos clásicos comparables, optimización de hiperparámetros fundamentada. |
| **Validación, métricas y errores** | 15 pts | Métricas seleccionadas (RMSLE, RMSE, MAE), tabla comparativa unificada, análisis cualitativo y cuantitativo de residuos. |
| **Aprendizaje no supervisado** | 15 pts | Clustering o reducción de dimensionalidad con justificación de parámetros e interpretación aplicada al negocio. |
| **AutoML y comparación crítica** | 10 pts | Benchmark con herramienta AutoML (ej. FLAML/AutoGluon), tabla de *trade-offs* (rendimiento, tiempo, interpretabilidad). |
| **Reproducibilidad y documentación** | 10 pts | Estructura de carpetas limpia, `requirements.txt`, `README.md` operativo e informe técnico (8–12 págs). |
| **Presentación y defensa oral** | 10 pts | Diapositivas siguiendo la secuencia oficial y dominio de las 10 preguntas guía. |

---

## 2. Fase 1: Comprensión del Negocio (Business Understanding)

### 2.1 Contexto y Necesidad Operativa
- **Problema real:** Los sistemas de micromovilidad compartida (*Capital Bikeshare*) enfrentan fluctuaciones horarias drásticas de demanda impulsadas por clima, rutinas laborales y estacionalidad.
- **Impacto operativo:** La escasez de bicicletas en estaciones clave durante horas pico genera pérdida directa de ingresos y fricción de servicio; el exceso genera costos elevados en camiones de redistribución y logística de mantenimiento preventivo.

### 2.2 Formulación del Problema y Target
- **Tipo de problema:** Regresión supervisada sobre series de tiempo tabulares multivariadas.
- **Variable objetivo ($y$):** Demanda horaria total de bicicletas (`cnt`).
- **Granularidad de trabajo elegida:** Nivel horario (`hour.csv`), justificada por responder directamente a la toma de decisiones logísticas de redistribución en tiempo real (la versión diaria `day.csv` oculta la dinámica de los turnos laborales).

### 2.3 Criterios de Éxito
- **Métrica primaria:** **RMSLE (Root Mean Squared Logarithmic Error)**
  $$\text{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^n \left( \log(y_i + 1) - \log(\hat{y}_i + 1) \right)^2}$$
  *Justificación:* Penaliza proporcionalmente errores relativos y castiga con mayor severidad subestimar la demanda frente a sobreestimarla.
- **Métricas secundarias:** 
  - **MAE (Mean Absolute Error):** Para reportar a partes no técnicas el error medio en número de bicicletas.
  - **RMSE (Root Mean Squared Error):** Sensibilidad ante valores extremos.
  - **$R^2$:** Porcentaje de varianza explicada.

### 2.4 Restricciones, Riesgos y Data Leakage
- **Fuga de datos (Data Leakage) Crítica:**
  - $cnt = casual + registered$. Las variables `casual` y `registered` **deben excluirse** de las matrices de entrada ($X$) en cualquier modelo predictivo de demanda total.
  - La variable `instant` es un índice secuencial artificial y debe descartarse.
- **Fuga temporal:** Prohibido usar K-Fold aleatorio convencional para validación.

---

## 3. Fase 2: Comprensión de los Datos (Data Understanding)

### 3.1 Ficha Técnica del Dataset
- **Fuente:** UCI Machine Learning Repository / Capital Bikeshare.
- **Archivo principal:** `data/raw/hour.csv` (17,379 filas $\times$ 17 columnas).
- **Período temporal:** 01/01/2011 al 31/12/2012.

### 3.2 Plan de Análisis Exploratorio de Datos (EDA)
Se documentarán 4 hallazgos clave obligatorios:
1. **Dinámica horaria condicional:**
   - Días laborales (`workingday = 1`): Bimodalidad clara con picos a las 08:00 y 17:00–18:00 (desplazamientos trabajo/estudio).
   - Fines de semana y feriados (`workingday = 0`): Unimodalidad con pico central entre las 12:00 y 15:00.
2. **Distribución del Target:**
   - Fuerte asimetría positiva (*right-skewed*) con presencia de valores atípicos superiores en días de clima óptimo y horas pico.
   - Justificación matemática para entrenar sobre variable transformada: $y^* = \log(cnt + 1)$.
3. **Multicolinealidad meteorológica:**
   - Correlación superior a $0.98$ entre `temp` y `atemp`. Acción: Retener `temp` y derivar variable diferencial `temp_diff = atemp - temp`.
4. **Comportamiento ante clima adverso:**
   - La categoría `weathersit = 4` (lluvia torrencial / aguanieve) tiene una frecuencia marginal ($< 0.1\%$). El modelo debe ser evaluado con cautela en este subgrupo.

---

## 4. Fase 3: Preparación de Datos (Data Preparation)

### 4.1 Estrategia de Partición (Train / Validation / Test)
- **Protocolo temporal estricto:**
  - **Train:** Datos desde enero 2011 hasta junio 2012 (aprox. 75%).
  - **Validation / Hiperparametrización:** Ventanas temporales mediante `TimeSeriesSplit` (3 a 5 folds cronológicos) dentro del conjunto de entrenamiento.
  - **Test (Hold-out final):** Julio 2012 a diciembre 2012 (aprox. 25%).
- **Justificación:** Prevenir el sesgo de anticipación (*lookahead bias*) y simular cómo operará el modelo en producción.

### 4.2 Ingeniería de Características (Feature Engineering)
Todo debe integrarse dentro de un pipeline scikit-learn (`ColumnTransformer` o clase personalizada):
1. **Codificación cíclica temporal:**
   $$\text{hr\_sin} = \sin\left(\frac{2\pi \cdot hr}{24}\right), \quad \text{hr\_cos} = \cos\left(\frac{2\pi \cdot hr}{24}\right)$$
   $$\text{mnth\_sin} = \sin\left(\frac{2\pi \cdot mnth}{12}\right), \quad \text{mnth\_cos} = \cos\left(\frac{2\pi \cdot mnth}{12}\right)$$
2. **Variables de interacción:**
   - `hr_x_workingday`: Captura la divergencia entre desplazamientos rutinarios y de ocio.
   - `temp_x_hum`: Estimación indirecta de sensación de bochorno o incomodidad térmica.
3. **Codificación categórica:**
   - `OneHotEncoder` para `season` y `weathersit` (manejando categorías infrecuentes).
4. **Escalado numérico:**
   - `StandardScaler` o `RobustScaler` (exclusivo para modelos sensibles a la escala como regresiones lineales regularizadas).

---

## 5. Fase 4: Aprendizaje No Supervisado (Obligatorio)

### 5.1 Propósito y Enfoque
- **Técnica:** **K-Means Clustering** y **PCA (Reducción de Dimensionalidad)** para interpretación de perfiles operativos.
- **Objetivo:** Descubrir agrupaciones intrínsecas de "comportamiento temporal-climático" de la ciudad sin usar la etiqueta de demanda total como discriminador.

### 5.2 Metodología
1. **Representación de entrada:** Matriz normalizada de condiciones de contexto: hora, temperatura, humedad, viento y día laboral.
2. **Selección del parámetro $k$:**
   - Curva del codo (Inercia).
   - Coeficiente de Silueta (*Silhouette Score*).
3. **Interpretación de Clusters:**
   - *Cluster 0:* Horas punta laborales en días templados (alta regularidad).
   - *Cluster 1:* Horas diurnas de fin de semana con clima favorable (ocio).
   - *Cluster 2:* Horarios nocturnos de baja demanda general.
   - *Cluster 3:* Períodos con condiciones climáticas adversas (desplome del uso casual).
4. **Utilidad analítica:** Comparar si los errores del modelo supervisado posterior se concentran desproporcionadamente en algún cluster específico.

---

## 6. Fase 5: Modelado Supervisado Clásico

### 6.1 Baseline Explícito
- **Baseline Heurístico:** Predicción por tabla de promedios históricos agrupados por `(hr, workingday)`.
- **Baseline Paramétrico:** Regresión Lineal simple sobre variables numéricas crudas.
- *Propósito:* Establecer el piso de rendimiento que cualquier modelo más sofisticado debe superar con creces.

### 6.2 Modelos Clásicos Candidatos (Mínimo 3)
1. **Modelo A (Lineal regularizado):** `Ridge` / `Lasso` (con escalado y transformaciones polinómicas básicas).
2. **Modelo B (Ensemble de Bagging):** `RandomForestRegressor`.
3. **Modelo C (Ensemble de Boosting):** `LightGBMRegressor` o `HistGradientBoostingRegressor`.

### 6.3 Ajuste de Hiperparámetros (Tuning)
- Método: `RandomizedSearchCV` o `Optuna` utilizando `TimeSeriesSplit` (evitando fugas temporales en la validación cruzada).
- Parámetros a optimizar:
  - Árboles: `n_estimators`, `max_depth`, `learning_rate`, `min_samples_split`, `subsample`.
  - Regularización: `alpha` / `l2_leaf_reg`.

---

## 7. Fase 6: AutoML como Benchmark (Obligatorio)

### 7.1 Protocolo Experimental
- **Herramienta sugerida:** **FLAML** (ligero, altamente eficiente y compatible con recursos limitados) o **AutoGluon**.
- **Restricciones del experimento:**
  - Exactamente el mismo conjunto de datos procesados (mismo `train` y `test`).
  - Misma métrica de optimización interna: `neg_root_mean_squared_log_error` o `rmse` sobre target logarítmico.
  - Límite de tiempo controlado: Presupuesto de tiempo (`time_budget = 300` segundos).

### 7.2 Análisis Comparativo de Trade-Offs
| Dimensión | Mejor Modelo Manual (ej. LightGBM) | Mejor Pipeline AutoML |
| :--- | :--- | :--- |
| **Métrica RMSLE en Test** | *A completar con evidencia* | *A completar con evidencia* |
| **Tiempo de desarrollo / cómputo** | Alto tiempo humano, bajo cómputo | Bajo tiempo humano, alto cómputo |
| **Interpretabilidad** | Alta (pesos, SHAP transparentes) | Media/Baja (posible ensamble complejo) |
| **Control del pipeline** | Control granular sobre leakage y splits | Comportamiento tipo caja negra |

---

## 8. Fase 7: Evaluación y Análisis de Residuos

### 8.1 Tabla Comparativa Unificada de Rendimiento (Test Hold-out)
| Modelo | RMSLE (Principal) | RMSE | MAE | $R^2$ | Observación |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline (Promedios hr/wday)** | -- | -- | -- | -- | Referencia ingenua |
| **Baseline (Regresión Lineal)** | -- | -- | -- | -- | Referencia lineal sin interacciones |
| **Ridge Regression** | -- | -- | -- | -- | Modelo lineal con regularización $L_2$ |
| **Random Forest** | -- | -- | -- | -- | Captura no-linealidades sin sobreajuste |
| **LightGBM** | -- | -- | -- | -- | Manejo óptimo de interacciones y splits |
| **AutoML (Mejor modelo)** | -- | -- | -- | -- | Benchmark automatizado |

### 8.2 Diagnóstico Cualitativo y Cuantitativo de Errores
- **Análisis de residuos ($y - \hat{y}$):**
  - Gráfico de residuos vs. valores predichos (verificar homocedasticidad).
  - Distribución del error en horas pico vs. horas valle.
  - Rendimiento específico en días festivos (`holiday = 1`) y clima desfavorable (`weathersit >= 3`).
- **Análisis de Importancia de Variables e Interpretabilidad (XAI):**
  - Aplicación de **SHAP Values** o *Permutation Importance* sobre el modelo final para explicar las 5 variables con mayor impacto en las predicciones.

---

## 9. Estructura de Entregables y Repositorio

### 9.1 Organización de Archivos
```text
bike-sharing-ml/
├── data/
│   ├── raw/                  # hour.csv, day.csv originales
│   └── processed/            # data_train.csv, data_test.csv
├── notebooks/
│   ├── 01_eda_comprension.ipynb
│   ├── 02_no_supervisado.ipynb
│   ├── 03_modelado_supervisado.ipynb
│   └── 04_automl_benchmark.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Lectura y split temporal
│   ├── features.py           # Pipelines scikit-learn
│   └── evaluate.py           # Funciones de cálculo de métricas y gráficos de residuos
├── models/                   # Artefactos .joblib del mejor modelo
├── reports/                  # Figuras y borrador del informe técnico (8-12 págs)
├── config/                   # Archivos de configuración o hiperparámetros
├── .gitignore
├── requirements.txt
└── README.md
```

### 9.2 Preparación para la Defensa (10 Preguntas Clave)
1. **¿Por qué este problema se resuelve con ML?** Relaciones no lineales complejas entre meteorología, dinámica laboral y estacionalidad que superan las reglas heurísticas estáticas.
2. **¿Qué hallazgo del EDA fue el más importante y qué decisión provocó?** La divergencia de patrones horarios entre días laborales y no laborales motivó la codificación de la interacción `hr * workingday`.
3. **¿Cómo evitaron el data leakage?** Excluyendo `casual`, `registered` e `instant`, e implementando división temporal estricta sin K-Fold aleatorio.
4. **¿Por qué eligieron esas métricas?** RMSLE penaliza proporcionalmente y previene subestimaciones críticas para la disponibilidad de flota; MAE otorga claridad en número de bicicletas a nivel operativo.
5. **¿Cuál fue el baseline y por qué era razonable?** El promedio histórico por hora y tipo de día, ya que representa la regla heurística actual de negocio antes de implementar ML.
6. **¿Qué modelo generalizó mejor y qué lo demuestra?** Se demuestra mediante la evaluación en el conjunto de test cronológico hold-out (segundo semestre de 2012).
7. **¿Qué tipos de errores sigue cometiendo el modelo?** Subestimación en eventos meteorológicos atípicos o festividades no registradas en el calendario común.
8. **¿Qué significado real tienen los clusters?** Representan arquetipos de demanda (movilidad obligada laboral vs. recreación de fin de semana).
9. **¿Qué encontró AutoML y en qué difiere del enfoque manual?** Comparación objetiva de si AutoML descubrió ensambles o arquitecturas con menor error que el tuning manual.
10. **¿Qué cambiarían antes de avanzar a la Parte II?** Incorporación de arquitecturas secuenciales (Deep Learning/LSTM), variables de rezago dinámicas (*lag features*) y monitoreo continuo de deriva de datos (*drift*).