# Fichas de decisión

Registro de todo lo que se prueba fuera de lo visto en clase (labs 1, 4, 5, 6 y 7), según `plan_sprints.md` §2.1.
Reglas: se compara contra la versión de referencia con **el mismo split temporal y la misma métrica** (RMSLE de validación); se registra aunque no mejore; si no mejora de forma consistente entre folds, no se adopta.

## Plantilla

```text
Tema:            (p. ej. objetivo Poisson en LightGBM)
Responsable:
Por qué se probó:
Qué se hizo:     (configuración exacta, semilla, split)
Resultado:       (tabla: referencia vs. variante, RMSLE por fold)
Decisión:        adoptar / descartar / pendiente
Fuente consultada:
```

## Convenciones de las fichas de Joan (Sprint 2)
- Todo en `notebooks/03_modelado_supervisado.ipynb` (Parte II), validación con `get_cv()` (5 folds, `test_size=1460`, ventana expansiva) y solo train. LightGBM con `random_state=42`, `deterministic=True`.
- **Regla de adopción** (propuesta de Joan, en `CONFIG["regla_adopcion"]`, ajustable por el grupo): mejora del RMSLE medio ≥ 0.002 **y** peor en ≤ 1 de los 5 folds (un empate no cuenta como peor).
- Los folds 1 a 3 solo tienen 2011 en train: no pueden evaluar `yr` (ver ficha 1).

---

## Ficha 1 — Variante de tendencia (`yr` / `sin_yr` / `trend`)
```text
Tema:            Variable de tendencia en LightGBM
Responsable:     Joan
Por qué se probó: §4.2 punto 6 y §12.2: el crecimiento interanual puede no extrapolarse (los árboles no extrapolan).
Qué se hizo:     LightGBM base (n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=20), objetivo log1p+MSE,
                 build_preprocessor(tendencia=...) y el resto provisional. Split: get_cv(), solo train.
Resultado:       RMSLE por fold (f1..f5) y media ± std
                 yr (referencia)  0.3951 0.4468 0.7938 0.4103 0.2755 | 0.4643 ± 0.1951
                 sin_yr           0.3951 0.4468 0.7938 0.6482 0.4009 | 0.5370 ± 0.1769  (peor en 2 folds; sin efecto en f1-f3)
                 trend            0.4577 0.4838 0.5344 0.4684 0.2753 | 0.4439 ± 0.0987  (mejor media, peor en 3 folds)
Decisión:        DESCARTAR sin_yr y trend; mantener yr. sin_yr empeora claramente cuando el train ya incluye 2012.
                 trend baja la media por el fold 3 pero empeora los folds 1, 2 y 4: no es consistente.
                 🔶 Hipótesis: la ganancia de trend en f3 es extrapolación de la tendencia intra-2011; a revisar si el test muestra sesgo.
Fuente consultada: plan §4.2 y §12.2. Sin fuente externa.
```

## Ficha 2 — Tratamiento de `windspeed = 0`
```text
Tema:            windspeed = 0: dejar / indicador / mediana
Responsable:     Joan
Por qué se probó: §4.3; el EDA mostró un salto 0 -> 0.0896 y frecuencia dependiente de hora y mes.
Qué se hizo:     Misma base que la ficha 1; build_preprocessor(windspeed_cero=...).
Resultado:       dejar (ref)  0.3951 0.4468 0.7938 0.4103 0.2755 | 0.4643 ± 0.1951
                 indicador    0.3951 0.4468 0.7938 0.4103 0.2755 | 0.4643 ± 0.1951  (idéntico: un árbol ya separa windspeed <= 0)
                 mediana      0.3922 0.4513 0.7940 0.4086 0.2778 | 0.4648 ± 0.1949  (mejor en 2 folds, peor en 3)
Decisión:        DESCARTAR indicador y mediana; mantener dejar. Sin efecto medible en LightGBM.
                 🔶 No dice nada sobre modelos lineales (Pardo).
Fuente consultada: ninguna externa.
```

## Ficha 3 — Tratamiento de `hum = 0`
```text
Tema:            hum = 0: mediana (provisional) vs dejar
Responsable:     Joan
Por qué se probó: §4.3; 22 filas, todas el 2011-03-10.
Qué se hizo:     Misma base; build_preprocessor(hum_cero=...).
Resultado:       mediana (ref)  0.3951 0.4468 0.7938 0.4103 0.2755 | 0.4643 ± 0.1951
                 dejar          0.3892 0.4464 0.7849 0.4155 0.2760 | 0.4624 ± 0.1914  (mejora 0.0019 < 0.002; peor en 2 folds)
Decisión:        DESCARTAR el cambio; mantener mediana. Efecto mínimo y no consistente.
Fuente consultada: ninguna externa.
```

## Ficha 4 — Tuning de LightGBM (`RandomizedSearchCV` + `TimeSeriesSplit`)
```text
Tema:            Hiperparámetros de LightGBM
Responsable:     Joan
Por qué se probó: §6.4: rangos iniciales del plan, "repetir con varias semillas y reportar la variación entre folds".
Qué se hizo:     RandomizedSearchCV con TransformedTargetRegressor(log1p/expm1), scoring = RMSLE de src.evaluate, cv=get_cv(),
                 n_iter=40, random_state=42, n_jobs=1. Rangos: num_leaves [31,127], min_child_samples [20,100],
                 learning_rate [0.03,0.08], n_estimators [300,1200], colsample_bytree [0.7,0.9], reg_lambda [0.1,10] (log).
                 Sin early stopping (n_estimators fijo como hiperparámetro). Tiempo de búsqueda: ~300-350 s.
Resultado:       Referencia 0.4643 ± 0.1951. Mejor: n_estimators=524, learning_rate=0.0462, num_leaves=31, min_child_samples=81,
                 colsample_bytree=0.8767, reg_lambda=0.2857 -> f1..f5 = 0.3709 0.4394 0.7797 0.4090 0.2754 | 0.4549 ± 0.1918
                 (mejor en 5 folds). 3 semillas (42, 7, 123): 0.4549, 0.4552, 0.4551; desviación entre semillas 0.0002.
                 Los 8 mejores candidatos están entre 0.4549 y 0.4587 (superficie plana).
Decisión:        ADOPTAR (cumple la regla). Puntaje optimista (mejor de 40 sobre los mismos folds). Valores en CONFIG["lgbm"]["params_tuneados"].
Fuente consultada: plan §6.4 (rangos). Sin fuente externa.
```

## Ficha 5 — Objetivo Poisson vs `log1p` + MSE
```text
Tema:            objective='poisson' sobre cnt vs log1p(cnt) con MSE
Responsable:     Joan
Por qué se probó: §6.3 y §12.2 punto 4; Poisson es natural para conteos y se compara en RMSLE y MAE.
Qué se hizo:     Hiperparámetros del tuning (ficha 4) en ambos; Poisson entrenado sobre cnt, log_target=False. Mismos folds.
Resultado:       RMSLE f1..f5 | media ± std | MAE medio
                 log1p+MSE  0.3709 0.4394 0.7797 0.4090 0.2754 | 0.4549 ± 0.1918 | 47.21
                 Poisson    0.3682 0.4659 0.7094 0.4063 0.2870 | 0.4473 ± 0.1602 | 46.69
                 Mejor en 3 folds, peor en 2; en f4-f5 efecto medio -0.0044. La ganancia viene del fold 3.
Decisión:        DESCARTAR (no cumple la regla: peor en 2 folds). Los hiperparámetros se tunearon para log1p, lo que puede
                 favorecerlo. 🔶 Merecería una prueba con tuning propio si el grupo quiere insistir.
Fuente consultada: plan §6.3. Sin fuente externa.
```

## Ficha 6 — Factor de *smearing* de Duan al revertir el log
```text
Tema:            Corrección de sesgo de expm1(log1p(y)): ŷ = exp(p)·φ − 1, φ = media(exp(residuo))
Responsable:     Joan
Por qué se probó: §6.3 y §12.1: medir el sesgo antes de corregir.
Qué se hizo:     Mejor configuración del tuning, objetivo log1p; φ calculado con residuos de train de cada fold (en muestra).
Resultado:       φ por fold: 1.0240 1.0261 1.0290 1.0305 1.0309 (media 1.028).
                 Sesgo medio (ŷ − y): -30.5 sin φ; -26.0 con φ.  RMSLE 0.4549 -> 0.4466; MAE 47.21 -> 45.93; RMSE 70.60 -> 69.11.
                 RMSLE empeora en f2 (0.4394 -> 0.4458) y f5 (0.2754 -> 0.2755).
Decisión:        NO ADOPTAR (peor en 2 folds). El sesgo mezcla la reversión del log con la subestimación por tendencia (f3, f4).
                 ⏳ φ en muestra es optimista; reevaluar en el Sprint 4 solo si interesa MAE o RMSE.
Fuente consultada: plan §6.3 y §12.1. Duan (1983), "The smearing estimate", JASA: citado por el plan, no consultado directamente (⏳).
```

## Ficha 7 — Lags de 24 h y 168 h (extra §6.4)
```text
Tema:            lag_24 y lag_168 de cnt
Responsable:     Joan
Por qué se probó: §6.4 extra 3 y §2.2: con el horizonte "día siguiente" ambos están disponibles al predecir el día D.
Qué se hizo:     agregar_lags(df) sobre la serie completa reindexada (solo mira el pasado); build_preprocessor(lags=...);
                 LightGBM con los parámetros de la ficha 4; get_cv(gap=24) (§4.1) y la referencia recalculada con ese mismo gap.
Resultado:       RMSLE f1..f5 | media ± std | folds mejor/peor
                 sin lags (ref) 0.3821 0.4395 0.7786 0.4061 0.2747 | 0.4562 ± 0.1905 |
                 lag_24         0.3818 0.4106 0.6339 0.3837 0.2734 | 0.4167 ± 0.1323 | 5/0
                 lag_168        0.3796 0.4554 0.5969 0.3565 0.2833 | 0.4143 ± 0.1191 | 3/2
                 lag_24+lag_168 0.3751 0.4374 0.5044 0.3572 0.2782 | 0.3905 ± 0.0853 | 4/1
Decisión:        ADOPTAR lag_24 + lag_168 (cumple la regla). Válido solo con el horizonte "día siguiente"; depende de que el día
                 anterior esté completo (lags NaN: 132 y 265 filas en train). ⏳ Decisión del grupo: si los demás modelos
                 también los usan, para que la tabla comparativa sea justa.
Fuente consultada: plan §2.2, §4.1, §6.4. Sin fuente externa.
```

## Convenciones de las fichas de Pardo (Sprint 2)
- Ridge/Lasso en `notebooks/03b_ridge.ipynb`, con `get_cv()` y `validar_temporal` (solo train), objetivo `log1p`, misma regla de adopción.
- Clustering en `notebooks/02_no_supervisado.ipynb`, ajustado solo con train, `random_state=42`.

## Ficha 8 — Codificación de la hora en Ridge (splines periódicos vs One-Hot vs sin/cos)
```text
Tema:            Codificación de la hora para modelos lineales
Responsable:     Pardo
Por qué se probó: plan_sprints.md §5; un lineal no puede representar dos picos con la hora entera.
Qué se hizo:     PrepLineal (src/lineales.py) sobre build_preprocessor(escalar=True) sin modificarlo; Ridge(alpha=1).
                 Splines: SplineTransformer(degree=3, 13 nudos en [0,24], extrapolation='periodic'). Con y sin hr_workingday.
Resultado:       Sin hr_workingday: entero 1.0992 | sin_cos 0.9475 | One-Hot 0.6786 | splines 0.6800.
                 Con hr_workingday (contrato): las cuatro entre 0.4930 y 0.4932 (sin/cos, One-Hot y splines peor en 2/5 folds).
                 Splines × workingday sin hr_workingday: k=7 0.5791, k=9 0.5404, k=13 0.4966, k=25 0.4932.
Decisión:        DESCARTAR el cambio; mantener hora='entero'. La interacción hora × workingday del contrato ya captura el perfil.
Fuente consultada: scikit-learn, "Time-related feature engineering" (galería de ejemplos); docs de SplineTransformer.
```

## Ficha 9 — Interacción hora × temperatura en Ridge
```text
Tema:            Splines de la hora multiplicados por temp_c (escalada)
Responsable:     Pardo
Por qué se probó: el efecto de la temperatura depende de la hora; un lineal no lo aprende sin interacción.
Qué se hizo:     PrepLineal(codif_hora='splines', interaccion_temp=True) + hr_workingday; Ridge(alpha=1). 90 columnas.
Resultado:       contrato (ref)  0.3831 0.5318 0.7578 0.4264 0.3661 | 0.4930 ± 0.1614
                 splines × temp  0.3821 0.5243 0.7489 0.4235 0.3638 | 0.4885 ± 0.1583  (mejor en 5/5 folds)
                 One-Hot × temp  0.4885 (igual, 24 columnas más).
                 Ridge alpha 0.1-3: 0.4885-0.4886; Lasso alpha=1e-4: 0.4884 (peor en 1 fold, mejora 0.0001 < 0.002).
Decisión:        ADOPTAR para Ridge (cumple la regla). Ridge(alpha=1). Lasso no se adopta.
Fuente consultada: scikit-learn, "Time-related feature engineering" (sección de interacciones).
```

## Ficha 10 — Lags de 24 h y 168 h en Ridge
```text
Tema:            lag_24 y lag_168 en el Ridge final
Responsable:     Pardo
Por qué se probó: decisión 8a del grupo (¿lags para todos?); mismo protocolo que la ficha 7.
Qué se hizo:     agregar_lags sobre la serie completa; NaN de los lags imputados con la mediana de train del fold
                 (dentro de PrepLineal); get_cv(gap=24); referencia recalculada con gap=24.
Resultado:       sin lags (ref)  0.3818 0.5243 0.7487 0.4246 0.3650 | 0.4889 ± 0.1579
                 con lags        0.3810 0.5031 0.6958 0.4338 0.3693 | 0.4766 ± 0.1335  (mejor en 3, peor en 2)
Decisión:        NO ADOPTAR con la regla actual (peor en 2 folds). Si el grupo decide lags para todos, Ridge mejora en media.
Fuente consultada: plan §2.2, §4.1. Sin fuente externa.
```

## Ficha 11 — Estabilidad de clusters (bootstrap y temporal)
```text
Tema:            Elección de k con estabilidad cuando los índices internos no deciden
Responsable:     Pardo
Por qué se probó: silueta 0.17-0.20 para todo k y BIC del GMM sin mínimo: los índices no eligen k.
Qué se hizo:     K-Means (n_init=10) sobre las 6 entradas de §5.1 escaladas con train. ARI contra el modelo base con:
                 semillas 1/7/123, 10 remuestreos bootstrap, y ajuste solo con 2011 asignando todo el train.
Resultado:       k=2: semillas 0.990, bootstrap 0.968 (mín 0.937), 2011→todo 0.876
                 k=4: semillas 0.997, bootstrap 0.950 (mín 0.864), 2011→todo 0.955
                 k=6: semillas 0.981, bootstrap 0.936 (mín 0.874), 2011→todo 0.805
Decisión:        ADOPTAR k=4 (más estable en el tiempo e interpretable). Se reporta que la estructura es débil.
Fuente consultada: docs de scikit-learn (adjusted_rand_score, silhouette_score); labs 5 y 6.
```

## Convenciones de las fichas de José (Sprint 2)
- Random Forest en `notebooks/03c_random_forest.ipynb` y AutoML en `notebooks/04_automl_benchmark.ipynb`, con `get_cv()` y `validar_temporal` (solo train), objetivo `log1p`, misma regla de adopción.

## Ficha 12 — Tuning de Random Forest (`max_features`, `min_samples_leaf`, `max_depth`)
```text
Tema:            Hiperparámetros de Random Forest
Responsable:     José
Por qué se probó: plan_sprints.md §5; el RF por defecto es el punto de partida del modelo de bagging.
Qué se hizo:     ParameterSampler (10 candidatos, random_state=42) sobre max_features [0.3, 0.5, 0.7, 1.0], min_samples_leaf [1, 2, 5, 10, 20],
                 max_depth [None, 15, 25]; cada candidato con n_estimators=150 y validar_temporal (get_cv(), 5 folds). El mejor se midió con 300 árboles.
Resultado:       RF base (300 árboles, defaults)   f1..f5 = 0.3946 0.4878 0.7650 0.4825 0.3133 | 0.4887 ± 0.1703
                 RF tuneado (max_features=0.5, min_samples_leaf=1, max_depth=25)
                                                    f1..f5 = 0.3780 0.4544 0.7630 0.5113 0.3063 | 0.4826 ± 0.1748
                 Mejora media 0.0060; mejor en 4 folds y peor en 1 (fold 4). Hojas grandes (min_samples_leaf >= 10) empeoran (0.50-0.51).
Decisión:        ADOPTAR (cumple la regla). Puntaje optimista (mejor de 10 sobre los mismos folds). RF queda 13.6 % mejor que el baseline (0.5585)
                 y 0.0277 peor que LightGBM con tuning (0.4549); equivale a Ridge (0.4885).
Fuente consultada: documentación de scikit-learn (RandomForestRegressor). Sin fuente externa adicional.
```

## Ficha 13 — Lags de 24 h y 168 h en Random Forest
```text
Tema:            lag_24 y lag_168 de cnt en Random Forest (insumo para la decisión 8a del grupo)
Responsable:     José
Por qué se probó: ficha 7 de Joan (lags con LightGBM) y ficha 10 de Pardo (lags con Ridge); la tabla es justa solo si se reportan las dos vías.
Qué se hizo:     agregar_lags(df) sobre la serie completa; build_preprocessor(lags=...); RF con los parámetros de la ficha 12;
                 get_cv(gap=24) y referencia recalculada con ese mismo gap. Los NaN de los lags los admite RandomForestRegressor.
Resultado:       RMSLE f1..f5 | media ± std | folds mejor/peor
                 sin lags (ref, gap=24) 0.3800 0.4571 0.7638 0.5075 0.3056 | 0.4828 ± 0.1748 |
                 lag_24 + lag_168       0.4176 0.4690 0.4595 0.3967 0.3195 | 0.4125 ± 0.0598 | 2/3
Decisión:        DESCARTAR según la regla (peor en 3 folds), aunque la media mejora 0.0703. La ganancia está concentrada en los folds 3 y 4
                 (donde el train solo cubre 2011 y los árboles no extrapolan la tendencia; los lags aportan el nivel reciente) y empeora en los folds 1, 2 y 5.
                 🔶 Hipótesis: los lags actúan como corrector de tendencia más que como señal horaria fina.
Fuente consultada: sin fuente externa.
```

## Ficha 14 — AutoML con FLAML como benchmark
```text
Tema:            Herramienta AutoML y comparación con el mejor modelo manual
Responsable:     José
Por qué se probó: guía EC2 §3.6 (AutoML obligatorio) y plan §7. Se eligió FLAML porque corre en Python 3.13 sin Java (PyCaret 3.3 exige <= 3.11,
                 H2O exige Java) y acepta un objeto splitter en split_type: se le pasa exactamente get_cv().
Qué se hizo:     FLAML 2.7.0, task=regression, metric=rmse sobre log1p(cnt) (equivale a RMSLE), eval_method=cv, split_type=get_cv() (gap=0) o get_cv(gap=24) con lags,
                 estimator_list=[lgbm, rf, extra_tree], time_budget=300 s, seed=42, n_jobs=-1. El mejor estimador se re-evaluó con validar_temporal.
                 Desviación controlada: el preprocesador se ajustó una vez con todo train para la búsqueda (FLAML recibe una matriz); la re-evaluación sí lo ajusta por fold.
Resultado:       RMSLE f1..f5 | media ± std
                 Manual: LightGBM + tuning (gap=0)  0.3709 0.4394 0.7797 0.4090 0.2754 | 0.4549 ± 0.1918
                 AutoML sin lags (lgbm, gap=0)      0.3732 0.4366 0.7702 0.3599 0.2623 | 0.4405 ± 0.1946  (mejor en 4/5; diferencia 0.0144)
                 Manual: LightGBM + lags (gap=24)   0.3751 0.4374 0.5044 0.3572 0.2782 | 0.3905 ± 0.0853
                 AutoML con lags (lgbm, gap=24)     0.3943 0.4234 0.4736 0.3581 0.2885 | 0.3876 ± 0.0697  (mejor en 2/5; diferencia 0.0029)
                 Pérdida interna de FLAML vs validar_temporal: 0.4407 vs 0.4405 (sin lags), 0.3876 vs 0.3876 (con lags). 12 y 12 configuraciones en 300 s (1 CPU).
                 Configuración ganadora sin lags: num_leaves=4, learning_rate=0.0799, n_estimators=2027, min_child_samples=18, reg_alpha=2.81 (árboles muy pequeños).
                 Configuración ganadora con lags: num_leaves=55, learning_rate=0.0602, n_estimators=185, min_child_samples=37, colsample_bytree=0.42.
Decisión:        ADOPTAR FLAML como benchmark. Sin lags AutoML supera al mejor manual (0.4405 vs 0.4549); con lags empatan (0.3876 vs 0.3905).
                 🔶 El tuning manual de Joan exploró num_leaves en [31, 127]; AutoML encontró una región (árboles pequeños) fuera de ese rango.
                 El puntaje de AutoML también es optimista (la búsqueda ve los mismos folds). No cambia el modelo final por sí solo: la elección es del Sprint 3.
Fuente consultada: documentación y código de FLAML (decide_split_type acepta splitter con split y get_n_splits). Sin fuente externa adicional.
Corrección (Joan, Sprint 3): las cifras de esta ficha se actualizaron para que coincidan con la salida ejecutada de
                 notebooks/04_automl_benchmark.ipynb, reports/jose/automl_comparacion.csv y reports/jose/automl_config.json.
                 Antes decía: RMSLE 0.4408 ± 0.1879 (sin lags) y 0.3881 ± 0.0803 (con lags); learning_rate=0.594, n_estimators=177,
                 num_leaves=4, min_child_samples=54, reg_alpha=6.4; 14 y 13 configuraciones; pérdidas internas 0.4404/0.4408 y
                 0.3873/0.3881; diferencias 0.0140 y 0.0024; RMSLE por fold distintos. Esos valores parecen venir de una ejecución
                 anterior de FLAML (su resultado depende del tiempo de cómputo). La decisión de José (adoptar FLAML como benchmark) no cambia.
```

## Convenciones de las fichas de Joan (Sprint 3)
- Todo en `notebooks/05_comparacion_y_errores.ipynb`, solo train y `get_cv()` (sin lags `gap=0`; con lags `gap=24`). Ningún modelo se reentrena con todo train: se ajustan dentro de cada fold y las predicciones analizadas son las de la ventana de validación (fuera de muestra). El test no se carga.

## Ficha 15 — Reconstrucción de AutoML a partir de su configuración ganadora
```text
Tema:            Reproducir el modelo ganador de FLAML sin repetir la búsqueda
Responsable:     Joan
Por qué se probó: la tabla comparativa y el análisis de residuos necesitan las predicciones por fold de AutoML, que el CSV de José no tiene;
                 la búsqueda de FLAML depende del tiempo de cómputo (las dos versiones de la ficha 14 lo muestran), así que repetirla no es reproducible.
Qué se hizo:     Se tomó mejor_config de reports/jose/automl_config.json y se construyó LGBMRegressor(max_bin=2**log_max_bin - 1, n_jobs=-1, ...)
                 dentro del mismo Pipeline (build_preprocessor + modelo). Validación con get_cv() (gap=0) y get_cv(gap=24) con lags.
                 Comprobación previa: con flaml.AutoML(starting_points, max_iter=1) el estimador resultante tiene los mismos parámetros.
Resultado:       RMSLE medio sin lags 0.4405 (JSON: 0.440451...), con lags 0.3876 (JSON: 0.387568...); coincide a 6 decimales y los 5 RMSLE por fold
                 coinciden con automl_comparacion.csv (tolerancia 6e-5). Es el mismo modelo que evaluó José.
Decisión:        ADOPTAR el método para el análisis del Sprint 3. No se modifica el notebook 04 ni los resultados de José.
Fuente consultada: código y documentación de FLAML (LGBMEstimator); automl_config.json. Se detectó que requirements.txt necesitaba flaml[automl].
```

## Ficha 16 — Comparación fold a fold y bootstrap por bloques de días entre los finalistas
```text
Tema:            ¿Se distinguen LightGBM manual y AutoML (FLAML)?
Responsable:     Joan
Por qué se probó: la diferencia entre modelos (~0.03) es menor que la desviación entre folds (0.06-0.19); §8.2 sugiere un bootstrap en bloques de días.
Qué se hizo:     Diferencia de RMSLE por fold (LightGBM - AutoML) y bootstrap de DÍAS completos sobre las 7 300 horas de validación
                 (2 000 réplicas, semilla 42; métrica = RMSLE agrupado, no media de folds).
Resultado:       Sin lags: diferencia por fold -0.0023 0.0028 0.0094 0.0491 0.0131 (AutoML mejor en 4/5); agrupada +0.0125, IC95% [0.0077, 0.0175].
                 Con lags: -0.0192 0.0140 0.0308 -0.0009 -0.0103 (AutoML mejor en 2/5); media +0.0029; agrupada +0.0053, IC95% [-0.0005, 0.0105];
                 LightGBM sale mejor en 3.5 % de las réplicas.
Decisión:        Sin lags AutoML es mejor de forma consistente; con lags es un empate práctico. 🔶 Los IC solo miden la variación entre días dentro
                 de las mismas 5 ventanas (no su heterogeneidad): son optimistas.
Fuente consultada: plan §8.2 (bootstrap en bloques de días). Sin fuente externa.
```

## Ficha 17 — Errores por cluster y por corte (hora, día, feriado, clima, mes)
```text
Tema:            Dónde falla el modelo (residuos fuera de muestra)
Responsable:     Joan
Por qué se probó: §8.2, plan_sprints.md §6 (residuos, sesgo y errores por cluster).
Qué se hizo:     Residuo e = log1p(y) - log1p(pred) sobre las 7 300 horas de validación de cada modelo. Cortes: valor predicho, mes, hora, día de la semana,
                 holiday, weathersit >= 3, workingday y cluster. Clusters de Pardo regenerados con src/clustering.py (K-Means k=4, ajustados con todo train,
                 sin target): tamaños 3495/3133/3373/3002 = los del notebook 02.
Resultado:       Sesgo medio +0.110 (LightGBM + lags), +0.103 (AutoML + lags), +0.188 (sin lags): subestiman. Concentrado en ene-mar de 2012
                 (+0.405, +0.299, +0.161; sin lags +0.766 y +0.587 en ene-feb). Dispersión del residuo 0.496 (poca demanda) -> 0.189 (mucha): no es homocedástico.
                 RMSLE: clima adverso 0.602 vs 0.370; feriado 0.582 vs 0.390; hora 2:00 0.533, mediodía-tarde 0.326-0.343; no laborable 0.442 vs 0.375.
                 Por cluster (LightGBM + lags): 0.282 / 0.426 / 0.464 / 0.392; los lags bajan el error en los 4.
Decisión:        DOCUMENTAR (no hay un cambio de modelo). Los clusters son hipótesis (silueta 0.18) y se usan solo para describir el error.
                 ⚠️ Mes y fold están confundidos; ningún mes de validación es julio ni agosto; las ventanas participaron en el ajuste del clustering (sin target).
Fuente consultada: plan §5 y §8.2; resumen de Pardo.
```

## Ficha 18 — Interpretabilidad: permutación por grupos y SHAP
```text
Tema:            Qué variables usa LightGBM con lags (y AutoML)
Responsable:     Joan
Por qué se probó: §8.3; SHAP y permutación no son parte de los labs 1, 4, 5, 6 y 7.
Qué se hizo:     Modelos ajustados dentro de los folds 3, 4 y 5 (vía con lags) y explicados sobre la ventana de validación de ese fold.
                 Permutación de columnas crudas por GRUPOS (temp+atemp, lags, tipo de día, ...) con 5 repeticiones, semilla 42, midiendo el aumento de RMSLE.
                 SHAP TreeExplainer (log1p), columnas agrupadas por origen; aditividad comprobada. Variable de control: dteday.
Resultado:       Hora (hr + hr x workingday) y lags dominan: SHAP hr ~23-25 %, lag_168 22-28 %, lag_24 ~8 %, hr x workingday 12-14 % del |SHAP| total.
                 Permutación: hr 0.65-0.81, lags 0.45-0.73. lag_168 pesa más que lag_24 en LightGBM; AutoML usa más lag_24. dteday = 0 (control).
Decisión:        DOCUMENTAR. 🔶 La permutación de yr da 0 porque yr es constante dentro de cada ventana de validación (limitación del método); SHAP sí lo detecta (~6 % en folds 4-5).
                 Salvedad: variables correlacionadas (temp/atemp, lag_24/lag_168) reparten importancia; no son efectos causales.
Fuente consultada: documentación de SHAP (TreeExplainer) y de scikit-learn (permutation importance, inspección de modelos). Sin fuente externa adicional.
```

## Ficha 19 — Propuesta de modelo final (pendiente de decisión de Joan)
```text
Tema:            Elección del modelo final entre LightGBM manual y AutoML (FLAML), con lags
Responsable:     Joan (propuesta del Sprint 3)
Por qué se probó: plan_sprints.md §6: elegir con argumentos, no solo por la métrica.
Qué se hizo:     Tabla de 12 criterios con cifras del notebook 05 (rendimiento, estabilidad, peor fold, folds ganados, IC por bootstrap, complejidad,
                 tiempo, interpretabilidad, reproducibilidad, dependencias). Se descartan baseline, Ridge y Random Forest por rendimiento.
Resultado:       Con lags: LightGBM manual 0.3905 ± 0.0853 (peor fold 0.5044; gana 3/5 folds; 524 x 31 hojas); AutoML 0.3876 ± 0.0697 (peor fold 0.4736; 185 x 55 hojas).
                 Sin lags: AutoML 0.4405 vs LightGBM 0.4549 (AutoML mejor en 4/5, IC95% [0.0077, 0.0175]).
Decisión:        PENDIENTE. Propuesta: LightGBM con lag_24 + lag_168 y la configuración encontrada por AutoML; el LightGBM manual queda como alternativa
                 igualmente válida (empate práctico). Si el grupo no usa lags: AutoML sin lags. El modelo final NO se reentrena con todo train hasta confirmar.
Fuente consultada: notebook 05, fichas 4, 7, 13, 14, 15 y 16.
```

## Ficha 20 — Horizonte de pronóstico y lags (día siguiente vs semana siguiente; `lag_336`)
```text
Tema:            Qué lags existen según el horizonte y cuánto rinde cada escenario
Responsable:     Joan
Por qué se probó: pregunta de negocio (¿día o semana?) previa a elegir el modelo final: lag_24 solo existe a <= 24 h; lag_168 sirve para toda la semana siguiente.
Qué se hizo:     notebooks/06_horizonte_y_lags.ipynb. Mismos hiperparámetros ya registrados (LightGBM de la ficha 4; AutoML de la ficha 14 con su config sin lags
                 y con lags, sin re-tuning), get_cv(gap=24), solo train. Escenarios: sin lags; solo lag_168; lag_168 + lag_336 (semana); solo lag_24;
                 lag_24 + lag_168 emitido a las 00:00 (día); idem emitido a las 18:00 (lag_24 = NaN en horas 18-23, entrenado así); y entrenado con lags
                 completos pero usado sin lag_24 (caso de falla). lag_336 = misma hora de hace dos semanas (nuevo).
Resultado:       RMSLE medio (LightGBM manual | AutoML config sin lags | AutoML config con lags):
                 sin lags 0.4562 | 0.4417 | 0.4491;  lag_168 0.4143 | 0.4075 | 0.4154;  lag_168+lag_336 0.3893 | 0.3863 | 0.3918;
                 lag_24 0.4167 | 0.4091 | 0.4126;  día 00:00 0.3905 | 0.3855 | 0.3876;  día 18:00 0.4015 | 0.3975 | 0.4014;
                 falla (sin lag_24) 0.4131 | 0.4173 | 0.4460.
                 Regla del equipo vs sin lags: la cumplen solo lag_24 (LightGBM y AutoML con lags) y lag_24+lag_168 (LightGBM); los esquemas semanales mejoran la media
                 (LightGBM 0.0669, 14.7 %) pero empeoran 3 folds (entre 0.009 y 0.014) y no la cumplen.
Decisión:        PENDIENTE (decide Joan/grupo). Evidencia: día (00:00) y semana (lag_168 + lag_336) empatan en precisión, así que los lags no obligan a un horizonte diario.
                 Falta medir el efecto de un pronóstico de clima a una semana (§8.4). La config de AutoML con lags es la más frágil si falta lag_24 (0.4460).
                 ⚠️ 6 escenarios x 3 configuraciones sobre los mismos folds: el mejor escenario está algo sobreestimado. ⏳ lag_24 + lag_168 + lag_336 no se probó.
Fuente consultada: plan §2.2, §4.1, §6.4, §8.4. Sin fuente externa.
```

## Ficha 21 — Tres lags (24, 168 y 336) y selección de horizonte y modelo
```text
Tema:            lag_24 + lag_168 + lag_336 y elección final entre día/semana y entre LightGBM manual/AutoML
Responsable:     Joan
Por qué se probó: ficha 20 dejó pendiente la combinación de los tres lags; la decisión de horizonte y modelo final depende de ella.
Qué se hizo:     notebooks/06_horizonte_y_lags.ipynb, secciones 2 a 5. Mismos hiperparámetros (ficha 4; ficha 14 con sus dos configuraciones), get_cv(gap=24), solo train.
                 Escenarios con los tres lags: emitido a las 00:00, emitido a las 18:00 (lag_24 = NaN en horas 18-23) y entrenado con los tres lags pero usado sin lag_24.
                 Bootstrap por días (2 000 réplicas, semilla 42) sobre las predicciones fuera de muestra de tres candidatos.
Resultado:       RMSLE medio (LightGBM manual | AutoML config sin lags | AutoML config con lags):
                 3 lags, 00:00: 0.3732 ± 0.0566 | 0.3738 ± 0.0673 | 0.3766 ± 0.0604;  3 lags, 18:00: 0.3823 | 0.3826 | 0.3838;
                 falla de lag_24 (entrenado con 3 lags): 0.3862 | 0.3985 | 0.4077.
                 LightGBM manual, 3 lags, por fold: 0.3902 0.4311 0.4117 0.3412 0.2916 (frente a sin lags 0.3821 0.4395 0.7786 0.4061 0.2747: peor en f1 y f5).
                 Bootstrap: A (LightGBM, 3 lags, día) - B (AutoML, 3 lags, día) = -0.0020, IC95% [-0.0081, 0.0039];
                 A - C (LightGBM, lag_168 + lag_336, semana) = -0.0188, IC95% [-0.0243, -0.0133]. Día mejor que semana en 3 de 5 folds (media 0.0161).
Decisión:        PROPUESTA (decide Joan): horizonte día siguiente (emitido a las 00:00) y LightGBM manual con lag_24, lag_168 y lag_336.
                 Formalmente NO cumple la regla de adopción (peor en 2 folds: f1 +0.008, f5 +0.017) aunque mejora la media en 0.0831 (18.2 %); la ganancia viene del fold 3.
                 Si se exige la regla estricta: LightGBM con lag_24 + lag_168 (0.3905). Alternativa semanal: lag_168 + lag_336 (0.3893).
                 ⚠️ Clima observado (no pronosticado); hiperparámetros tuneados sin lags; varios escenarios sobre los mismos folds (optimismo).
Fuente consultada: plan §2.2, §4.1, §6.4, §8.2 y §8.4. Sin fuente externa.
```

## Convenciones de las fichas de Joan (Sprint 4)
- El modelo final se fijó en el Sprint 3 (fichas 20 y 21) **antes de ver el test**. Se reentrena una vez con todo train y se evalúa una vez (notebook 07); la validación operativa (notebook 08) usa esas mismas predicciones, sin reajustar nada.

## Ficha 22 — Evaluación final en test y política de no cambiar el modelo después de verlo
```text
Tema:            Evaluación única en el hold-out jul-dic 2012
Responsable:     Joan
Por qué se probó: §4.1 y §8.1: una sola evaluación del modelo final; tabla comparativa de referencia.
Qué se hizo:     notebooks/07_evaluacion_final_test.ipynb. LightGBM (params de la ficha 4) + lag_24, lag_168, lag_336, objetivo log1p, reentrenado con las 13 003 horas
                 de train y evaluado sobre las 4 376 horas con registro del test. Predicciones guardadas (reports/joan/test_predicciones.csv). Los otros modelos
                 (baseline, Ridge, Random Forest, LightGBM sin lags, AutoML) se evaluaron como referencia; no se usaron para decidir.
Resultado:       FINAL en test: RMSLE 0.3505 | RMSE 69.99 | MAE 43.11 | R2 0.8985 (validación 0.3732 ± 0.0566). Baseline 0.5999 (el final lo mejora 41.6 %).
                 Referencias: LightGBM sin lags 0.3463, Ridge 0.3857/0.3802, Random Forest 0.3812/0.3795, AutoML (config sin lags) 0.3271/0.3389 (sin lags / con 3 lags).
                 Los lags NO mejoraron a LightGBM en el test: peor en jul-oct (0.269 -> 0.333 en julio), mejor en nov-dic (0.492 -> 0.466 en diciembre).
                 RMSLE mensual del final: 0.333 0.243 0.290 0.308 0.412 0.466 (jul-dic). Peores cortes: clima adverso 0.654 (sesgo +0.182), feriados 0.664 (sobreestima),
                 madrugada (2:00 = 0.520). Sandy: el 30-oct predice 1.46 veces lo real.
Decisión:        MANTENER el modelo elegido en el Sprint 3. AutoML sin lags fue el mejor en el test (0.3271, 6.7 % mejor), pero cambiar de modelo con el test
                 dejaría la cifra sin estimación independiente; si el grupo lo decide, debe documentarse que la elección usó el test.
                 🔶 Los lags se justificaron por el cambio de nivel de ene-mar de 2012; en un periodo estable de verano no aportan. Ningún fold de validación cubría jul-ago (riesgo ya advertido).
Fuente consultada: plan §4.1, §8.1 y §8.2. Sin fuente externa.
```

## Ficha 23 — Validación operativa (§8.4)
```text
Tema:            ¿Funcionaría en un caso real?
Responsable:     Joan
Por qué se probó: §8.4 puntos 1 a 6.
Qué se hizo:     notebooks/08_validacion_operativa.ipynb y src/predict.py. Modelo fijo cargado desde models/lgbm_final.joblib. Ruido en el clima (5 niveles, 20 réplicas, semilla 42+réplica,
                 independiente por hora y correlacionado por día), backtest día a día (más un reentrenamiento mensual expansivo como extra), métrica operativa (umbral 20 %, propuesta de Joan),
                 casos límite con predict.py y deriva train/test (KS y PSI).
Resultado:       predict.py reproduce el notebook 07 (178 días, diferencia máxima 5.7e-14); rechaza los 6 días con horas ausentes.
                 Ruido: RMSLE 0.3505 -> 0.3716 (+6.0 %, nivel 2) -> 0.4231 (+20.7 %, nivel 4); sin lags degrada más (+26.6 % en el nivel 4).
                 Backtest: RMSLE diario mediana 0.267, P90 0.475; error del total diario mediana 10.3 %, P90 25.7 %, sesgo -6.9 %.
                 Reentrenar cada mes: 0.3505 -> 0.3469 (-1.0 %).
                 Operativa: subestima > 20 % en 23.2 % de las horas (16.1 % en horas pico); faltante medio de 176 bicicletas en las 139 horas pico con subestimación > 20 %.
                 Casos límite: ninguno falla ni predice negativo; SIN historial (3 lags ausentes) la predicción cae 84.8 %.
                 Deriva: PSI de la demanda 0.223, de lag_168 0.264, de temp 0.213; humedad 0.070 y viento 0.041 estables.
Decisión:        DOCUMENTAR. Se añadió un aviso en predict.py cuando no hay ningún lag. ⏳ Pendiente: interpolar horas ausentes del pronóstico, un modelo sin lags como respaldo y
                 calibrar los niveles de ruido con pronósticos reales. El umbral de 20 % de la métrica operativa lo define el grupo.
Fuente consultada: plan §8.4. Sin fuente externa.
```
