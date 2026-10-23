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
                                                    f1..f5 = 0.3780 0.4544 0.7630 0.5114 0.3063 | 0.4827 ± 0.1748
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
                 sin lags (ref, gap=24) 0.3800 0.4571 0.7638 0.5075 0.3055 | 0.4828 ± 0.1748 |
                 lag_24 + lag_168       0.4176 0.4690 0.4595 0.3964 0.3195 | 0.4124 ± 0.0599 | 2/3
Decisión:        DESCARTAR según la regla (peor en 3 folds), aunque la media mejora 0.0704. La ganancia está concentrada en los folds 3 y 4
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
Qué se hizo:     FLAML 2.7.0 en Python 3.13.14 / Windows 10, task=regression, metric=rmse sobre log1p(cnt) (equivale a RMSLE), eval_method=cv,
                 split_type=get_cv() (gap=0) o get_cv(gap=24) con lags, estimator_list=[lgbm, rf, extra_tree], time_budget=300 s, seed=42, n_jobs=-1.
                 El mejor estimador se re-evaluó con validar_temporal.
                 Desviación controlada: el preprocesador se ajustó una vez con todo train para la búsqueda (FLAML recibe una matriz); la re-evaluación sí lo ajusta por fold.
Resultado:       RMSLE f1..f5 | media ± std
                 Manual: LightGBM + tuning (gap=0)  0.3709 0.4394 0.7797 0.4090 0.2754 | 0.4549 ± 0.1918
                 AutoML sin lags (lgbm, gap=0)      0.3732 0.4366 0.7702 0.3599 0.2623 | 0.4405 ± 0.1946  (mejor en 4/5; diferencia 0.0144)
                 Manual: LightGBM + lags (gap=24)   0.3751 0.4374 0.5044 0.3572 0.2782 | 0.3905 ± 0.0853
                 AutoML con lags (lgbm, gap=24)     0.3943 0.4234 0.4736 0.3581 0.2885 | 0.3876 ± 0.0697  (mejor en 2/5; diferencia 0.0029)
                 Pérdida interna de FLAML vs validar_temporal: 0.4407 vs 0.4405 (sin lags), 0.3876 vs 0.3876 (con lags). 12 configuraciones por vía en 300 s.
                 Configuración ganadora sin lags: num_leaves=4, min_child_samples=18, learning_rate=0.080, n_estimators=2027, reg_alpha=2.8.
                 Configuración ganadora con lags: num_leaves=55, min_child_samples=37, learning_rate=0.060, n_estimators=185, colsample_bytree=0.42.
Decisión:        ADOPTAR FLAML como benchmark. Sin lags AutoML supera al mejor manual (0.4405 vs 0.4549); con lags prácticamente empatan (0.3876 vs 0.3905).
                 🔶 Sin lags AutoML eligió árboles muy pequeños (num_leaves=4); según la ficha 4, el tuning manual exploró num_leaves en [31, 127].
                 Con lags eligió árboles más grandes (num_leaves=55). Con la misma semilla y distinto hardware la configuración ganadora cambió, pero el RMSLE casi no:
                 la búsqueda no es reproducible al detalle y el resultado sí es estable. El puntaje de AutoML también es optimista (la búsqueda ve los mismos folds).
                 No cambia el modelo final por sí solo: la elección es del Sprint 3.
Fuente consultada: documentación y código de FLAML (decide_split_type acepta splitter con split y get_n_splits). Sin fuente externa adicional.
```
