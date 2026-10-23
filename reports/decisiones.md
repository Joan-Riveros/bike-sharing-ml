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
