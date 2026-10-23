# Resumen del Sprint 2 — parte de Pardo (Ridge/Lasso y clustering)

✅ verificado con código ejecutado · 🔶 hipótesis · ⏳ pendiente. Todo sale de `notebooks/03b_ridge.ipynb` y `notebooks/02_no_supervisado.ipynb`, ejecutados en un `.venv` con Python 3.13 y los pines de `requirements.txt` (+ `umap-learn`). Validación temporal solo con train; **el test no se ha usado**. Fichas 8 a 11 en `reports/decisiones.md`.

## Ridge / Lasso (RMSLE medio ± std, 5 folds, `gap=0` salvo indicación)
| Variante | RMSLE | Folds |
| :--- | :---: | :--- |
| Hora entera, **sin** `hr_workingday` | 1.0992 ± 0.1093 | |
| sin/cos, sin `hr_workingday` | 0.9475 ± 0.0998 | |
| One-Hot / splines, sin `hr_workingday` | 0.6786 / 0.6800 | |
| Contrato (`hora='entero'` + `hr_workingday`) | 0.4930 ± 0.1614 | 0.3831 0.5318 0.7578 0.4264 0.3661 |
| sin/cos, One-Hot o splines + `hr_workingday` | 0.4930–0.4932 | empate (±0.0002) |
| **Final: + splines × temperatura, Ridge `alpha=1`** | **0.4885 ± 0.1583** | 0.3821 0.5243 0.7489 0.4235 0.3638 (mejor en 5/5) |
| Final + `lag_24` + `lag_168` (`gap=24`) | 0.4766 ± 0.1335 | vs 0.4889 sin lags: mejor en 3, peor en 2 |

- ✅ Ridge final: **12.5 % mejor que el baseline de calendario** (0.5585) y 0.034 peor que LightGBM con tuning (0.4549).
- ✅ La codificación de la hora importa **solo sin interacción**; con `hr_workingday` las cuatro empatan. No se cambia `CONFIG["features"]["hora"]`.
- ✅ Lasso no mejora (con `alpha=1e-3` anula 28 de 90 coeficientes y empeora). `alpha` de Ridge plano entre 0.1 y 3.
- ✅ Lags en Ridge: mejoran la media en 0.012 pero **no cumplen la regla** (peor en folds 4 y 5). Insumo para la decisión 8a.
- Código: `src/lineales.py` (`PrepLineal`), que envuelve `build_preprocessor(escalar=True)` **sin modificarlo**.

## Clustering (K-Means, entradas §5.1, solo train)
| k | Silueta | ARI semillas (mín) | ARI 2011 → todo el train |
| :---: | :---: | :---: | :---: |
| 2 | 0.194 | 0.990 | 0.876 |
| **4** | **0.177** | **0.997** | **0.955** |
| 6 | 0.190 | 0.981 | 0.805 |

- ✅ **Estructura débil**: silueta 0.17–0.20 para todo k; BIC del GMM sin mínimo. Se reporta así.
- ✅ **k = 4** por estabilidad temporal e interpretabilidad. Arquetipos: tarde cálida y seca (demanda media 278), mañana laborable (138), no laborable (125), noche laborable (118).
- ✅ Los clusters explican el 17.5 % de la varianza de `log1p(cnt)` frente al 81 % de hora × `workingday`: **se usan para segmentar el error, no como feature**.
- ✅ PCA: varianza repartida (4 componentes para 79 %). UMAP: dos bloques por `workingday` con un gradiente horario continuo.
- Modelo: `models/clustering_pardo.joblib` (no viaja con git; se regenera con el notebook 02 en ~1 min). Uso: `joblib.load(ruta)["pipeline"].predict(X)`.

## Para el resto del equipo
- **Joan:** `features.py` y `evaluate.py` no se tocaron. Agregué `src/lineales.py` y `src/clustering.py` (archivos míos). `umap-learn` no está en `requirements.txt`: propongo `umap-learn==0.5.12` (instala bien en 3.13 con `numba 0.68`). Lo dejo a tu decisión porque el archivo es tuyo.
- **José (Sprint 3):** el Ridge final para la tabla es `Pipeline([("prep", PrepLineal(codif_hora="splines", interaccion_temp=True)), ("reg", Ridge(alpha=1.0))])` con `log_target=True`.

## Limitaciones
1. Como en LightGBM, el fold 3 domina el error (train solo 2011): el lineal no extrapola el crecimiento con una sola `yr`.
2. La elección de `alpha` y de la interacción se hizo sobre los mismos folds: el puntaje es algo optimista (la superficie es plana, así que el sesgo es pequeño).
3. 🔶 Sin la lluvia como entrada, ningún cluster captura "clima adverso"; agregar `weathersit` cambiaría los arquetipos.
4. ⏳ Sprint 3: RMSLE por cluster del modelo final y residuos por hora y día de la semana.
