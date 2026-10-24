# Nota de investigación — Pardo (Sprint 0/2)

Temas asignados (`plan_sprints.md` §3): elección de k, variables cíclicas en clustering, estabilidad de clusters; y para el Sprint 2, codificación de la hora en modelos lineales.

## 1. Elección de k
- **Codo (inercia):** útil solo si hay un quiebre claro; con datos continuos la curva baja suave y el "codo" es subjetivo.
- **Silueta:** mide qué tan separado está cada punto de su cluster vecino (−1 a 1). Valores < 0.25 indican estructura débil. Tiende a favorecer k pequeños (k = 2) porque las particiones gruesas separan mejor. Cuesta O(n²): se calcula sobre una muestra fija.
- **Davies-Bouldin:** razón entre dispersión interna y distancia entre centroides (menor es mejor); suele bajar lento con k.
- **BIC (GMM):** penaliza parámetros; con covarianza completa y datos continuos puede seguir bajando sin mínimo, porque más componentes siempre aproximan mejor una densidad.
- **Conclusión aplicada:** cuando los índices no coinciden, se decide por **estabilidad + interpretabilidad** y se dice explícitamente que la estructura es débil.

## 2. Variables cíclicas en clustering
- La hora entera rompe la continuidad (23 h y 0 h quedan a 23 unidades). Se usa `sin(2πh/24)` y `cos(2πh/24)`: puntos sobre un círculo, distancia correcta.
- Ojo: el par sin/cos son **dos** dimensiones, así que la hora pesa el doble que una variable simple en la distancia euclidiana. Con 6 entradas, la hora es un tercio de la distancia.
- Variables binarias (`workingday`) escaladas dominan la distancia y "parten" el espacio en dos bloques; se ve en UMAP.

## 3. Estabilidad de clusters
- **Semillas:** re-ajustar K-Means con otra `random_state` y comparar con el índice de Rand ajustado (ARI; 1 = idéntico).
- **Bootstrap:** re-ajustar en remuestreos y asignar todas las filas; la media y el mínimo del ARI indican qué tan frágiles son las fronteras.
- **Temporal (específico de series):** ajustar con un periodo (2011) y asignar otro; imita lo que pasará al asignar el test.
- Un cluster estable no es "real": solo dice que el algoritmo lo reproduce. La validez externa se juzga con interpretación y utilidad (p. ej. error por cluster en el Sprint 3).

## 4. Codificación de la hora en modelos lineales
Fuente principal: ejemplo de scikit-learn *Time-related feature engineering* (Bike Sharing Demand).
- **Entero:** un lineal solo puede subir o bajar con la hora; no puede representar dos picos.
- **sin/cos:** una sola onda; capta día/noche pero no los picos de 8 h y 17–18 h.
- **One-Hot:** 24 coeficientes libres, sin suavidad; capta cualquier perfil.
- **Splines periódicos** (`SplineTransformer(extrapolation="periodic")`): curvas suaves que empalman 23 h con 0 h; con suficientes nudos igualan al One-Hot con menos columnas.
- **Interacciones:** el ejemplo muestra que el salto grande viene de que el perfil horario cambie según el tipo de día (y otras variables). En nuestro caso el contrato ya trae `hr_workingday` (One-Hot de 48 categorías), así que la codificación de la hora sola deja de importar; lo que suma es una interacción nueva, hora × temperatura.

## Fuentes
- scikit-learn, *Time-related feature engineering* (galería de ejemplos) y documentación de `SplineTransformer`, `KMeans`, `GaussianMixture`, `silhouette_score`, `adjusted_rand_score`.
- McInnes et al., documentación de `umap-learn` (*How to use UMAP*, sección sobre interpretación de distancias).
- Labs 5 y 6 del curso.
