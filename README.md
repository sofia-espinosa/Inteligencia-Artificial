# Inteligencia Artificial — Clasificación de TEA en adultos con Machine Learning

Trabajo final (Evaluación 5) de la asignatura **Inteligencia Artificial**, Ingeniería Civil Informática, Universidad Católica del Maule.
Se comparan tres algoritmos de clasificación supervisada —**Naive Bayes**, **SVM** y **Redes Neuronales**— para predecir si una persona adulta presenta rasgos del trastorno del espectro autista (TEA).

## Dataset

[Autism Screening Adult](https://archive.ics.uci.edu/dataset/426) — UCI Machine Learning Repository (id `426`), 704 registros y 21 columnas.

- **Numéricas:** `A1_Score`–`A10_Score` (respuestas del cuestionario), `age`, `result` (puntaje total).
- **Categóricas:** `gender`, `ethnicity`, `jaundice`, `family_pdd`, `country_of_res`, `used_app_before`, `age_desc`, `relation`.
- **Objetivo:** `class` (`YES` / `NO`) → clasificación binaria.

Se descarga automáticamente con `ucimlrepo`; también se incluye una copia en `Autism-Adult-Data.xlsx`.

## Estructura

| Archivo | Contenido |
|---|---|
| `NaiveBayes.ipynb` | EDA, preprocesamiento y `GaussianNB` (base, `RandomizedSearchCV`, `GridSearchCV` sobre `var_smoothing`) |
| `svm.ipynb` | EDA, preprocesamiento y `SVC` (base, búsqueda manual de `C`/`gamma` con kernel RBF, `GridSearchCV` con k-fold) |
| `RN.ipynb` | Red neuronal en Keras (base 32-16-1, ajuste con Keras-Tuner, validación K-Fold con SciKeras) |
| `Autism-Adult-Data.xlsx` | Copia local del dataset |
| `Informe final inteligencia artificial.pdf` | Informe con metodología, métricas y comparación |

## Metodología

1. **Limpieza:** edades imposibles (<0 o >120) → NaN; imputación de `age` con mediana y de `ethnicity`/`relation` con moda; eliminación de duplicados.
2. **Codificación:** One-Hot para categóricas multiclase, Label Encoding para binarias.
3. **Escalado:** `StandardScaler` / Min-Max según el modelo.
4. **Partición:** 80/20 estratificada (`random_state=42`).
5. **Tres casos por algoritmo:** (1) sin optimización, (2) optimización de hiperparámetros, (3) optimización + validación cruzada.
6. **Métricas:** exactitud, sensibilidad, especificidad, VPP/VPN, matriz de confusión y curva ROC/AUC.

## Resultados principales

| Modelo | Exactitud | AUC | Observación |
|---|---|---|---|
| Naive Bayes – Caso 1 | 0.31 | 0.558 | Casi azar; 140 falsos positivos |
| **Naive Bayes – Caso 2** | **0.971** | **0.997** | Mejor modelo global (sensibilidad 0.964) |
| Naive Bayes – Caso 3 | ~0.97 | 0.997 | Máxima especificidad (0.993) y VPP (0.989) |
| SVM – Caso 1 | 0.929 | — | Parámetros por defecto |
| SVM – Casos 2 y 3 | 0.943 | — | La optimización aporta; el k-fold no cambia el resultado |
| Red Neuronal | alta | — | Caso 2 generaliza mejor; Caso 3 muestra sobreajuste |

**Conclusión:** la optimización de hiperparámetros fue el factor decisivo. Naive Bayes optimizado (Caso 2) obtuvo las mejores métricas, mientras que las redes neuronales destacaron por su estabilidad.

## Cómo ejecutar

Los notebooks están pensados para **Google Colab** (instalan sus dependencias en la primera celda).

En local:

```bash
pip install ucimlrepo pandas numpy scikit-learn seaborn matplotlib tensorflow keras-tuner scikeras jupyter
jupyter notebook
```
