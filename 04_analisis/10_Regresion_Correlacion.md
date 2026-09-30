# Análisis - Regresión Lineal y Correlación

## 1. Introducción

En este trabajo se implementó un algoritmo de regresión lineal simple
utilizando datos relacionados con la producción de arroz.

El objetivo es analizar la relación existente entre la superficie cultivada,
medida en hectáreas, y la producción de arroz, medida en quintales.

Las variables utilizadas son:

- Variable independiente: SUPERFICIE (Ha)
- Variable dependiente: PRODUCCION (qq)

---

## 2. Objetivo

El objetivo es analizar si existe una relación lineal entre la superficie
cultivada y la producción de arroz y utilizar esta relación para realizar
predicciones mediante un modelo de regresión lineal.

---

## 3. Datos utilizados

Los datos fueron obtenidos del archivo:

`0901_Produccion_Arroz.xlsx`

El archivo se encuentra dentro de la carpeta:

`datos`

El programa utiliza las columnas:

- `SUPERFICIE (Ha)`
- `PRODUCCION (qq)`

---

## 4. Preprocesamiento

Antes de entrenar el modelo se realizó una limpieza básica de los datos.

Se eliminaron los registros que contenían valores vacíos en las columnas
SUPERFICIE (Ha) y PRODUCCION (qq).

El código utilizado fue:

```python
data = data.dropna(
    subset=['SUPERFICIE (Ha)', 'PRODUCCION (qq)']
)
**Correlación = 0.9342**

Después coloca:

**R² = 0.7664**

**MAE = 2136.91 qq**

**RMSE = 19427.76 qq**

**MSE = 377438039.86**

**MAPE = 110.44 %**

Y la ecuación:

**Producción = -495.50 + (48.70 × Superficie)**
#Análisis de los resultados

Después de realizar la limpieza de los datos se trabajó con 1375 registros
que contenían información válida para las variables SUPERFICIE (Ha) y
PRODUCCION (qq).

El coeficiente de correlación de Pearson obtenido fue de 0.9342. Este
resultado indica una relación lineal positiva fuerte entre la superficie
cultivada y la producción de arroz. En términos generales, los registros con
mayor superficie tienden a presentar también una mayor producción.

El modelo de regresión lineal fue entrenado utilizando el 70% de los datos
(962 registros) y evaluado con el 30% restante (413 registros).

El coeficiente de determinación obtenido sobre los datos de prueba fue
R² = 0.7664. Esto indica que aproximadamente el 76.64% de la variabilidad
observada en la producción queda explicada por la superficie mediante el
modelo lineal utilizado.

La ecuación obtenida fue:

Producción = -495.50 + (48.70 × Superficie)

La pendiente de 48.70 indica que, según el modelo, un incremento de una
hectárea en la superficie está asociado con un incremento estimado de
aproximadamente 48.70 quintales en la producción.

Sin embargo, las métricas de error muestran que existen diferencias
importantes entre algunas predicciones y los valores reales. El MAE obtenido
fue de 2136.91 qq y el RMSE fue de 19427.76 qq. La diferencia considerable
entre ambas métricas indica que existen algunos errores de gran magnitud que
incrementan el RMSE.

El MAPE obtenido fue de 110.44%. Este resultado debe interpretarse con
precaución, ya que el MAPE puede aumentar considerablemente cuando existen
valores reales pequeños en la variable que se está prediciendo.

Por lo tanto, aunque existe una relación lineal fuerte entre superficie y
producción, la superficie por sí sola no explica completamente todos los
factores que afectan la producción de arroz. Variables adicionales podrían
ser consideradas en futuros modelos para representar mejor las diferencias
observadas en la producción.
## Conclusión

El algoritmo desarrollado permitió analizar la relación entre la superficie
cultivada y la producción de arroz mediante correlación de Pearson y
regresión lineal simple.

La correlación obtenida fue de 0.9342, mostrando una relación lineal positiva
fuerte entre ambas variables. El modelo de regresión obtuvo un R² de 0.7664
sobre los datos de prueba.

Sin embargo, las métricas de error muestran que existen diferencias
importantes entre determinadas predicciones y los valores reales, por lo que
el modelo basado únicamente en la superficie no representa completamente la
variabilidad de la producción.

El análisis demuestra la utilidad de combinar la correlación numérica con la
regresión lineal para estudiar relaciones entre variables y realizar
estimaciones, además de identificar las limitaciones del modelo utilizado.