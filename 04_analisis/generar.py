import json
from pathlib import Path

DIR_SALIDA = Path(__file__).resolve().parent

def crear_notebook(celdas):
    return {
        "cells": celdas,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def celda_md(texto):
    lineas = [linea + "\n" for linea in texto.strip().split("\n")]
    if lineas:
        lineas[-1] = lineas[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": lineas}

# Diccionario con el contenido estricto de cada cuaderno
cuadernos = {
    "12_Analisis_Regresion_V3.ipynb": [
        celda_md("""# Analisis V3 - Modelos Parametricos de Regresion: Lineal y Polinomial Ridge

## 1. Contexto y Objetivos
Evaluacion de los modelos parametricos definitivos (V3) para estimar la Produccion (qq) en funcion de la Superficie (Ha).
El objetivo principal fue corregir el severo error porcentual observado en V1 (MAPE > 110%) mediante transformacion logaritmica y exclusion de valores de apalancamiento extremo."""),
        celda_md("""## 2. Metodologia y Correccion Matematica
* Datos: 0901_Produccion_Arroz_V2_Regresion.xlsx (1,373 registros, sin outliers >= 3,000 Ha).
* Transformacion: ln(1 + x) en predictor y variable objetivo para convertir la relacion de potencia en lineal:
  ln(Produccion) = ln(alpha) + beta * ln(Superficie)
* Modelos: Regresion Lineal OLS (10_Regresion_Lineal_V3.py) y Polinomial Ridge grado 2 con alpha optimizado via GridSearchCV (11_Regresion_Polinomial_V3.py).
* Metricas: Calculadas en quintales reales aplicando expm1."""),
        celda_md("""## 3. Resultados y Comparacion con Versiones Anteriores

| Version | Modelo | Condicion de Datos | R2 Score | MAE (qq) | RMSE (qq) | MAPE (%) |
|---|---|---|:---:|:---:|:---:|:---:|
| V1 | Lineal Simple | Crudo (con mega-outlier) | 0.7664 | 2,136.91 | 19,427.76 | 110.44% |
| V1 | Polinomial Grado 4 | Crudo sin regularizar | 0.8193 | 1,134.59 | 3,566.07 | 69.18% |
| V2 (Escenario B) | Polinomial Grado 4 | Filtrado sin logaritmo | -0.1581 | 1,744.17 | 9,986.51 | 69.62% |
| V3 (Definitiva) | Lineal V3 | Filtrado + ln(1+x) | 0.6787 | 1,470.68 | 5,259.79 | 49.03% |
| V3 (Definitiva) | Polinomial Ridge V3 | Filtrado + ln(1+x) (alpha=10) | 0.5651 | 1,365.46 | 6,119.86 | 51.05% |

Formula Calibrada para Produccion Web:
Produccion (qq) = exp(3.5059 + 0.9667 * ln(Superficie + 1)) - 1"""),
        celda_md("""## 4. Interpretacion Tecnica
1. Elasticidad unitaria: La pendiente b1 = 0.9667 confirma rendimientos constantes a escala (Produccion proporcional a Superficie^0.97).
2. Reduccion de error relativo: El MAPE descendio de 110.44% en V1 a 49.03% en V3, eliminando la sobrestimacion en pequenos productores.
3. Comportamiento de Ridge: Pese a reducir el MAE a 1,365 qq, su R2 desciende a 0.5651 por penalizacion en colas superiores."""),
        celda_md("""## 5. Visualizacion
![Regresion Lineal V3](../03_imagenes/28_regresion_lineal_v3.png)
![Regresion Polinomial Ridge V3](../03_imagenes/29_regresion_polinomial_v3.png)"""),
        celda_md("""## 6. Conclusiones y Veredicto de Implementacion
La Regresion Lineal V3 es la opcion recomendada para despliegue web serverless. Garantiza un error relativo acotado (<50%) y permite computar inferencias directamente en JavaScript sin dependencias de backend.""")
    ],

    "13_Analisis_Clasificacion_V3.ipynb": [
        celda_md("""# Analisis V3 - Clasificacion de Campanas Agricolas: Regresion Logistica y KNN

## 1. Contexto y Justificacion del Submuestreo
En V1 y V2 la distribucion desbalanceada (89.8% verano frente a 10.2% invierno) inducia a los modelos a predecir siempre verano, logrando un Accuracy enganoso del 90% con Recall de invierno nulo (2.4% en KNN).
En V3 se evaluaron ambos algoritmos sobre el dataset balanceado 50/50 (0901_Produccion_Arroz_V3_Clasificacion.xlsx)."""),
        celda_md("""## 2. Metodologia
* Datos: 280 observaciones (140 Verano vs 140 Invierno).
* Variables: Superficie (Ha) y Produccion (qq) estandarizadas con StandardScaler.
* Particion: 70% entrenamiento y 30% evaluacion estratificada (42 muestras por clase en test).
* Optimizacion: GridSearchCV sobre C en Logistica, y sobre K y ponderacion en KNN."""),
        celda_md("""## 3. Resultados y Comparacion Historica

| Algoritmo | Version | Balance de Datos | Accuracy | Recall (Inv) | Precision (Inv) | F1-Score | ROC-AUC |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| KNN | V1 | Desbalanceado (89.8% Verano) | 89.35% | 2.38% (1/42) | 25.00% | 0.0450 | 0.5120 |
| KNN | V2 | Desbalanceado + Distancia | 86.44% | 19.05% (8/42) | 26.67% | 0.2222 | 0.6147 |
| KNN | V3 | Balanceado 50/50 (K=5) | 60.71% | 57.14% (24/42) | 61.54% | 0.5926 | 0.6842 |
| Reg. Logistica | V1 | Desbalanceado (Umbral 0.5) | 90.55% | 7.14% (2/28) | 100.0% | 0.1250 | 0.5350 |
| Reg. Logistica | V2 | Desbalanceado + Balanced | 82.81% | 33.33% (14/42) | 24.56% | 0.2828 | 0.6572 |
| Reg. Logistica | V3 | Balanceado 50/50 (C=100) | 67.86% | 47.62% (20/42) | 80.00% | 0.5970 | 0.7426 |

Coeficientes Estandarizados Regresion Logistica V3:
Superficie: +10.5026 | Produccion: -9.6289 | Intercepto: +0.4805"""),
        celda_md("""## 4. Interpretacion Tecnica
1. Recuperacion de KNN: El balanceo permitio pasar de 1 acierto en V1 a 24 aciertos en V3 (Recall = 57.14%), demostrando que su fallo previo era muestral y no del principio de distancia.
2. Precision de Regresion Logistica: Con C=100 alcanza un 80% de precision en invierno con solo 5 falsos positivos.
3. Regla fisica: Un mayor coeficiente positivo en Superficie y negativo en Produccion indica que parcelas extensas con rendimientos moderados caracterizan al ciclo invernal."""),
        celda_md("""## 5. Visualizacion
![Regresion Logistica V3](../03_imagenes/30_logistica_v3.png)
![KNN V3](../03_imagenes/31_knn_v3.png)"""),
        celda_md("""## 6. Conclusiones y Veredicto
Regresion Logistica V3 supera a KNN al registrar mayor precision (80% vs 61.5%) y mayor ROC-AUC (0.7426 vs 0.6842), eliminando ademas la necesidad de retener el conjunto de entrenamiento en memoria durante la inferencia web.""")
    ],

    "14_Analisis_Naive_Bayes_Clasificacion_V2.ipynb": [
        celda_md("""# Analisis V2 - Naive Bayes Gaussiano: Clasificacion de Campana Agricola

## 1. Contexto y Correccion de Formulacion
En la version inicial se forzo Naive Bayes a predecir produccion continua discretizando artificialmente en 3 niveles, obteniendo un rendimiento del 50%.
En esta version V2 se devuelve el algoritmo a su dominio natural de clasificacion supervisada, empleando Gaussian Naive Bayes sobre variables continuas y el dataset balanceado V3."""),
        celda_md("""## 2. Metodologia
* Datos: 0901_Produccion_Arroz_V3_Clasificacion.xlsx (140 Verano vs 140 Invierno).
* Cumplimiento del supuesto gaussiano: Se aplico ln(1+x) a Superficie y Produccion para normalizar la asimetria positiva de las distribuciones.
* Evaluacion: Particion estratificada 70/30 (42 muestras por clase en test)."""),
        celda_md("""## 3. Resultados y Comparacion con V1

| Metrica | Naive Bayes V1 (Multinomial / Geografia) | Naive Bayes V2 (Gaussiano Balanceado + ln(1+x)) |
|---|:---:|:---:|
| Accuracy Global | 88.62% | 67.86% |
| Precision (Invierno) | 40.00% | 74.19% |
| Recall (Invierno) | 23.80% (10/42) | 54.76% (23/42) |
| F1-Score (Invierno) | 0.3010 | 0.6301 |
| ROC-AUC | 0.6180 | 0.7120 |

Matriz de Confusion V2:
Verano: 34 aciertos, 8 confusiones con invierno.
Invierno: 23 aciertos, 19 confusiones con verano."""),
        celda_md("""## 4. Interpretacion Tecnica
1. Impacto de log1p: La compresion logaritmica permitio que las funciones de densidad condicional P(X|C) modelaran centroides claros sin desviaciones estandar infladas.
2. Desempeno frente a KNN y Logistica: El F1-Score de 0.6301 y Precision de 74.19% superan a KNN V3 (0.5926) y Regresion Logistica V3 (0.5970)."""),
        celda_md("""## 5. Visualizacion
![Naive Bayes Gaussiano V2](../03_imagenes/32_nb_clasificacion_v2.png)"""),
        celda_md("""## 6. Conclusiones y Veredicto
Gaussian Naive Bayes V2 se valida como un clasificador probabilistico rapido y solido tras corregir la distribucion de entrada, constituyendo una alternativa eficiente para microservicios web ligeros.""")
    ],

    "15_Analisis_Arboles_Decision_V2.ipynb": [
        celda_md("""# Analisis V2 - Arboles de Decision: Regresion y Clasificacion

## 1. Contexto y Erradicacion de Fuga de Datos
En V1 el arbol de regresion incluia RENDIMIENTO (Kg/Ha) junto a SUPERFICIE (Ha) para predecir PRODUCCION (qq), aprendiendo una identidad aritmetica trivial.
En V2 se elimino totalmente la variable de rendimiento en regresion y se evaluo el arbol de clasificacion sobre la muestra balanceada V3."""),
        celda_md("""## 2. Metodologia
* Regresion V2: 0901_Produccion_Arroz_V2_Regresion.xlsx (1,373 filas), modelando sobre ln(1 + Superficie). Poda optima con GridSearchCV(KFold=5) sobre profundidad y muestras minimas en hoja.
* Clasificacion V2: 0901_Produccion_Arroz_V3_Clasificacion.xlsx (280 filas), evaluando Superficie, Produccion y Rendimiento con GridSearchCV(StratifiedKFold=5)."""),
        celda_md("""## 3. Resultados y Comparacion Historica

Regresion de Produccion (Quintales Reales):
Hiperparametros V2: max_depth = 4, min_samples_leaf = 10, min_samples_split = 2.

| Metrica | Arbol V1 (Con Fuga y Datos Crudos) | Arbol V2 (Sin Fuga + ln(1+x) + Poda) | Mejora |
|---|:---:|:---:|:---:|
| R2 Score | 0.3726 | 0.6704 | +0.2978 |
| MAE | 2,660.00 qq | 1,435.64 qq | -46.0% |
| RMSE | 31,837.48 qq | 5,327.80 qq | -83.3% |
| MAPE | 66.10% | 49.56% | -16.5 pp |

Clasificacion de Campana Agricola (Test 42 vs 42):
Hiperparametros V2: criterion = 'gini', max_depth = 3, min_samples_leaf = 2, min_samples_split = 10.

| Metrica | Arbol V1 (Desbalanceado + Balanced) | Arbol V2 (Balanceado 50/50 + Poda) |
|---|:---:|:---:|
| Accuracy | 80.15% | 69.05% |
| Precision (Invierno) | 27.27% | 71.05% |
| Recall (Invierno) | 57.14% (24/42) | 64.29% (27/42) |
| F1-Score (Invierno) | 0.3700 | 0.6750 |
| ROC-AUC | 0.7020 | 0.7364 |"""),
        celda_md("""## 4. Interpretacion y Reglas de Decision
1. Poda y linealizacion: La restriccion a profundidad 4 y hojas de minimo 10 observaciones estabilizo la varianza, subiendo el R2 de 0.37 a 0.67 y reduciendo el RMSE en 83%.
2. Reglas Agronomicas en Clasificacion:
   * Corte raiz: Superficie <= 56.0 Ha.
   * Parcelas extensas (>56 Ha) con Rendimiento <= 2,316.15 Kg/Ha se clasifican como Invierno con pureza Gini = 0.085.
   * Parcelas pequenas (<=56 Ha) con Rendimiento <= 1,035.4 Kg/Ha discriminan Invierno."""),
        celda_md("""## 5. Visualizacion
![Arbol de Regresion V2](../03_imagenes/33_dt_regresion_v2.png)
![Arbol de Clasificacion V2](../03_imagenes/34_dt_clasificacion_v2.png)"""),
        celda_md("""## 6. Conclusiones y Veredicto
El Arbol de Clasificacion V2 es el mejor modelo explicable de caja blanca del proyecto: alcanza un Recall de 64.29% y F1 de 0.6750, y su logica puede trasladarse a estructuras if/else directas en codigo web.""")
    ],

    "16_Analisis_Boosting_V2.ipynb": [
        celda_md("""# Analisis V2 - Ensambles Boosting: AdaBoost vs LightGBM

## 1. Contexto y Objetivos
Evaluacion de tecnicas de potenciacion secuencial (AdaBoost y LightGBM) tras erradicar la fuga de datos en regresion y balancear la muestra en clasificacion."""),
        celda_md("""## 2. Metodologia
* Regresion V2: 0901_Produccion_Arroz_V2_Regresion.xlsx, modelando produccion en escala ln(1+y) solo a partir de Superficie.
* Clasificacion V2: 0901_Produccion_Arroz_V3_Clasificacion.xlsx (280 filas, 140 vs 140) con 100 estimadores y tasa de aprendizaje 0.05."""),
        celda_md("""## 3. Resultados y Comparacion Historica

Regresion de Produccion (Escala Real en Quintales):

| Algoritmo | Version | R2 Score | MAE (qq) | RMSE (qq) | MAPE (%) |
|---|---|:---:|:---:|:---:|:---:|
| AdaBoost | V1 (Con Fuga) | 0.3761 | 2,979.76 | 18,200.00 | ~75% |
| AdaBoost | V2 (Sin Fuga + Log) | 0.4140 | 1,777.09 | 7,103.87 | 49.78% |
| LightGBM | V1 (Con Fuga) | 0.2154 | 2,559.12 | 24,100.00 | ~62% |
| LightGBM | V2 (Sin Fuga + Log) | 0.7682 | 1,254.62 | 4,467.49 | 45.20% |

Clasificacion de Campana Agricola (Test 42 vs 42):

| Algoritmo | Version | Recall (Inv) | Precision (Inv) | F1-Score | ROC-AUC |
|---|---|:---:|:---:|:---:|:---:|
| AdaBoost | V1 (Desbalanceado) | 2.38% (1/42) | 50.00% | 0.0455 | 0.7100 |
| AdaBoost | V2 (Balanceado) | 45.24% (19/42) | 76.00% | 0.5672 | 0.6610 |
| LightGBM | V1 (Desbalanceado) | 26.19% (11/42) | 22.92% | 0.2444 | 0.6960 |
| LightGBM | V2 (Balanceado) | 61.90% (26/42) | 66.67% | 0.6420 | 0.7183 |"""),
        celda_md("""## 4. Interpretacion Tecnica
1. Dominio de LightGBM en Regresion: Con R2 = 0.7682, MAE = 1,254.62 qq y MAPE = 45.20%, LightGBM se establece como el modelo mas preciso de todo el repositorio.
2. Control residual: Sus residuos se concentran simetricamente en cero dentro de un rango de +-20,000 qq, a diferencia de AdaBoost cuyos residuos se dispersan hasta -80,000 qq.
3. Desempeno en Clasificacion: LightGBM supera a AdaBoost en Recall (61.90% vs 45.24%) y ROC-AUC (0.7183 vs 0.6610)."""),
        celda_md("""## 5. Visualizacion
![Boosting Regresion V2](../03_imagenes/35_boosting_reg_v2.png)
![Boosting Clasificacion V2](../03_imagenes/36_boosting_clf_v2.png)"""),
        celda_md("""## 6. Conclusiones y Veredicto
LightGBM Regresion V2 es el modelo ganador para el backend de estimacion de cosechas en la web, al garantizar el menor margen de error relativo medio (45.20%).""")
    ],

    "17_Analisis_GradientBoosting_V2.ipynb": [
        celda_md("""# Analisis V2 - Gradient Boosting Avanzado: XGBoost vs CatBoost

## 1. Contexto y Comparacion Estructural
Evaluacion de XGBoost y CatBoost incorporando DEPARTAMENTO y CAMPAÑA AGRICOLA en regresion (sin fuga de datos) y en el dataset balanceado V3 para clasificacion.
Se comparo la eficiencia de dimensionalidad entre One-Hot Encoding y manejo nativo de categorias."""),
        celda_md("""## 2. Metodologia
* Regresion V2: 0901_Produccion_Arroz_V2_Regresion.xlsx. Predictores: Superficie, Departamento y Campana en escala ln(1+y).
* Clasificacion V2: 0901_Produccion_Arroz_V3_Clasificacion.xlsx (280 filas). Predictores: Departamento, Superficie, Produccion y Rendimiento.
* XGBoost: Requiere pd.get_dummies (8 columnas en matriz).
* CatBoost: cat_features nativo (mantiene 4 columnas)."""),
        celda_md("""## 3. Resultados y Comparacion Historica

Regresion de Produccion (Quintales Reales):

| Metrica | XGBoost V2 | CatBoost V2 | Ganador |
|---|:---:|:---:|:---:|
| R2 Score | 0.4545 | 0.3582 | XGBoost |
| MAE | 1,426.35 qq | 1,506.54 qq | XGBoost |
| RMSE | 6,854.22 qq | 7,434.33 qq | XGBoost |
| MAPE (%) | 52.64% | 51.71% | CatBoost |

Clasificacion de Campana Agricola (Test 42 vs 42):

| Metrica | CatBoost V1 (Desbalanceado) | XGBoost V2 (Balanceado) | CatBoost V2 (Balanceado) |
|---|:---:|:---:|:---:|
| Accuracy | 72.63% | 78.57% | 79.76% |
| Precision (Inv) | 24.82% | 77.27% | 76.60% |
| Recall (Inv) | 83.33% (35/42) | 80.95% (34/42) | 85.71% (36/42) |
| F1-Score (Inv) | 0.3825 | 0.7907 | 0.8090 |
| ROC-AUC | 0.7850 | 0.8583 | 0.8679 |
| Columnas Matriz | 4 | 8 (One-Hot) | 4 (Nativo) |"""),
        celda_md("""## 4. Interpretacion Tecnica
1. Record Absoluto en Clasificacion: CatBoost V2 alcanza las mejores metricas de todo el proyecto: ROC-AUC = 0.8679, Recall = 85.71% (36 de 42 inviernos detectados) y F1-Score = 0.8090.
2. Eficiencia de Memoria: CatBoost logra este desempeno preservando la matriz en 4 columnas frente a las 8 requeridas por XGBoost, reduciendo consumo de RAM al 50%.
3. Regresion: XGBoost supera ligeramente a CatBoost en R2 (0.4545 vs 0.3582), pero ambos quedan por detras de LightGBM V2 (R2 = 0.7682) al introducir ruido por variables departamentales dispersas."""),
        celda_md("""## 5. Visualizacion
![Gradient Boosting Regresion V2](../03_imagenes/37_gb_reg_v2.png)
![Gradient Boosting Clasificacion V2](../03_imagenes/38_gb_clf_v2.png)"""),
        celda_md("""## 6. Conclusiones y Veredicto
CatBoost Clasificador V2 es el ganador indiscutible en clasificacion estacional para produccion web, gracias a su tasa de deteccion del 85.71% y procesamiento nativo de variables de texto.""")
    ],

    "18_Analisis_SVM_V2.ipynb": [
        celda_md("""# Analisis V2 - Support Vector Machines: SVR y SVC

## 1. Contexto y Diagnostico del Fallo en V1
En V1 el SVR colapso con R2 = -0.0080 y predicciones planas en cero porque la produccion (hasta 800,000 qq) no fue escalada frente a C=10 y epsilon=0.1.
En V2 se aplico la transformacion logaritmica bilog ln(1+x) y estandarizacion, y se evaluo SVC sobre el dataset balanceado V3."""),
        celda_md("""## 2. Metodologia
* SVR V2: 0901_Produccion_Arroz_V2_Regresion.xlsx (1,373 filas). StandardScaler sobre variables transformadas y optimizacion con GridSearchCV(KFold=5) sobre C, epsilon y gamma.
* SVC V2: 0901_Produccion_Arroz_V3_Clasificacion.xlsx (280 filas), kernel RBF y proyeccion 2D con PCA para inspeccion geometrica de vectores de soporte."""),
        celda_md("""## 3. Resultados y Comparacion Historica

SVR en Regresion (Quintales Reales):
Hiperparametros V2: kernel = 'rbf', C = 100, epsilon = 0.01, gamma = 0.1.

| Metrica | SVR V1 (Sin Escalar Target) | SVR V2 (Con ln(1+x) + Estandarizacion) |
|---|:---:|:---:|
| R2 Score | -0.0080 | 0.2207 |
| MAE | 4,226.07 qq | 1,485.89 qq |
| RMSE | 40,355.34 qq | 8,192.08 qq |
| MAPE | > 300% | 55.58% |

SVC en Clasificacion (Test 42 vs 42):
Hiperparametros V2: kernel = 'rbf', C = 50, gamma = 'scale'.

| Metrica | SVC V1 (Desbalanceado + Balanced) | SVC V2 (Dataset Balanceado 50/50) |
|---|:---:|:---:|
| Accuracy | 83.05% | 59.52% |
| Precision (Inv) | 26.67% | 72.22% |
| Recall (Inv) | 38.10% (16/42) | 30.95% (13/42) |
| F1-Score (Inv) | 0.3137 | 0.4333 |
| ROC-AUC | 0.6840 | 0.7574 |"""),
        celda_md("""## 4. Interpretacion Tecnica
1. Recuperacion de SVR: Al escalar penalizaciones, SVR paso de R2 negativo a +0.2207 y redujo el MAE a 1,485.89 qq. Sin embargo, el kernel RBF penaliza fuertemente las parcelas mayores a 1,500 Ha.
2. Geometria de Soporte en SVC: La proyeccion PCA muestra que ambas clases estan densamente solapadas cerca del origen. La frontera RBF es excesivamente conservadora, logrando alta precision (72.22%) a costa de un bajo Recall (30.95%)."""),
        celda_md("""## 5. Visualizacion
![SVR Regresion V2](../03_imagenes/39_svm_reg_v2.png)
![SVC Clasificacion V2](../03_imagenes/40_svm_clf_v2.png)"""),
        celda_md("""## 6. Conclusiones y Veredicto
Aunque las Support Vector Machines fueron corregidas de su fallo previo, su costo computacional cubico O(n^3) y sus metricas inferiores respecto a LightGBM (R2=0.768) y CatBoost (F1=0.809) justifican descartarlas para produccion web.""")
    ],

    "19_Consolidado_Final_y_Veredicto_Web.ipynb": [
        celda_md("""# Sintesis Global de Modelos y Arquitectura de Despliegue Web

## 1. Cuadro Maestro Comparativo: Regresion (Produccion en qq)
Evaluado sobre 0901_Produccion_Arroz_V2_Regresion.xlsx (1,373 registros, sin outliers >3,000 Ha, sin fuga de datos):

| Rango | Algoritmo | Script Fuente | R2 Score | MAE (qq) | RMSE (qq) | MAPE (%) | Veredicto de Produccion |
|:---:|---|---|:---:|:---:|:---:|:---:|---|
| 1 | LightGBM V2 | 13_Boosting_Regresion_V2.py | 0.7682 | 1,254.62 | 4,467.49 | 45.20% | Ganador Absoluto (Backend API) |
| 2 | Regresion Lineal V3 | 10_Regresion_Lineal_V3.py | 0.6787 | 1,470.68 | 5,259.79 | 49.03% | Ganador Absoluto (Frontend Serverless) |
| 3 | Arbol de Decision V2 | 12_Arbol_Decision_Regresion_V2.py | 0.6704 | 1,435.64 | 5,327.80 | 49.56% | Interpretable por cortes |
| 4 | Polinomial Ridge V3 | 11_Regresion_Polinomial_V3.py | 0.5651 | 1,365.46 | 6,119.86 | 51.05% | Regularizado (alpha=10) |
| 5 | CatBoost V2 | 14_GradientBoosting_Regresion_V2.py | 0.3582 | 1,506.54 | 7,434.33 | 51.71% | Robusto regionalmente |
| 6 | XGBoost V2 | 14_GradientBoosting_Regresion_V2.py | 0.4545 | 1,426.35 | 6,854.22 | 52.64% | Penalizado por One-Hot |
| 7 | AdaBoost V2 | 13_Boosting_Regresion_V2.py | 0.4140 | 1,777.09 | 7,103.87 | 49.78% | Sensible a varianza |
| 8 | SVR V2 | 15_SVM_Regresion_V2.py | 0.2207 | 1,485.89 | 8,192.08 | 55.58% | No competitivo en exactitud |"""),
        celda_md("""## 2. Cuadro Maestro Comparativo: Clasificacion (Deteccion de Invierno)
Evaluado sobre 0901_Produccion_Arroz_V3_Clasificacion.xlsx (280 registros, test 42 vs 42):

| Rango | Algoritmo | Script Fuente | Accuracy | Recall (Inv) | Precision (Inv) | F1-Score | ROC-AUC | Veredicto de Produccion |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|---|
| 1 | CatBoost V2 | 15_GradientBoosting_Clasificacion_V2.py | 79.76% | 85.71% (36/42) | 76.60% | 0.8090 | 0.8679 | Ganador Absoluto (Produccion) |
| 2 | XGBoost V2 | 15_GradientBoosting_Clasificacion_V2.py | 78.57% | 80.95% (34/42) | 77.27% | 0.7907 | 0.8583 | Alto desempeno, 8 columnas |
| 3 | Arbol de Decision V2 | 13_Arbol_Decision_Clasificacion_V2.py | 69.05% | 64.29% (27/42) | 71.05% | 0.6750 | 0.7364 | Mejor modelo caja blanca |
| 4 | LightGBM V2 | 14_Boosting_Clasificacion_V2.py | 65.48% | 61.90% (26/42) | 66.67% | 0.6420 | 0.7183 | Equilibrado y compacto |
| 5 | Naive Bayes Gaussiano V2 | 12_Naive_Bayes_Clasificacion_V2.py | 67.86% | 54.76% (23/42) | 74.19% | 0.6301 | 0.7120 | Rapido, para microservicios |
| 6 | Regresion Logistica V3 | 10_Regresion_Logistica_V3.py | 67.86% | 47.62% (20/42) | 80.00% | 0.5970 | 0.7426 | Maxima precision |
| 7 | KNN V3 | 11_KNN_V3.py | 60.71% | 57.14% (24/42) | 61.54% | 0.5926 | 0.6842 | Requiere matriz en memoria |
| 8 | AdaBoost V2 | 14_Boosting_Clasificacion_V2.py | 65.48% | 45.24% (19/42) | 76.00% | 0.5672 | 0.6610 | Sensibilidad moderada |
| 9 | SVC (RBF) V2 | 16_SVM_Clasificacion_V2.py | 59.52% | 30.95% (13/42) | 72.22% | 0.4333 | 0.7574 | Conservador en frontera |"""),
        celda_md("""## 3. Veredicto Arquitectonico para Despliegue Web
1. Motor Estimador de Cosecha:
   * Backend Python (FastAPI): LightGBM Regresion V2 serializado (.joblib) con MAE = 1,254.62 qq y MAPE = 45.20%.
   * Frontend Serverless (JavaScript): Regresion Lineal V3 calculada directamente en navegador:
     Produccion = Math.expm1(3.5059 + 0.9667 * Math.log1p(superficieHa))
2. Motor Clasificador Estacional:
   * Backend Python: CatBoost Clasificador V2 (.joblib) con ROC-AUC = 0.8679, Recall = 85.71% y soporte nativo de departamentos.""")
    ]
}

for nombre_archivo, celdas in cuadernos.items():
    ruta_completa = DIR_SALIDA / nombre_archivo
    with open(ruta_completa, "w", encoding="utf-8") as f:
        json.dump(crear_notebook(celdas), f, indent=1, ensure_ascii=False)
    print(f"Generado exitosamente: {nombre_archivo}")

print("\nTodos los cuadernos .ipynb han sido generados en su formato correcto.")