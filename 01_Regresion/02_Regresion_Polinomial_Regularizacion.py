import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# 1. Cargar datos reales
data = pd.read_excel('../datos/0901_Produccion_Arroz.xlsx')
data = data.dropna(subset=['SUPERFICIE (Ha)', 'PRODUCCION (qq)'])

X = data[['SUPERFICIE (Ha)']].values
y = data['PRODUCCION (qq)'].values

# 2. Separar datos de entrenamiento y evaluación
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.25, random_state=42)

# Valores para dibujar las curvas continuas
X_grid = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

# Usamos un grado menor al sintético (ej. 4) para evitar sobreajuste extremo en datos reales
grado = 4 

# 3. Definir los modelos (Pipeline con escalado)
modelos = {
    "Lineal": LinearRegression(),
    "Polinomial": make_pipeline(PolynomialFeatures(grado, include_bias=False), StandardScaler(), LinearRegression()),
    "Ridge": make_pipeline(PolynomialFeatures(grado, include_bias=False), StandardScaler(), Ridge(alpha=1.0)),
    "Lasso": make_pipeline(PolynomialFeatures(grado, include_bias=False), StandardScaler(), Lasso(alpha=0.1, max_iter=100000)),
    "Elastic Net": make_pipeline(PolynomialFeatures(grado, include_bias=False), StandardScaler(), ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=100000))
}

# 4. Entrenamiento y Evaluación
resultados, predicciones = {}, {}

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_eval)
    resultados[nombre] = {"MSE": mean_squared_error(y_eval, y_pred), "R2": r2_score(y_eval, y_pred)}
    predicciones[nombre] = modelo.predict(X_grid)

# 5. Mostrar resultados
print("\nRESULTADOS DE LOS MODELOS (Arroz)")
print("-" * 55)
for nombre, metricas in resultados.items():
    print(f"{nombre:12s} | MSE = {metricas['MSE']:.2e} | R² = {metricas['R2']:.4f}")

# 6. Gráfica Comparativa
colores = {"Lineal": "#C62828", "Polinomial": "#8E24AA", "Ridge": "#1565C0", "Lasso": "#2E7D32", "Elastic Net": "#EF6C00"}
plt.figure(figsize=(10, 6))

plt.scatter(X_train, y_train, color="black", s=25, label="Entrenamiento")
plt.scatter(X_eval, y_eval, color="#546E7A", marker="D", s=25, label="Evaluación", zorder=5)

for nombre, y_grid in predicciones.items():
    estilo = "--" if nombre == "Lineal" else "-"
    plt.plot(X_grid, y_grid, color=colores[nombre], linestyle=estilo, linewidth=2, label=nombre)

plt.title("Regresión Polinomial y Regularización - Producción de Arroz")
plt.xlabel("Superficie (Ha)")
plt.ylabel("Producción (qq)")
plt.legend()
plt.tight_layout()
plt.show()