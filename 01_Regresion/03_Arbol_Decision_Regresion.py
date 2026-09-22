import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Cargar datos
df = pd.read_excel('../datos/0901_Produccion_Arroz.xlsx', sheet_name='Hoja1')

# 2. Variables para Regresión
X = df[['SUPERFICIE (Ha)', 'RENDIMIENTO (Kg/Ha)']]
y = df['PRODUCCION (qq)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print("=" * 60)
print("ÁRBOL DE DECISIÓN - REGRESIÓN (Producción de Arroz)")
print("=" * 60)

# 3. Entrenar modelo
arbol_reg = DecisionTreeRegressor(max_depth=4, random_state=42)
arbol_reg.fit(X_train, y_train)

pred_test = arbol_reg.predict(X_test)
r2 = r2_score(y_test, pred_test)
mae = mean_absolute_error(y_test, pred_test)

print(f"R² Test: {r2:.4f} | MAE: {mae:.2f} qq")

# 4. Gráfico 1: Estructura del Árbol de Regresión (Imagen 07)
plt.figure(figsize=(16, 8))
plot_tree(arbol_reg, feature_names=X.columns, filled=True, rounded=True, fontsize=9)
plt.title("Árbol de Decisión - Regresión de Producción")
plt.tight_layout()
plt.savefig("../03_imagenes/07_dt_regresion_arbol.png", dpi=150, bbox_inches="tight")
plt.show()

# 5. Gráfico 2: Real vs Predicho (Imagen 08)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, pred_test, alpha=0.6, color="purple")
minimo = min(y_test.min(), pred_test.min())
maximo = max(y_test.max(), pred_test.max())
plt.plot([minimo, maximo], [minimo, maximo], linestyle="--", color="red", linewidth=2)
plt.xlabel("Producción Real (qq)")
plt.ylabel("Producción Predicha (qq)")
plt.title("Árbol de Regresión: Real vs Predicho")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../03_imagenes/08_dt_regresion_scatter.png", dpi=150)
plt.show()