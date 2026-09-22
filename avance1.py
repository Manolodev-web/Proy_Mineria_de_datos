import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# Cargar la base de datos
df = pd.read_excel("0901_Produccion_Arroz.xlsx")

# ==========================================
# 1. MODELO DE CLASIFICACIÓN
# ==========================================
# Clasificar 'CAMPAÑA AGRICOLA' (verano = 0, invierno = 1)
data_cls = df[['SUPERFICIE (Ha)', 'PRODUCCION (qq)', 'RENDIMIENTO (Kg/Ha)', 'CAMPAÑA AGRICOLA']].dropna()
data_cls['CAMPAÑA AGRICOLA'] = data_cls['CAMPAÑA AGRICOLA'].map({'verano': 0, 'invierno': 1})

X_cls = data_cls[['SUPERFICIE (Ha)', 'PRODUCCION (qq)', 'RENDIMIENTO (Kg/Ha)']]
y_cls = data_cls['CAMPAÑA AGRICOLA']

# División de datos (80% entrenamiento, 20% prueba)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

# Entrenamiento y evaluación
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X_train_c, y_train_c)
y_pred_c = clf.predict(X_test_c)
print(f"Precisión Clasificación: {accuracy_score(y_test_c, y_pred_c):.2f}")

# Visualización
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=X_cls.columns, class_names=['Verano', 'Invierno'])
plt.title("Árbol de Clasificación - Campaña Agrícola")
plt.show()

# ==========================================
# 2. MODELO DE REGRESIÓN
# ==========================================
# Predecir 'PRODUCCION (qq)' basado en 'SUPERFICIE (Ha)'
# Filtrar atípicos para mejor ajuste gráfico (< 2000 Ha)
data_reg = df[df['SUPERFICIE (Ha)'] < 2000].dropna(subset=['SUPERFICIE (Ha)', 'PRODUCCION (qq)'])

X_reg = data_reg[['SUPERFICIE (Ha)']].values
y_reg = data_reg['PRODUCCION (qq)'].values

# División de datos
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Entrenamiento y evaluación
reg = DecisionTreeRegressor(max_depth=4, random_state=42)
reg.fit(X_train_r, y_train_r)

y_pred_r = reg.predict(X_test_r)
print(f"MSE Regresión: {mean_squared_error(y_test_r, y_pred_r):.2f}")
print(f"R^2 Regresión: {r2_score(y_test_r, y_pred_r):.2f}")

# Visualización
X_grid = np.arange(np.min(X_reg), np.max(X_reg), 0.5).reshape(-1, 1)
plt.figure(figsize=(10, 6))
plt.scatter(X_reg, y_reg, color='blue', label='Datos reales', alpha=0.5, s=15)
plt.plot(X_grid, reg.predict(X_grid), color='red', label='Predicción del modelo (Árbol)')
plt.xlabel('Superficie (Ha)')
plt.ylabel('Producción (qq)')
plt.title('Regresión: Producción vs Superficie')
plt.legend()
plt.show()