# ============================================================
# 12_Arbol_Decision_Regresion_V2.py
# Árbol de Regresión de Producción sin Fuga de Datos (V2)
# Dataset: 0901_Produccion_Arroz_V2_Regresion.xlsx
# Imagen:  ../03_imagenes/33_dt_regresion_v2.png
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import (
    r2_score, mean_absolute_error, root_mean_squared_error,
    mean_absolute_percentage_error
)

# 1. Configuración de rutas
BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_DATOS = BASE_DIR / "datos" / "0901_Produccion_Arroz_V2_Regresion.xlsx"
RUTA_IMG = BASE_DIR / "03_imagenes" / "33_dt_regresion_v2.png"
RUTA_IMG.parent.mkdir(exist_ok=True)

# 2. Cargar datos V2 (Sin fuga: solo Superficie)
df = pd.read_excel(RUTA_DATOS)
X_raw = df[['SUPERFICIE (Ha)']].values
y_raw = df['PRODUCCION (qq)'].values

# Transformación logarítmica
X_log = np.log1p(X_raw)
y_log = np.log1p(y_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X_log, y_log, test_size=0.25, random_state=42
)

# 3. Optimización y Poda del Árbol con GridSearchCV
param_grid = {
    'max_depth': [2, 3, 4, 5, 6],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [2, 5, 10, 20]
}
cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=cv, scoring='r2', n_jobs=-1)
grid.fit(X_train, y_train)

best_dt = grid.best_estimator_

# 4. Predicciones y reversión a escala real (quintales)
y_pred_log = best_dt.predict(X_test)
y_pred_real = np.expm1(y_pred_log)
y_test_real = np.expm1(y_test)

# Métricas en quintales reales
r2 = r2_score(y_test_real, y_pred_real)
mae = mean_absolute_error(y_test_real, y_pred_real)
rmse = root_mean_squared_error(y_test_real, y_pred_real)
mape = mean_absolute_percentage_error(y_test_real, y_pred_real) * 100

print("=" * 60)
print("ÁRBOL DE DECISIÓN REGRESIÓN V2 (SIN FUGA DE DATOS)")
print("=" * 60)
print(f"Mejores Hiperparámetros : {grid.best_params_}")
print(f"R² Score en Quintales   : {r2:.4f}")
print(f"MAE (Error Absoluto)    : {mae:.2f} qq")
print(f"RMSE (Error Cuadrático) : {rmse:.2f} qq")
print(f"MAPE (Error Relativo)   : {mape:.2f} %")
print("=" * 60)

# 5. Gráficas diagnósticas (Dispersión y Estructura Podada)
fig = plt.figure(figsize=(16, 7))

# 5.1 Dispersión Real vs Predicho
ax1 = plt.subplot(1, 2, 1)
ax1.scatter(y_test_real, y_pred_real, color='purple', alpha=0.5, edgecolor='k', s=35, label='Muestras de Test')
min_val = min(y_test_real.min(), y_pred_real.min())
max_val = max(y_test_real.max(), y_pred_real.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción Perfecta')
ax1.set_title("Árbol de Regresión V2: Real vs Predicho (qq)", fontsize=12, fontweight='bold')
ax1.set_xlabel("Producción Real (qq)")
ax1.set_ylabel("Producción Predicha (qq)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 5.2 Estructura del Árbol Podado
ax2 = plt.subplot(1, 2, 2)
plot_tree(best_dt, feature_names=['log1p(Superficie)'], filled=True, rounded=True, fontsize=8, ax=ax2)
ax2.set_title(f"Estructura Podada (max_depth={grid.best_params_['max_depth']})", fontsize=12, fontweight='bold')

plt.suptitle("Árbol de Decisión Regresión V2 - Predicción de Producción", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_IMG, dpi=130)
print(f"\nGráfica guardada en: {RUTA_IMG}")
plt.show()