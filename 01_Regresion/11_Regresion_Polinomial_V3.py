# ============================================================
# 11_Regresion_Polinomial_V3.py
# Modelo de Regresión Polinomial Regularizada Ridge V3
# Dataset: 0901_Produccion_Arroz_V2_Regresion.xlsx
# Imagen:  ../03_imagenes/29_regresion_polinomial_v3.png
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error
)

# 1. Configuración de rutas
BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_DATOS = BASE_DIR / "datos" / "0901_Produccion_Arroz_V2_Regresion.xlsx"
RUTA_IMG = BASE_DIR / "03_imagenes" / "29_regresion_polinomial_v3.png"
RUTA_IMG.parent.mkdir(exist_ok=True)

# 2. Carga y transformación
df = pd.read_excel(RUTA_DATOS)
X_raw = df[['SUPERFICIE (Ha)']].values
y_raw = df['PRODUCCION (qq)'].values

X_log = np.log1p(X_raw)
y_log = np.log1p(y_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X_log, y_log, test_size=0.25, random_state=42
)

# 3. Pipeline con Grado 2 (estable, sin oscilaciones de Runge)
pipeline_ridge = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])

param_grid = {'ridge__alpha': [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]}
grid = GridSearchCV(pipeline_ridge, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
best_alpha = grid.best_params_['ridge__alpha']

# 4. Evaluación en escala real
y_pred_log = best_model.predict(X_test)
y_pred_real = np.expm1(y_pred_log)
y_test_real = np.expm1(y_test)
X_test_real = np.expm1(X_test).flatten()

r2 = r2_score(y_test_real, y_pred_real)
mae = mean_absolute_error(y_test_real, y_pred_real)
rmse = root_mean_squared_error(y_test_real, y_pred_real)
mape = mean_absolute_percentage_error(y_test_real, y_pred_real) * 100

print("=" * 60)
print("REGRESIÓN POLINOMIAL RIDGE V3 (LOGARÍTMICA)")
print("=" * 60)
print(f"Alpha óptimo (GridSearchCV) : {best_alpha}")
print(f"R² Score                    : {r2:.4f}")
print(f"MAE (Error Absoluto)        : {mae:.2f} qq")
print(f"RMSE (Error Cuadrático)     : {rmse:.2f} qq")
print(f"MAPE (Error Relativo)       : {mape:.2f} %")
print("=" * 60)

# 5. Gráfico de dispersión y curva continua
X_grid_log = np.linspace(X_log.min(), X_log.max(), 300).reshape(-1, 1)
y_grid_log = best_model.predict(X_grid_log)
X_grid_real = np.expm1(X_grid_log).flatten()
y_grid_real = np.expm1(y_grid_log)

plt.figure(figsize=(10, 6))
plt.scatter(X_test_real, y_test_real, color='black', alpha=0.5, s=25, label='Datos Reales (Test)')
plt.plot(X_grid_real, y_grid_real, color='#1565C0', lw=2.5, label=f'Curva Ridge V3 (Grado 2, α={best_alpha})')
plt.title("Regresión Polinomial Ridge V3 - Producción de Arroz", fontsize=13, fontweight='bold')
plt.xlabel("Superficie (Ha)")
plt.ylabel("Producción (qq)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RUTA_IMG, dpi=130)
print(f"Gráfica guardada en: {RUTA_IMG}")
plt.show()