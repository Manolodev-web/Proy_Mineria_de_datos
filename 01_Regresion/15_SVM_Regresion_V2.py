# ============================================================
# 15_SVM_Regresion_V2.py
# Support Vector Regressor (SVR) con Transformación Logarítmica V2
# Dataset: 0901_Produccion_Arroz_V2_Regresion.xlsx
# Imagen:  ../03_imagenes/39_svm_reg_v2.png
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import (
    r2_score, mean_absolute_error, root_mean_squared_error,
    mean_absolute_percentage_error
)
import warnings
warnings.filterwarnings('ignore')

# 1. Pagur-ongan dagiti files
BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_DATOS = BASE_DIR / "datos" / "0901_Produccion_Arroz_V2_Regresion.xlsx"
RUTA_IMG = BASE_DIR / "03_imagenes" / "39_svm_reg_v2.png"
RUTA_IMG.parent.mkdir(exist_ok=True)

# 2. Panangikarga kadagiti nadalus a datos V2
df = pd.read_excel(RUTA_DATOS)
X_raw = df[['SUPERFICIE (Ha)']].values
y_raw = df['PRODUCCION (qq)'].values

# Transformasion log1p para iti X ken y
X_log = np.log1p(X_raw)
y_log = np.log1p(y_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X_log, y_log, test_size=0.25, random_state=42
)

# Standard scaling para iti SVR
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

print("=" * 65)
print("SUPPORT VECTOR REGRESSION (SVR) V2 (CON LOG1P Y ESCALADO)")
print("=" * 65)

# 3. Panagsapul kadagiti kasayaatan a parametro babaen ti GridSearchCV
param_grid = {
    'C': [1, 10, 50, 100, 500],
    'epsilon': [0.01, 0.05, 0.1, 0.2],
    'gamma': ['scale', 'auto', 0.01, 0.1]
}
cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=cv, scoring='r2', n_jobs=-1)
grid.fit(X_train_scaled, y_train)

best_svr = grid.best_estimator_

# 4. Panagipadto ken panangisubli iti orihinal a rukod (quintales)
y_pred_log = best_svr.predict(X_test_scaled)
y_pred_real = np.expm1(y_pred_log)
y_test_real = np.expm1(y_test)
X_test_real = np.expm1(X_test).flatten()

# Panangkuenta kadagiti rukod ti panagpatingga
r2 = r2_score(y_test_real, y_pred_real)
mae = mean_absolute_error(y_test_real, y_pred_real)
rmse = root_mean_squared_error(y_test_real, y_pred_real)
mape = mean_absolute_percentage_error(y_test_real, y_pred_real) * 100

print(f"Mejores Hiperparámetros : {grid.best_params_}")
print(f"R² Score en Quintales   : {r2:.4f}")
print(f"MAE (Error Absoluto)    : {mae:.2f} qq")
print(f"RMSE (Error Cuadrático) : {rmse:.2f} qq")
print(f"MAPE (Error Relativo)   : {mape:.2f} %")
print("=" * 65)

# 5. Panangidrowing kadagiti grapiko ti panangsukimat (Panel 2x2)
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
min_val = min(y_test_real.min(), y_pred_real.min())
max_val = max(y_test_real.max(), y_pred_real.max())
orden = np.argsort(X_test_real)

# 5.1 Real vs Predicho en Quintales
axes[0, 0].scatter(y_test_real, y_pred_real, color='darkviolet', alpha=0.5, edgecolor='k', s=35, label='Muestras Test')
axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Predicción Perfecta")
axes[0, 0].set_title(f"SVR V2: Real vs Predicho (R²={r2:.3f})", fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel("Producción Real (qq)")
axes[0, 0].set_ylabel("Producción Predicha (qq)")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 5.2 Curva de Ajuste SVR en Escala Real
X_grid_raw = np.linspace(X_raw.min(), X_raw.max(), 300).reshape(-1, 1)
X_grid_scaled = scaler_X.transform(np.log1p(X_grid_raw))
y_grid_real = np.expm1(best_svr.predict(X_grid_scaled))

axes[0, 1].scatter(X_test_real, y_test_real, color='darkviolet', alpha=0.4, s=25, label='Datos Reales')
axes[0, 1].plot(X_grid_raw, y_grid_real, color='black', lw=2.5, label='Curva SVR (Kernel RBF)')
axes[0, 1].set_title("Curva No Lineal SVR en Escala Real", fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel("Superficie (Ha)")
axes[0, 1].set_ylabel("Producción (qq)")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 5.3 Residuos
residuos = y_test_real - y_pred_real
sns.histplot(residuos, kde=True, ax=axes[1, 0], color='darkviolet', bins=30)
axes[1, 0].axvline(0, color='red', linestyle='--', lw=2)
axes[1, 0].set_title("Distribución de Residuos (Errores SVR)", fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel("Residuo: y_real - y_pred (qq)")
axes[1, 0].set_ylabel("Frecuencia")

# 5.4 Residuos vs Predicho
axes[1, 1].scatter(y_pred_real, residuos, color='darkviolet', alpha=0.5, edgecolor='k', s=30)
axes[1, 1].axhline(0, color='red', linestyle='--', lw=2)
axes[1, 1].set_title("Homocedasticidad: Residuos vs Predicciones", fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel("Producción Predicha (qq)")
axes[1, 1].set_ylabel("Residuo (qq)")
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle("Support Vector Regression V2 - Diagnóstico Completo", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_IMG, dpi=130)
print(f"\nGráfica guardada en: {RUTA_IMG}")
plt.show()