# ============================================================
# 10_Regresion_Lineal_V3.py
# Modelo de Regresión Lineal Logarítmico (Versión Final)
# Dataset: 0901_Produccion_Arroz_V2_Regresion.xlsx
# Imagen:  ../03_imagenes/28_regresion_lineal_v3.png
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error
)

# 1. Configuración de rutas
BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_DATOS = BASE_DIR / "datos" / "0901_Produccion_Arroz_V2_Regresion.xlsx"
RUTA_IMG = BASE_DIR / "03_imagenes" / "28_regresion_lineal_v3.png"
RUTA_IMG.parent.mkdir(exist_ok=True)

# 2. Carga de datos V2
df = pd.read_excel(RUTA_DATOS)
X_raw = df[['SUPERFICIE (Ha)']].values
y_raw = df['PRODUCCION (qq)'].values

# 3. Transformación logarítmica (log1p para proteger ceros)
X_log = np.log1p(X_raw)
y_log = np.log1p(y_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X_log, y_log, test_size=0.25, random_state=42
)

# 4. Entrenamiento del modelo
reg = LinearRegression()
reg.fit(X_train, y_train)

# 5. Predicción y reversión a escala real (quintales)
y_pred_log = reg.predict(X_test)
y_pred_real = np.expm1(y_pred_log)
y_test_real = np.expm1(y_test)
X_test_real = np.expm1(X_test).flatten()

# 6. Cálculo de métricas sobre la escala real
r2 = r2_score(y_test_real, y_pred_real)
mae = mean_absolute_error(y_test_real, y_pred_real)
rmse = root_mean_squared_error(y_test_real, y_pred_real)
mape = mean_absolute_percentage_error(y_test_real, y_pred_real) * 100

b0 = reg.intercept_
b1 = reg.coef_[0]

print("=" * 60)
print("REGRESIÓN LINEAL V3 - EVALUACIÓN EN QUINTALES REALES")
print("=" * 60)
print(f"R² Score              : {r2:.4f}")
print(f"MAE (Error Absoluto)  : {mae:.2f} qq")
print(f"RMSE (Error Cuad.)    : {rmse:.2f} qq")
print(f"MAPE (Error Relativo) : {mape:.2f} %")
print("-" * 60)
print("FÓRMULA MATEMÁTICA PARA LA WEB:")
print(f"ln(Produccion + 1) = {b0:.4f} + {b1:.4f} * ln(Superficie + 1)")
print("Produccion = exp(ln(Produccion + 1)) - 1")
print("=" * 60)

# 7. Visualización diagnóstica (Panel 2x2)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
orden = np.argsort(X_test_real)

# Panel 1: Ajuste en escala real
axes[0, 0].scatter(X_test_real, y_test_real, color='crimson', alpha=0.5, s=25, label='Datos Reales (Test)')
axes[0, 0].plot(X_test_real[orden], y_pred_real[orden], color='navy', lw=2.5, label='Curva del Modelo')
axes[0, 0].set_title("Ajuste en Escala Real (Quintales)", fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel("Superficie (Ha)")
axes[0, 0].set_ylabel("Producción (qq)")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Panel 2: Linealidad en escala logarítmica
axes[0, 1].scatter(X_test, y_test, color='darkorange', alpha=0.5, s=25, label='Datos log1p (Test)')
axes[0, 1].plot(X_test.flatten()[orden], y_pred_log[orden], color='navy', lw=2.5, label=f'Recta: y = {b0:.2f} + {b1:.2f}x')
axes[0, 1].set_title("Linealidad en Espacio Logarítmico (log1p)", fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel("log1p(Superficie)")
axes[0, 1].set_ylabel("log1p(Producción)")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Panel 3: Distribución de residuos
residuos = y_test_real - y_pred_real
axes[1, 0].hist(residuos, bins=30, color='royalblue', edgecolor='k', alpha=0.7)
axes[1, 0].axvline(0, color='red', linestyle='--', lw=2)
axes[1, 0].set_title("Distribución de Residuos (Escala Real)", fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel("Error (y_real - y_pred) [qq]")
axes[1, 0].set_ylabel("Frecuencia")
axes[1, 0].grid(True, alpha=0.3)

# Panel 4: Homocedasticidad
axes[1, 1].scatter(y_pred_real, residuos, color='purple', alpha=0.5, s=25)
axes[1, 1].axhline(0, color='red', linestyle='--', lw=2)
axes[1, 1].set_title("Residuos vs Predicción", fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel("Producción Predicha (qq)")
axes[1, 1].set_ylabel("Residuo (qq)")
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle("Regresión Lineal V3 - Diagnóstico Completo", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_IMG, dpi=130)
print(f"\nGráfica guardada en: {RUTA_IMG}")
plt.show()