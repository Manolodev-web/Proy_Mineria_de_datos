# ============================================================
# 13_Boosting_Regresion_V2.py
# Ensambles Boosting (AdaBoost vs LightGBM) sin Fuga de Datos
# Dataset: 0901_Produccion_Arroz_V2_Regresion.xlsx
# Imagen:  ../03_imagenes/35_boosting_reg_v2.png
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import (
    r2_score, mean_absolute_error, root_mean_squared_error,
    mean_absolute_percentage_error
)
import warnings
warnings.filterwarnings('ignore')

# 1. Configuración de rutas
BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_DATOS = BASE_DIR / "datos" / "0901_Produccion_Arroz_V2_Regresion.xlsx"
RUTA_IMG = BASE_DIR / "03_imagenes" / "35_boosting_reg_v2.png"
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
y_test_real = np.expm1(y_test)

print("=" * 65)
print("ENSAMBLES BOOSTING REGRESIÓN V2 (ADABOOST vs LIGHTGBM)")
print("=" * 65)

# 3. Entrenamiento de Modelos
# 3.1 AdaBoost
ada_reg = AdaBoostRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
ada_reg.fit(X_train, y_train)
pred_ada_log = ada_reg.predict(X_test)
pred_ada_real = np.expm1(pred_ada_log)

# 3.2 LightGBM
lgb_reg = LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, num_leaves=15, random_state=42, verbose=-1)
lgb_reg.fit(X_train, y_train)
pred_lgb_log = lgb_reg.predict(X_test)
pred_lgb_real = np.expm1(pred_lgb_log)

# 4. Cálculo de Métricas en escala real (Quintales)
metricas_ada = {
    "R²": r2_score(y_test_real, pred_ada_real),
    "MAE": mean_absolute_error(y_test_real, pred_ada_real),
    "RMSE": root_mean_squared_error(y_test_real, pred_ada_real),
    "MAPE": mean_absolute_percentage_error(y_test_real, pred_ada_real) * 100
}

metricas_lgb = {
    "R²": r2_score(y_test_real, pred_lgb_real),
    "MAE": mean_absolute_error(y_test_real, pred_lgb_real),
    "RMSE": root_mean_squared_error(y_test_real, pred_lgb_real),
    "MAPE": mean_absolute_percentage_error(y_test_real, pred_lgb_real) * 100
}

print(f"{'Métrica':<15} | {'AdaBoost':<15} | {'LightGBM':<15}")
print("-" * 50)
print(f"{'R² Score':<15} | {metricas_ada['R²']:<15.4f} | {metricas_lgb['R²']:<15.4f}")
print(f"{'MAE (qq)':<15} | {metricas_ada['MAE']:<15.2f} | {metricas_lgb['MAE']:<15.2f}")
print(f"{'RMSE (qq)':<15} | {metricas_ada['RMSE']:<15.2f} | {metricas_lgb['RMSE']:<15.2f}")
print(f"{'MAPE (%)':<15} | {metricas_ada['MAPE']:<15.2f} | {metricas_lgb['MAPE']:<15.2f}")
print("=" * 65)

# 5. Gráficas Diagnósticas (Panel 2x2: Dispersión y Residuos)
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
min_val = min(y_test_real.min(), pred_ada_real.min(), pred_lgb_real.min())
max_val = max(y_test_real.max(), pred_ada_real.max(), pred_lgb_real.max())

# 5.1 Real vs Predicho AdaBoost
axes[0, 0].scatter(y_test_real, pred_ada_real, color='darkorange', alpha=0.5, edgecolor='k', s=30)
axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Predicción Perfecta")
axes[0, 0].set_title(f"AdaBoost V2: Real vs Predicho (R²={metricas_ada['R²']:.3f})", fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel("Producción Real (qq)")
axes[0, 0].set_ylabel("Producción Predicha (qq)")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 5.2 Real vs Predicho LightGBM
axes[0, 1].scatter(y_test_real, pred_lgb_real, color='teal', alpha=0.5, edgecolor='k', s=30)
axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Predicción Perfecta")
axes[0, 1].set_title(f"LightGBM V2: Real vs Predicho (R²={metricas_lgb['R²']:.3f})", fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel("Producción Real (qq)")
axes[0, 1].set_ylabel("Producción Predicha (qq)")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 5.3 Residuos AdaBoost
res_ada = y_test_real - pred_ada_real
sns.histplot(res_ada, kde=True, ax=axes[1, 0], color='darkorange', bins=30)
axes[1, 0].axvline(0, color='red', linestyle='--', lw=2)
axes[1, 0].set_title("Distribución de Residuos - AdaBoost", fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel("Residuo (qq)")
axes[1, 0].set_ylabel("Frecuencia")

# 5.4 Residuos LightGBM
res_lgb = y_test_real - pred_lgb_real
sns.histplot(res_lgb, kde=True, ax=axes[1, 1], color='teal', bins=30)
axes[1, 1].axvline(0, color='red', linestyle='--', lw=2)
axes[1, 1].set_title("Distribución de Residuos - LightGBM", fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel("Residuo (qq)")
axes[1, 1].set_ylabel("Frecuencia")

plt.suptitle("Ensambles Boosting V2 - Regresión de Producción (Sin Fuga)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_IMG, dpi=130)
print(f"\nGráfica guardada en: {RUTA_IMG}")
plt.show()