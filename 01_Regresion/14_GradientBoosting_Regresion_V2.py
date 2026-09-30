# ============================================================
# 14_GradientBoosting_Regresion_V2.py
# Gradient Boosting (XGBoost vs CatBoost) sin Fuga de Datos
# Dataset: 0901_Produccion_Arroz_V2_Regresion.xlsx
# Imagen:  ../03_imagenes/37_gb_reg_v2.png
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_absolute_error, root_mean_squared_error,
    mean_absolute_percentage_error
)
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# 1. Rutas
BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_DATOS = BASE_DIR / "datos" / "0901_Produccion_Arroz_V2_Regresion.xlsx"
RUTA_IMG = BASE_DIR / "03_imagenes" / "37_gb_reg_v2.png"
RUTA_IMG.parent.mkdir(exist_ok=True)

# 2. Cargar datos V2 (Sin fuga: Superficie, Departamento y Campaña)
df = pd.read_excel(RUTA_DATOS)
features = ['SUPERFICIE (Ha)', 'DEPARTAMENTO', 'CAMPAÑA AGRICOLA']
X = df[features].copy()
y = df['PRODUCCION (qq)'].values

# Transformación logarítmica de la variable objetivo
y_log = np.log1p(y)

print("=" * 65)
print("GRADIENT BOOSTING REGRESIÓN V2 (XGBOOST vs CATBOOST)")
print("=" * 65)

# 3. Preparación de datos según la arquitectura de cada algoritmo
# 3.1 XGBoost requiere One-Hot Encoding en variables categóricas
X_xgb = pd.get_dummies(X, columns=['DEPARTAMENTO', 'CAMPAÑA AGRICOLA'], drop_first=True)
X_xgb['SUPERFICIE (Ha)'] = np.log1p(X_xgb['SUPERFICIE (Ha)'])

# 3.2 CatBoost procesa variables categóricas nativamente
X_cat = X.copy()
X_cat['SUPERFICIE (Ha)'] = np.log1p(X_cat['SUPERFICIE (Ha)'])
cat_features = ['DEPARTAMENTO', 'CAMPAÑA AGRICOLA']

# División estratificada de particiones idénticas
X_tr_xgb, X_te_xgb, y_tr, y_te = train_test_split(X_xgb, y_log, test_size=0.25, random_state=42)
X_tr_cat, X_te_cat, _, _ = train_test_split(X_cat, y_log, test_size=0.25, random_state=42)

y_test_real = np.expm1(y_te)

# 4. Entrenamiento de Modelos
# 4.1 XGBoost
xgb_reg = xgb.XGBRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
xgb_reg.fit(X_tr_xgb, y_tr)
pred_xgb_log = xgb_reg.predict(X_te_xgb)
pred_xgb_real = np.expm1(pred_xgb_log)

# 4.2 CatBoost
cat_reg = CatBoostRegressor(
    iterations=200,
    learning_rate=0.05,
    depth=5,
    cat_features=cat_features,
    verbose=0,
    random_state=42
)
cat_reg.fit(X_tr_cat, y_tr)
pred_cat_log = cat_reg.predict(X_te_cat)
pred_cat_real = np.expm1(pred_cat_log)

# 5. Métricas en escala real (Quintales)
metricas_xgb = {
    "R²": r2_score(y_test_real, pred_xgb_real),
    "MAE": mean_absolute_error(y_test_real, pred_xgb_real),
    "RMSE": root_mean_squared_error(y_test_real, pred_xgb_real),
    "MAPE": mean_absolute_percentage_error(y_test_real, pred_xgb_real) * 100
}

metricas_cat = {
    "R²": r2_score(y_test_real, pred_cat_real),
    "MAE": mean_absolute_error(y_test_real, pred_cat_real),
    "RMSE": root_mean_squared_error(y_test_real, pred_cat_real),
    "MAPE": mean_absolute_percentage_error(y_test_real, pred_cat_real) * 100
}

print(f"{'Métrica':<15} | {'XGBoost':<15} | {'CatBoost':<15}")
print("-" * 50)
print(f"{'R² Score':<15} | {metricas_xgb['R²']:<15.4f} | {metricas_cat['R²']:<15.4f}")
print(f"{'MAE (qq)':<15} | {metricas_xgb['MAE']:<15.2f} | {metricas_cat['MAE']:<15.2f}")
print(f"{'RMSE (qq)':<15} | {metricas_xgb['RMSE']:<15.2f} | {metricas_cat['RMSE']:<15.2f}")
print(f"{'MAPE (%)':<15} | {metricas_xgb['MAPE']:<15.2f} | {metricas_cat['MAPE']:<15.2f}")
print("=" * 65)

# 6. Gráficas Diagnósticas (Panel 2x2: Dispersión y Residuos)
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
min_val = min(y_test_real.min(), pred_xgb_real.min(), pred_cat_real.min())
max_val = max(y_test_real.max(), pred_xgb_real.max(), pred_cat_real.max())

# 6.1 Real vs Predicho XGBoost
axes[0, 0].scatter(y_test_real, pred_xgb_real, color='crimson', alpha=0.5, edgecolor='k', s=30)
axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="Predicción Perfecta")
axes[0, 0].set_title(f"XGBoost V2: Real vs Predicho (R²={metricas_xgb['R²']:.3f})", fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel("Producción Real (qq)")
axes[0, 0].set_ylabel("Producción Predicha (qq)")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 6.2 Real vs Predicho CatBoost
axes[0, 1].scatter(y_test_real, pred_cat_real, color='royalblue', alpha=0.5, edgecolor='k', s=30)
axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="Predicción Perfecta")
axes[0, 1].set_title(f"CatBoost V2: Real vs Predicho (R²={metricas_cat['R²']:.3f})", fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel("Producción Real (qq)")
axes[0, 1].set_ylabel("Producción Predicha (qq)")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 6.3 Residuos XGBoost
res_xgb = y_test_real - pred_xgb_real
sns.histplot(res_xgb, kde=True, ax=axes[1, 0], color='crimson', bins=30)
axes[1, 0].axvline(0, color='black', linestyle='--', lw=2)
axes[1, 0].set_title("Distribución de Residuos - XGBoost", fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel("Residuo (qq)")
axes[1, 0].set_ylabel("Frecuencia")

# 6.4 Residuos CatBoost
res_cat = y_test_real - pred_cat_real
sns.histplot(res_cat, kde=True, ax=axes[1, 1], color='royalblue', bins=30)
axes[1, 1].axvline(0, color='black', linestyle='--', lw=2)
axes[1, 1].set_title("Distribución de Residuos - CatBoost", fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel("Residuo (qq)")
axes[1, 1].set_ylabel("Frecuencia")

plt.suptitle("Gradient Boosting V2 - Regresión con Región y Temporada (Sin Fuga)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_IMG, dpi=130)
print(f"\nGráfica guardada en: {RUTA_IMG}")
plt.show()