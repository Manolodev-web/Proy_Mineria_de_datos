import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from catboost import CatBoostRegressor

# 1. Cargar datos
df = pd.read_excel('../datos/0901_Produccion_Arroz.xlsx', sheet_name='Hoja1')

print("=" * 60)
print("GRADIENT BOOSTING - REGRESIÓN (XGBoost vs CatBoost)")
print("=" * 60)

# 2. Variables (Incluimos 'DEPARTAMENTO' como variable categórica)
X = df[['DEPARTAMENTO', 'SUPERFICIE (Ha)', 'RENDIMIENTO (Kg/Ha)']]
y = df['PRODUCCION (qq)']

# 3. Preparación diferente para cada modelo
# XGBoost necesita One-Hot Encoding
X_xgb = pd.get_dummies(X, columns=['DEPARTAMENTO'])
X_train_xgb, X_test_xgb, y_train, y_test = train_test_split(X_xgb, y, test_size=0.30, random_state=42)

# CatBoost usa los datos originales
X_train_cat, X_test_cat, _, _ = train_test_split(X, y, test_size=0.30, random_state=42)
cat_features = ['DEPARTAMENTO']

# 4. Entrenar Modelos
xgb_reg = xgb.XGBRegressor(n_estimators=100, random_state=42)
xgb_reg.fit(X_train_xgb, y_train)
pred_xgb = xgb_reg.predict(X_test_xgb)

cat_reg = CatBoostRegressor(iterations=100, cat_features=cat_features, verbose=0, random_state=42)
cat_reg.fit(X_train_cat, y_train)
pred_cat = cat_reg.predict(X_test_cat)

print(f"XGBoost  - R²: {r2_score(y_test, pred_xgb):.4f} | MAE: {mean_absolute_error(y_test, pred_xgb):.2f}")
print(f"CatBoost - R²: {r2_score(y_test, pred_cat):.4f} | MAE: {mean_absolute_error(y_test, pred_cat):.2f}")

# 5. Gráfico 1: Real vs Predicho (Imagen 17)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_test, pred_xgb, alpha=0.5, color='crimson', edgecolor='k')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
axes[0].set_title('XGBoost: Real vs Predicho')
axes[0].set_xlabel('Producción Real (qq)')
axes[0].set_ylabel('Producción Predicha (qq)')

axes[1].scatter(y_test, pred_cat, alpha=0.5, color='royalblue', edgecolor='k')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
axes[1].set_title('CatBoost: Real vs Predicho')
axes[1].set_xlabel('Producción Real (qq)')

plt.tight_layout()
plt.savefig("../03_imagenes/17_gb_reg_scatter.png", dpi=150)
plt.show()

# 6. Gráfico 2: Residuos (Imagen 18)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(y_test - pred_xgb, kde=True, ax=axes[0], color='crimson', bins=40)
axes[0].set_title('XGBoost: Residuos (Errores)')

sns.histplot(y_test - pred_cat, kde=True, ax=axes[1], color='royalblue', bins=40)
axes[1].set_title('CatBoost: Residuos (Errores)')

plt.tight_layout()
plt.savefig("../03_imagenes/18_gb_reg_residuos.png", dpi=150)
plt.show()