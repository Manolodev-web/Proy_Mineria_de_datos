import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Cargar datos
df = pd.read_excel('../datos/0901_Produccion_Arroz.xlsx', sheet_name='Hoja1')

print("=" * 60)
print("ENSAMBLES BOOSTING - REGRESIÓN (Producción de Arroz)")
print("=" * 60)

# 2. Variables predictoras y objetivo
X = df[['SUPERFICIE (Ha)', 'RENDIMIENTO (Kg/Ha)']]
y = df['PRODUCCION (qq)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# 3. Entrenar Modelos
ada_reg = AdaBoostRegressor(n_estimators=100, random_state=42)
ada_reg.fit(X_train, y_train)
pred_ada = ada_reg.predict(X_test)

lgb_reg = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1)
lgb_reg.fit(X_train, y_train)
pred_lgb = lgb_reg.predict(X_test)

# Métricas
print(f"AdaBoost - R²: {r2_score(y_test, pred_ada):.4f} | MAE: {mean_absolute_error(y_test, pred_ada):.2f}")
print(f"LightGBM - R²: {r2_score(y_test, pred_lgb):.4f} | MAE: {mean_absolute_error(y_test, pred_lgb):.2f}")

# 4. Gráfico 1: Real vs Predicho Comparativo (Imagen 13)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_test, pred_ada, alpha=0.5, color='orange', edgecolor='k')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_title('AdaBoost: Real vs Predicho')
axes[0].set_xlabel('Producción Real (qq)')
axes[0].set_ylabel('Producción Predicha (qq)')

axes[1].scatter(y_test, pred_lgb, alpha=0.5, color='teal', edgecolor='k')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_title('LightGBM: Real vs Predicho')
axes[1].set_xlabel('Producción Real (qq)')

plt.tight_layout()
plt.savefig("../03_imagenes/13_boosting_reg_scatter.png", dpi=150)
plt.show()

# 5. Gráfico 2: Distribución de Residuos (Imagen 14)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(y_test - pred_ada, kde=True, ax=axes[0], color='orange', bins=40)
axes[0].set_title('AdaBoost: Residuos (Errores)')
axes[0].set_xlabel('Residuo (qq)')

sns.histplot(y_test - pred_lgb, kde=True, ax=axes[1], color='teal', bins=40)
axes[1].set_title('LightGBM: Residuos (Errores)')
axes[1].set_xlabel('Residuo (qq)')

plt.tight_layout()
plt.savefig("../03_imagenes/14_boosting_reg_residuos.png", dpi=150)
plt.show()