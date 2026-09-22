import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Cargar datos
df = pd.read_excel('../datos/0901_Produccion_Arroz.xlsx', sheet_name='Hoja1')

print("=" * 60)
print("SUPPORT VECTOR MACHINE (SVR) - REGRESIÓN (Producción)")
print("=" * 60)

# 2. Variables predictoras y objetivo
X = df[['SUPERFICIE (Ha)', 'RENDIMIENTO (Kg/Ha)']]
y = df['PRODUCCION (qq)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# 3. Escalar los datos (Paso obligatorio para SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Entrenar el modelo SVR
modelo_svr = SVR(kernel="rbf", C=10, gamma="scale")
modelo_svr.fit(X_train_scaled, y_train)

# 5. Predicciones y Métricas
pred_svr = modelo_svr.predict(X_test_scaled)

mae = mean_absolute_error(y_test, pred_svr)
rmse = np.sqrt(mean_squared_error(y_test, pred_svr))
r2 = r2_score(y_test, pred_svr)

print(f"R² Score : {r2:.4f}")
print(f"MAE      : {mae:.2f} qq")
print(f"RMSE     : {rmse:.2f} qq")

# 6. Gráfico: Real vs Predicho (Imagen 21)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, pred_svr, alpha=0.5, color='darkviolet', edgecolor='k')
minimo = min(y_test.min(), pred_svr.min())
maximo = max(y_test.max(), pred_svr.max())
plt.plot([minimo, maximo], [minimo, maximo], 'r--', lw=2, label="Predicción Perfecta")

plt.title("SVR: Producción Real vs Producción Predicha")
plt.xlabel("Producción Real (qq)")
plt.ylabel("Producción Predicha (qq)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../03_imagenes/21_svr_reg_scatter.png", dpi=150)
plt.show()