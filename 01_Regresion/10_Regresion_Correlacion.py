import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)

# ==========================================================
# 1. CARGAR LOS DATOS
# ==========================================================

data = pd.read_excel('../datos/0901_Produccion_Arroz.xlsx')

print("Datos originales:")
print(data.head())

# ==========================================================
# 2. LIMPIEZA DE DATOS
# ==========================================================

data = data.dropna(
    subset=['SUPERFICIE (Ha)', 'PRODUCCION (qq)']
)

print("\nCantidad de datos después de la limpieza:")
print(len(data))

# ==========================================================
# 3. DEFINIR LAS VARIABLES
# ==========================================================

X = data[['SUPERFICIE (Ha)']]
y = data['PRODUCCION (qq)']

# ==========================================================
# 4. CORRELACIÓN NUMÉRICA
# ==========================================================

correlacion = data['SUPERFICIE (Ha)'].corr(
    data['PRODUCCION (qq)']
)

print("\n========================================")
print("CORRELACIÓN DE PEARSON")
print("========================================")
print("Correlación:", correlacion)

# ==========================================================
# 5. SEPARAR DATOS PARA ENTRENAMIENTO Y PRUEBA
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

print("\nDatos de entrenamiento:", len(X_train))
print("Datos de prueba:", len(X_test))

# ==========================================================
# 6. CREAR Y ENTRENAR EL MODELO
# ==========================================================

reg = LinearRegression()

reg.fit(X_train, y_train)

# ==========================================================
# 7. REALIZAR PREDICCIONES
# ==========================================================

y_pred = reg.predict(X_test)

# ==========================================================
# 8. MÉTRICAS DE EVALUACIÓN
# ==========================================================

r_squared = reg.score(X_test, y_test)

rmse = root_mean_squared_error(
    y_test,
    y_pred
)

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

mape = mean_absolute_percentage_error(
    y_test,
    y_pred
) * 100

# ==========================================================
# 9. MOSTRAR RESULTADOS
# ==========================================================

print("\n========================================")
print("REGRESIÓN LINEAL SIMPLE")
print("========================================")

print("Intercepto:", reg.intercept_)
print("Pendiente:", reg.coef_[0])

print("R²:", r_squared)
print("MAPE:", mape, "%")
print("MAE:", mae)
print("RMSE:", rmse)
print("MSE:", mse)

# ==========================================================
# 10. ECUACIÓN DE REGRESIÓN
# ==========================================================

print("\nEcuación del modelo:")

print(
    f"Producción = {reg.intercept_:.2f} "
    f"+ ({reg.coef_[0]:.2f} × Superficie)"
)

# ==========================================================
# 11. GRÁFICO DE CORRELACIÓN
# ==========================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    data['SUPERFICIE (Ha)'],
    data['PRODUCCION (qq)']
)

plt.title(
    "Correlación entre Superficie y Producción de Arroz"
)

plt.xlabel("Superficie (Ha)")
plt.ylabel("Producción (qq)")

plt.grid(True)

plt.show()

# ==========================================================
# 12. GRÁFICO DE REGRESIÓN
# ==========================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    X_test,
    y_test,
    label="Datos reales"
)

plt.plot(
    X_test,
    y_pred,
    label="Predicción"
)

plt.title(
    "Regresión Lineal: Producción de Arroz"
)

plt.xlabel("Superficie (Ha)")
plt.ylabel("Producción (qq)")

plt.legend()
plt.grid(True)

plt.show()