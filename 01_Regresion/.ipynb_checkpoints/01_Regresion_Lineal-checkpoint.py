import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# 1. Leer el archivo Excel desde la carpeta de datos
data = pd.read_excel('../datos/0901_Produccion_Arroz.xlsx')

# Limpieza básica para evitar errores de celdas vacías
data = data.dropna(subset=['SUPERFICIE (Ha)', 'PRODUCCION (qq)'])

# 2. Definir las variables
X = data[['SUPERFICIE (Ha)']] # Variable independiente (Matriz)
y = data['PRODUCCION (qq)'].values # Variable dependiente (Vector)

# 3. Separar: 70% entrenamiento, 30% evaluación
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Entrenar el modelo
reg = LinearRegression()
reg.fit(X_train, y_train)

# 5. Predecir
y_pred = reg.predict(X_test)

# 6. Evaluación (Métricas de la exposición)
r_squared = reg.score(X_test, y_test)
rmse = root_mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100

print("Regresión Lineal Simple - Producción de Arroz")
print("R² (Calificación)  :", r_squared)
print("MAPE (% de error)  :", mape, "%")
print("MAE (Error abs)    :", mae)
print("RMSE (Error grande):", rmse)
print("MSE (Cuadrático)   :", mse)

# 7. Graficar
plt.scatter(X_test, y_test, color="red", label="Datos Reales")
plt.plot(X_test, y_pred, color="blue", label="Línea de Predicción")
plt.title("Producción de Arroz respecto a la Superficie")
plt.xlabel("Superficie (Ha)")
plt.ylabel("Producción (qq)")
plt.legend()
plt.show()