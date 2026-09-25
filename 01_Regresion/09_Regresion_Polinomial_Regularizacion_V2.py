# ============================================================
# 09_Regresion_Polinomial_Regularizacion_V2.py
# Regresión Polinomial + Regularización (Ridge, Lasso, ElasticNet)
# Versión 2: Filtrado de outliers + Transformación logarítmica
# ============================================================
# MEJORAS RESPECTO A V1 (02_Regresion_Polinomial_Regularizacion.py):
#   [V2-1] Filtro de outliers (SUPERFICIE < 3000 Ha)
#   [V2-2] Comparación: modelos con datos crudos vs filtrados
#   [V2-3] Transformación logarítmica opcional para linealizar
#   [V2-4] Métricas completas: R², RMSE, MAE, MAPE, MSE
#   [V2-5] Selección automática de alpha con GridSearchCV
#   [V2-6] Guardado automático de imagen en ../03_imagenes/
#   [V2-7] Rutas robustas con pathlib (funciona desde cualquier cwd)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)

# ------------------------------------------------------------
# 0. Rutas robustas (relativas al script, no al cwd)
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_IMAGENES = BASE_DIR / "03_imagenes"
RUTA_DATOS = BASE_DIR / "datos"
RUTA_IMAGENES.mkdir(exist_ok=True)

# ------------------------------------------------------------
# 1. Cargar y filtrar datos
# ------------------------------------------------------------
data = pd.read_excel(RUTA_DATOS / '0901_Produccion_Arroz.xlsx')
data = data.dropna(subset=['SUPERFICIE (Ha)', 'PRODUCCION (qq)'])

UMBRAL_SUPERFICIE = 3000
data_filtrada = data[data['SUPERFICIE (Ha)'] < UMBRAL_SUPERFICIE].copy()

print(f"Registros originales : {len(data)}")
print(f"Registros filtrados  : {len(data_filtrada)}")
print(f"Outliers excluidos   : {len(data) - len(data_filtrada)}")
print("-" * 55)

# ------------------------------------------------------------
# 2. Función para construir y evaluar modelos
# ------------------------------------------------------------
GRADO = 4  # grado polinomial (bajo, para evitar overfitting extremo)

def construir_modelos():
    """Devuelve el diccionario de modelos con sus pipelines."""
    return {
        "Lineal": LinearRegression(),
        "Polinomial": make_pipeline(
            PolynomialFeatures(GRADO, include_bias=False),
            StandardScaler(),
            LinearRegression()
        ),
        "Ridge": make_pipeline(
            PolynomialFeatures(GRADO, include_bias=False),
            StandardScaler(),
            Ridge(alpha=1.0)
        ),
        "Lasso": make_pipeline(
            PolynomialFeatures(GRADO, include_bias=False),
            StandardScaler(),
            Lasso(alpha=0.1, max_iter=100000)
        ),
        "Elastic Net": make_pipeline(
            PolynomialFeatures(GRADO, include_bias=False),
            StandardScaler(),
            ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=100000)
        ),
    }


def evaluar_modelos(X, y, usar_log=False, etiqueta="Original"):
    """Entrena todos los modelos y devuelve métricas + predicciones en grid."""
    if usar_log:
        X_proc = np.log1p(X)
        y_proc = np.log1p(y)
    else:
        X_proc = X
        y_proc = y

    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y_proc, test_size=0.25, random_state=42
    )

    # Grid para dibujar curvas continuas
    X_grid = np.linspace(X_proc.min(), X_proc.max(), 300).reshape(-1, 1)

    resultados = []
    predicciones_grid = {}

    for nombre, modelo in construir_modelos().items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        y_grid = modelo.predict(X_grid)

        # Volver a escala original si aplicamos log
        if usar_log:
            y_test_real = np.expm1(y_test)
            y_pred_real = np.expm1(y_pred)
        else:
            y_test_real = y_test
            y_pred_real = y_pred

        resultados.append({
            "Escenario": etiqueta,
            "Modelo": nombre,
            "R²": r2_score(y_test_real, y_pred_real),
            "RMSE": root_mean_squared_error(y_test_real, y_pred_real),
            "MAE": mean_absolute_error(y_test_real, y_pred_real),
            "MAPE (%)": mean_absolute_percentage_error(y_test_real, y_pred_real) * 100,
            "MSE": mean_squared_error(y_test_real, y_pred_real),
        })

        predicciones_grid[nombre] = (X_grid, y_grid)

    return resultados, predicciones_grid, (X_test, y_test_real)


# ------------------------------------------------------------
# 3. Evaluar 4 escenarios
# ------------------------------------------------------------
X_all = data[['SUPERFICIE (Ha)']].values
y_all = data['PRODUCCION (qq)'].values

X_fil = data_filtrada[['SUPERFICIE (Ha)']].values
y_fil = data_filtrada['PRODUCCION (qq)'].values

todos_resultados = []
escenarios_grafica = {}

for etiqueta, X_use, y_use, usar_log in [
    ("A) Original",         X_all, y_all, False),
    ("B) Filtrado",         X_fil, y_fil, False),
    ("C) Original + Log",   X_all, y_all, True),
    ("D) Filtrado + Log",   X_fil, y_fil, True),
]:
    res, preds, datos_test = evaluar_modelos(X_use, y_use,
                                             usar_log=usar_log,
                                             etiqueta=etiqueta)
    todos_resultados.extend(res)
    escenarios_grafica[etiqueta] = (preds, datos_test, usar_log)

# ------------------------------------------------------------
# 4. Tabla comparativa
# ------------------------------------------------------------
df_resultados = pd.DataFrame(todos_resultados)
print("\nTABLA COMPARATIVA - REGRESIÓN POLINOMIAL V2")
print("=" * 90)
print(df_resultados.to_string(index=False))
print("=" * 90)

# Mejor modelo por escenario
print("\nMEJOR MODELO POR ESCENARIO (según R²):")
for esc in df_resultados["Escenario"].unique():
    sub = df_resultados[df_resultados["Escenario"] == esc]
    mejor = sub.loc[sub["R²"].idxmax()]
    print(f"  {esc:20s} → {mejor['Modelo']:12s} (R²={mejor['R²']:.4f}, "
          f"MAPE={mejor['MAPE (%)']:.2f}%)")

# ------------------------------------------------------------
# 5. Gráficas comparativas (2x2)
# ------------------------------------------------------------
colores = {
    "Lineal": "#C62828",
    "Polinomial": "#8E24AA",
    "Ridge": "#1565C0",
    "Lasso": "#2E7D32",
    "Elastic Net": "#EF6C00",
}

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
axes = axes.flatten()

for ax, (etiqueta, (preds, (X_test, y_test_real), usar_log)) in zip(
        axes, escenarios_grafica.items()):

    ax.scatter(X_test, y_test_real, color="black", s=18,
               label="Test", zorder=5)
    for nombre, (X_grid, y_grid) in preds.items():
        estilo = "--" if nombre == "Lineal" else "-"
        ax.plot(X_grid, y_grid, color=colores[nombre], linestyle=estilo,
                linewidth=2, label=nombre)
    ax.set_title(etiqueta, fontsize=11)
    ax.set_xlabel("Superficie (Ha)" if not usar_log else "log1p(Superficie)")
    ax.set_ylabel("Producción (qq)" if not usar_log else "log1p(Producción)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle("Regresión Polinomial y Regularización V2 - Comparación de Estrategias",
             fontsize=13, fontweight='bold')
plt.tight_layout()

# [V2-6] Guardado automático
ruta_img = RUTA_IMAGENES / "25_regresion_polinomial_v2.png"
plt.savefig(ruta_img, dpi=120, bbox_inches='tight')
print(f"\n✅ Imagen guardada en: {ruta_img}")
plt.show()

# ------------------------------------------------------------
# 6. Optimización de alpha con GridSearchCV (solo en mejor escenario)
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("OPTIMIZACIÓN DE HIPERPARÁMETROS (GridSearchCV)")
print("=" * 60)

# Tomamos el escenario D (Filtrado + Log) que suele dar mejor resultado
X_proc = np.log1p(X_fil)
y_proc = np.log1p(y_fil)
X_train, X_test, y_train, y_test = train_test_split(
    X_proc, y_proc, test_size=0.25, random_state=42
)

# Ridge optimizado
pipe_ridge = make_pipeline(
    PolynomialFeatures(GRADO, include_bias=False),
    StandardScaler(),
    Ridge()
)
param_ridge = {"ridge__alpha": [0.01, 0.1, 1, 10, 100]}
gs_ridge = GridSearchCV(pipe_ridge, param_ridge, cv=5,
                        scoring="r2", n_jobs=-1)
gs_ridge.fit(X_train, y_train)
print(f"Ridge    → mejor alpha = {gs_ridge.best_params_['ridge__alpha']}, "
      f"R²(cv) = {gs_ridge.best_score_:.4f}")

# Lasso optimizado
pipe_lasso = make_pipeline(
    PolynomialFeatures(GRADO, include_bias=False),
    StandardScaler(),
    Lasso(max_iter=100000)
)
param_lasso = {"lasso__alpha": [0.001, 0.01, 0.1, 1, 10]}
gs_lasso = GridSearchCV(pipe_lasso, param_lasso, cv=5,
                        scoring="r2", n_jobs=-1)
gs_lasso.fit(X_train, y_train)
print(f"Lasso    → mejor alpha = {gs_lasso.best_params_['lasso__alpha']}, "
      f"R²(cv) = {gs_lasso.best_score_:.4f}")

# ElasticNet optimizado
pipe_en = make_pipeline(
    PolynomialFeatures(GRADO, include_bias=False),
    StandardScaler(),
    ElasticNet(max_iter=100000)
)
param_en = {
    "elasticnet__alpha": [0.01, 0.1, 1, 10],
    "elasticnet__l1_ratio": [0.2, 0.5, 0.8]
}
gs_en = GridSearchCV(pipe_en, param_en, cv=5, scoring="r2", n_jobs=-1)
gs_en.fit(X_train, y_train)
print(f"Elastic  → {gs_en.best_params_}, R²(cv) = {gs_en.best_score_:.4f}")

print("\n✅ Script finalizado. Revisa la imagen 25 en 03_imagenes/")