# ============================================================
# 08_Regresion_Lineal_V2.py
# Regresión Lineal Simple - Producción de Arroz
# Versión 2: Filtrado de outliers + Transformación logarítmica
# ============================================================
# MEJORAS RESPECTO A V1 (01_Regresion_Lineal.py):
#   [V2-1] Filtro de outliers (SUPERFICIE < 3000 Ha)
#   [V2-2] Comparación: modelo con datos crudos vs datos filtrados
#   [V2-3] Transformación logarítmica para linealizar la relación
#   [V2-4] Métricas completas (R², RMSE, MAE, MAPE, MSE)
#   [V2-5] Guardado automático de imagen en ../03_imagenes/
#   [V2-6] Rutas robustas con pathlib (funciona desde cualquier cwd)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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
# 1. Cargar datos
# ------------------------------------------------------------
data = pd.read_excel(RUTA_DATOS / '0901_Produccion_Arroz.xlsx')
data = data.dropna(subset=['SUPERFICIE (Ha)', 'PRODUCCION (qq)'])

# [V2-1] Filtro de outliers: excluir superficies extremas (>3000 Ha)
# Justificación: el caso de 8545 Ha sesga la pendiente del modelo.
UMBRAL_SUPERFICIE = 3000
data_filtrada = data[data['SUPERFICIE (Ha)'] < UMBRAL_SUPERFICIE].copy()

print(f"Registros originales : {len(data)}")
print(f"Registros filtrados  : {len(data_filtrada)}")
print(f"Outliers excluidos   : {len(data) - len(data_filtrada)}")
print("-" * 55)

# ------------------------------------------------------------
# 2. Función auxiliar para entrenar y evaluar
# ------------------------------------------------------------
def entrenar_evaluar(X, y, usar_log=False, nombre="Modelo"):
    """Entrena regresión lineal y devuelve modelo, métricas y datos de test."""
    if usar_log:
        # [V2-3] Transformación logarítmica
        X_proc = np.log1p(X)   # log(1+x) para evitar log(0)
        y_proc = np.log1p(y)
    else:
        X_proc = X
        y_proc = y

    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y_proc, test_size=0.3, random_state=42
    )

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    # Si usamos log, volvemos a escala original para reportar métricas reales
    if usar_log:
        y_test_real = np.expm1(y_test)
        y_pred_real = np.expm1(y_pred)
    else:
        y_test_real = y_test
        y_pred_real = y_pred

    metricas = {
        "Modelo"    : nombre,
        "R²"        : r2_score(y_test_real, y_pred_real),
        "RMSE"      : root_mean_squared_error(y_test_real, y_pred_real),
        "MAE"       : mean_absolute_error(y_test_real, y_pred_real),
        "MAPE (%)"  : mean_absolute_percentage_error(y_test_real, y_pred_real) * 100,
        "MSE"       : mean_squared_error(y_test_real, y_pred_real),
    }
    return reg, metricas, (X_test, y_test_real, y_pred_real, usar_log)


# ------------------------------------------------------------
# 3. Entrenar 4 variantes y comparar
# ------------------------------------------------------------
resultados = []
predicciones_grafica = {}

# --- Caso A: datos originales (sin filtro, sin log) ---
X_all = data[['SUPERFICIE (Ha)']].values
y_all = data['PRODUCCION (qq)'].values
_, met_a, pred_a = entrenar_evaluar(X_all, y_all, usar_log=False,
                                    nombre="A) Original")
resultados.append(met_a)
predicciones_grafica["A) Original"] = pred_a

# --- Caso B: datos filtrados (sin log) ---
X_fil = data_filtrada[['SUPERFICIE (Ha)']].values
y_fil = data_filtrada['PRODUCCION (qq)'].values
_, met_b, pred_b = entrenar_evaluar(X_fil, y_fil, usar_log=False,
                                    nombre="B) Filtrado")
resultados.append(met_b)
predicciones_grafica["B) Filtrado"] = pred_b

# --- Caso C: datos originales con log ---
_, met_c, pred_c = entrenar_evaluar(X_all, y_all, usar_log=True,
                                    nombre="C) Original + Log")
resultados.append(met_c)
predicciones_grafica["C) Original + Log"] = pred_c

# --- Caso D: datos filtrados con log ---
_, met_d, pred_d = entrenar_evaluar(X_fil, y_fil, usar_log=True,
                                    nombre="D) Filtrado + Log")
resultados.append(met_d)
predicciones_grafica["D) Filtrado + Log"] = pred_d

# ------------------------------------------------------------
# 4. Mostrar tabla de resultados
# ------------------------------------------------------------
df_resultados = pd.DataFrame(resultados)
print("\nTABLA COMPARATIVA - REGRESIÓN LINEAL V2")
print("=" * 70)
print(df_resultados.to_string(index=False))
print("=" * 70)

# ------------------------------------------------------------
# 5. Gráfica comparativa (2x2)
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
titulos = list(predicciones_grafica.keys())
colores = ["#C62828", "#2E7D32", "#1565C0", "#EF6C00"]

for ax, titulo, color in zip(axes, titulos, colores):
    X_test, y_test_real, y_pred_real, usar_log = predicciones_grafica[titulo]

    # Para graficar, reordenamos por X para que la línea salga bien
    orden = np.argsort(X_test.flatten())
    X_ord = X_test.flatten()[orden]
    y_ord = y_pred_real[orden]

    ax.scatter(X_test, y_test_real, color="red", s=15, alpha=0.6,
               label="Datos Reales")
    ax.plot(X_ord, y_ord, color=color, linewidth=2, label="Predicción")
    ax.set_title(titulo, fontsize=11)
    ax.set_xlabel("Superficie (Ha)" if not usar_log else "log1p(Superficie)")
    ax.set_ylabel("Producción (qq)" if not usar_log else "log1p(Producción)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle("Regresión Lineal V2 - Comparación de Estrategias\n(Producción de Arroz)",
             fontsize=13, fontweight='bold')
plt.tight_layout()

# [V2-5] Guardado automático de imagen
ruta_img = RUTA_IMAGENES / "24_regresion_lineal_v2.png"
plt.savefig(ruta_img, dpi=120, bbox_inches='tight')
print(f"\n✅ Imagen guardada en: {ruta_img}")
plt.show()