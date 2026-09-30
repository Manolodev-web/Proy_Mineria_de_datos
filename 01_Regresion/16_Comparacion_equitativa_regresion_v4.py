# ============================================================
# 16_Comparacion_Equitativa_Regresion_V4.py
# Comparación EQUITATIVA de los 8 modelos de regresión:
#   - Mismas variables para todos los modelos (2 escenarios)
#   - Validación cruzada repetida (5 pliegues x 5 repeticiones)
#   - Modelo de referencia (baseline) para medir el aporte real
# Dataset: 0901_Produccion_Arroz_V2_Regresion.xlsx
# Imagen:  ../03_imagenes/41_comparacion_equitativa_reg_v4.png
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import RepeatedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer, StandardScaler, OneHotEncoder, PolynomialFeatures
)
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import (
    r2_score, mean_absolute_error, root_mean_squared_error,
    mean_absolute_percentage_error
)
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# 1. Rutas
BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_DATOS = BASE_DIR / "datos" / "0901_Produccion_Arroz_V2_Regresion.xlsx"
RUTA_IMG = BASE_DIR / "03_imagenes" / "41_comparacion_equitativa_reg_v4.png"
RUTA_IMG.parent.mkdir(exist_ok=True)

# 2. Cargar datos V2 y transformar el objetivo (igual que en los scripts 10-15)
df = pd.read_excel(RUTA_DATOS)
COL_NUM = ['SUPERFICIE (Ha)']
COL_CAT = ['DEPARTAMENTO', 'CAMPAÑA AGRICOLA']
y = df['PRODUCCION (qq)'].values
y_log = np.log1p(y)

# 3. Escenarios de variables (TODOS los modelos usan las mismas)
ESCENARIOS = {
    'A: solo superficie': COL_NUM,
    'B: superficie + departamento + campaña': COL_NUM + COL_CAT,
}

print("=" * 75)
print("COMPARACIÓN EQUITATIVA DE REGRESIÓN V4 (CV 5x5, mismas variables)")
print("=" * 75)
print(f"Registros: {len(df)} | Pliegues: 5 x 5 repeticiones = 25 evaluaciones/modelo")


# 4. Preprocesamiento común: log1p + estandarización de la superficie
#    y One-Hot para las variables categóricas (solo en el escenario B)
def crear_preprocesador(columnas):
    transformadores = [
        ('num', Pipeline([
            ('log', FunctionTransformer(np.log1p)),
            ('esc', StandardScaler())
        ]), COL_NUM)
    ]
    cats = [c for c in columnas if c in COL_CAT]
    if cats:
        transformadores.append(
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cats)
        )
    return ColumnTransformer(transformadores)


# 5. Modelos con hiperparámetros FIJOS (los mismos para ambos escenarios)
def crear_modelos():
    return {
        'Baseline (media)': DummyRegressor(strategy='mean'),
        'Regresión lineal': LinearRegression(),
        'Polinomial Ridge': Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('ridge', Ridge(alpha=10))
        ]),
        'Árbol de decisión': DecisionTreeRegressor(
            max_depth=5, min_samples_leaf=10, random_state=42),
        'SVR (RBF)': SVR(kernel='rbf', C=10, epsilon=0.1),
        'AdaBoost': AdaBoostRegressor(
            n_estimators=100, learning_rate=0.05, random_state=42),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=4, num_leaves=15,
            random_state=42, verbose=-1),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42),
        'CatBoost': CatBoostRegressor(
            iterations=200, learning_rate=0.05, depth=4, random_seed=42,
            verbose=0, allow_writing_files=False),
    }


# 6. Validación cruzada repetida (métricas en quintales, como en scripts previos)
cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
resultados = {}

for nombre_esc, columnas in ESCENARIOS.items():
    print(f"\nEjecutando escenario {nombre_esc} ...")
    X = df[columnas].copy()
    acumulado = {m: [] for m in crear_modelos()}

    for idx_tr, idx_te in cv.split(X):
        X_tr, X_te = X.iloc[idx_tr], X.iloc[idx_te]
        y_tr_log = y_log[idx_tr]
        y_te_real = y[idx_te]

        prep = crear_preprocesador(columnas)
        Xp_tr = prep.fit_transform(X_tr)
        Xp_te = prep.transform(X_te)

        for nombre_mod, modelo in crear_modelos().items():
            modelo.fit(Xp_tr, y_tr_log)
            pred_log = modelo.predict(Xp_te)
            pred = np.expm1(pred_log)
            acumulado[nombre_mod].append([
                r2_score(y_te_real, pred),
                mean_absolute_error(y_te_real, pred),
                root_mean_squared_error(y_te_real, pred),
                mean_absolute_percentage_error(y_te_real, pred) * 100,
                r2_score(np.log1p(y_te_real), pred_log),
            ])

    filas = []
    for nombre_mod, vals in acumulado.items():
        v = np.array(vals)
        filas.append({
            'Modelo': nombre_mod,
            'R2_media': v[:, 0].mean(), 'R2_std': v[:, 0].std(),
            'MAE_media': v[:, 1].mean(), 'MAE_std': v[:, 1].std(),
            'RMSE_media': v[:, 2].mean(), 'RMSE_std': v[:, 2].std(),
            'MAPE_media': v[:, 3].mean(), 'MAPE_std': v[:, 3].std(),
            'R2log_media': v[:, 4].mean(), 'R2log_std': v[:, 4].std(),
        })
    resultados[nombre_esc] = pd.DataFrame(filas).set_index('Modelo')

# 7. Impresión de tablas
for nombre_esc, tabla in resultados.items():
    print("\n" + "=" * 75)
    print(f"ESCENARIO {nombre_esc}  (media ± desviación estándar en 25 pliegues)")
    print("=" * 75)
    print(f"{'Modelo':<20} | {'R² (qq)':<15} | {'R² (log)':<15} | {'MAE (qq)':<18} | {'MAPE (%)':<15}")
    print("-" * 95)
    for modelo, f in tabla.sort_values('R2_media', ascending=False).iterrows():
        print(f"{modelo:<20} | {f.R2_media:6.4f} ± {f.R2_std:5.4f} | "
              f"{f.R2log_media:6.4f} ± {f.R2log_std:5.4f} | "
              f"{f.MAE_media:8.1f} ± {f.MAE_std:6.1f} | "
              f"{f.MAPE_media:5.1f} ± {f.MAPE_std:4.1f}")

print("\n" + "=" * 75)
print("¿AYUDAN LAS VARIABLES EXTRA? (cambio de R² al pasar de A a B)")
print("=" * 75)
A, B = list(resultados.values())
print(f"{'Modelo':<20} | {'R² A':<8} | {'R² B':<8} | {'Δ R²':<8}")
print("-" * 52)
for modelo in A.index:
    print(f"{modelo:<20} | {A.loc[modelo, 'R2_media']:<8.4f} | "
          f"{B.loc[modelo, 'R2_media']:<8.4f} | "
          f"{B.loc[modelo, 'R2_media'] - A.loc[modelo, 'R2_media']:+.4f}")

mejor_A = A['R2_media'].idxmax()
mejor_B = B['R2_media'].idxmax()
print(f"\nMejor modelo escenario A: {mejor_A} | Mejor modelo escenario B: {mejor_B}")
print("Nota: si la diferencia entre dos modelos es menor que su desviación")
print("estándar, no puede afirmarse que uno sea realmente mejor que el otro.")
print("=" * 75)

# 8. Gráfica: R² y MAE (media ± desv. est.), escenarios A y B
orden = A.sort_values('R2_media', ascending=False).index.tolist()
x = np.arange(len(orden))
ancho = 0.38

fig, ejes = plt.subplots(1, 2, figsize=(16, 6))
for ax, (col, titulo, etiqueta) in zip(
        ejes, [('R2', 'R² en validación cruzada (mayor es mejor)', 'R²'),
               ('MAE', 'MAE en validación cruzada (menor es mejor)', 'MAE (qq)')]):
    ax.bar(x - ancho / 2, A.loc[orden, f'{col}_media'], ancho,
           yerr=A.loc[orden, f'{col}_std'], capsize=3,
           label='A: solo superficie', color='#4C72B0')
    ax.bar(x + ancho / 2, B.loc[orden, f'{col}_media'], ancho,
           yerr=B.loc[orden, f'{col}_std'], capsize=3,
           label='B: + departamento + campaña', color='#DD8452')
    ax.set_xticks(x)
    ax.set_xticklabels(orden, rotation=35, ha='right')
    ax.set_ylabel(etiqueta)
    ax.set_title(titulo, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

plt.suptitle('Comparación equitativa de regresión (CV 5x5, mismas variables para todos)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_IMG, dpi=130)
print(f"\nGráfica guardada en: {RUTA_IMG}")
plt.show()