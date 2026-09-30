# ============================================================
# 17_Comparacion_Equitativa_Clasificacion_V4.py
# Comparación EQUITATIVA de los modelos de clasificación:
#   - Mismas variables para todos los modelos (2 escenarios)
#   - Validación cruzada estratificada repetida (5 x 5)
#   - Modelo de referencia (baseline) para medir el aporte real
# Dataset: 0901_Produccion_Arroz_V3_Clasificacion.xlsx (balanceado)
# Imagen:  ../03_imagenes/42_comparacion_equitativa_clf_v4.png
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# 1. Rutas
BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_DATOS = BASE_DIR / "datos" / "0901_Produccion_Arroz_V3_Clasificacion.xlsx"
RUTA_IMG = BASE_DIR / "03_imagenes" / "42_comparacion_equitativa_clf_v4.png"
RUTA_IMG.parent.mkdir(exist_ok=True)

# 2. Cargar datos V3 (mismo criterio que el script 15: invierno = 1)
df = pd.read_excel(RUTA_DATOS)
df['target'] = df['CAMPAÑA AGRICOLA'].str.lower().map({'verano': 0, 'invierno': 1})
y = df['target'].values

COL_NUM = ['SUPERFICIE (Ha)', 'PRODUCCION (qq)', 'RENDIMIENTO (Kg/Ha)']
COL_CAT = ['DEPARTAMENTO']

# 3. Escenarios de variables (TODOS los modelos usan las mismas)
ESCENARIOS = {
    'A: variables numéricas': COL_NUM,
    'B: numéricas + departamento': COL_NUM + COL_CAT,
}

print("=" * 75)
print("COMPARACIÓN EQUITATIVA DE CLASIFICACIÓN V4 (CV 5x5, mismas variables)")
print("=" * 75)
print(f"Registros: {len(df)} (invierno={int(y.sum())}, verano={int((1 - y).sum())})")
print("Clase positiva: invierno | 5 pliegues x 5 repeticiones = 25 evaluaciones/modelo")


# 4. Preprocesamiento común: log1p + estandarización de las variables
#    numéricas y One-Hot del departamento (solo en el escenario B)
def crear_preprocesador(columnas):
    transformadores = [
        ('num', Pipeline([
            ('log', FunctionTransformer(np.log1p)),
            ('esc', StandardScaler())
        ]), COL_NUM)
    ]
    if 'DEPARTAMENTO' in columnas:
        transformadores.append(
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             COL_CAT)
        )
    return ColumnTransformer(transformadores)


# 5. Modelos con hiperparámetros FIJOS (los mismos para ambos escenarios)
def crear_modelos():
    return {
        'Baseline (azar)': DummyClassifier(strategy='stratified', random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=7),
        'Regresión logística': LogisticRegression(max_iter=1000),
        'Naive Bayes': GaussianNB(),
        'Árbol de decisión': DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=5, random_state=42),
        'SVC (RBF)': SVC(kernel='rbf', C=1, probability=True, random_state=42),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=100, learning_rate=0.05, random_state=42),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=4, num_leaves=15,
            random_state=42, verbose=-1),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=4,
            eval_metric='logloss', random_state=42),
        'CatBoost': CatBoostClassifier(
            iterations=200, learning_rate=0.05, depth=4, random_seed=42,
            verbose=0, allow_writing_files=False),
    }


# 6. Validación cruzada estratificada repetida
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
resultados = {}

for nombre_esc, columnas in ESCENARIOS.items():
    print(f"\nEjecutando escenario {nombre_esc} ...")
    X = df[columnas].copy()
    acumulado = {m: [] for m in crear_modelos()}

    for idx_tr, idx_te in cv.split(X, y):
        X_tr, X_te = X.iloc[idx_tr], X.iloc[idx_te]
        y_tr, y_te = y[idx_tr], y[idx_te]

        prep = crear_preprocesador(columnas)
        Xp_tr = prep.fit_transform(X_tr)
        Xp_te = prep.transform(X_te)

        for nombre_mod, modelo in crear_modelos().items():
            modelo.fit(Xp_tr, y_tr)
            pred = modelo.predict(Xp_te)
            proba = modelo.predict_proba(Xp_te)[:, 1]
            acumulado[nombre_mod].append([
                accuracy_score(y_te, pred),
                precision_score(y_te, pred, zero_division=0),
                recall_score(y_te, pred, zero_division=0),
                f1_score(y_te, pred, zero_division=0),
                roc_auc_score(y_te, proba),
            ])

    filas = []
    for nombre_mod, vals in acumulado.items():
        v = np.array(vals)
        filas.append({
            'Modelo': nombre_mod,
            'ACC_media': v[:, 0].mean(), 'ACC_std': v[:, 0].std(),
            'PRE_media': v[:, 1].mean(), 'PRE_std': v[:, 1].std(),
            'REC_media': v[:, 2].mean(), 'REC_std': v[:, 2].std(),
            'F1_media': v[:, 3].mean(), 'F1_std': v[:, 3].std(),
            'AUC_media': v[:, 4].mean(), 'AUC_std': v[:, 4].std(),
        })
    resultados[nombre_esc] = pd.DataFrame(filas).set_index('Modelo')

# 7. Impresión de tablas
for nombre_esc, tabla in resultados.items():
    print("\n" + "=" * 100)
    print(f"ESCENARIO {nombre_esc}  (media ± desviación estándar en 25 pliegues)")
    print("=" * 100)
    print(f"{'Modelo':<20} | {'Accuracy':<14} | {'Recall inv.':<14} | "
          f"{'Precision inv.':<14} | {'F1':<14} | {'ROC-AUC':<14}")
    print("-" * 100)
    for modelo, f in tabla.sort_values('F1_media', ascending=False).iterrows():
        print(f"{modelo:<20} | {f.ACC_media:.3f} ± {f.ACC_std:.3f} | "
              f"{f.REC_media:.3f} ± {f.REC_std:.3f} | "
              f"{f.PRE_media:.3f} ± {f.PRE_std:.3f}  | "
              f"{f.F1_media:.3f} ± {f.F1_std:.3f} | "
              f"{f.AUC_media:.3f} ± {f.AUC_std:.3f}")

print("\n" + "=" * 75)
print("¿AYUDA EL DEPARTAMENTO? (cambio de F1 al pasar de A a B)")
print("=" * 75)
A, B = list(resultados.values())
print(f"{'Modelo':<20} | {'F1 A':<8} | {'F1 B':<8} | {'Δ F1':<8}")
print("-" * 52)
for modelo in A.index:
    print(f"{modelo:<20} | {A.loc[modelo, 'F1_media']:<8.4f} | "
          f"{B.loc[modelo, 'F1_media']:<8.4f} | "
          f"{B.loc[modelo, 'F1_media'] - A.loc[modelo, 'F1_media']:+.4f}")

print(f"\nMejor modelo escenario A: {A['F1_media'].idxmax()} | "
      f"Mejor modelo escenario B: {B['F1_media'].idxmax()}")
print("Nota: si la diferencia entre dos modelos es menor que su desviación")
print("estándar, no puede afirmarse que uno sea realmente mejor que el otro.")
print("=" * 75)

# 8. Gráfica: F1 y ROC-AUC (media ± desv. est.), escenarios A y B
orden = B.sort_values('F1_media', ascending=False).index.tolist()
x = np.arange(len(orden))
ancho = 0.38

fig, ejes = plt.subplots(1, 2, figsize=(16, 6))
for ax, (col, titulo, etiqueta) in zip(
        ejes, [('F1', 'F1 (invierno) en validación cruzada', 'F1'),
               ('AUC', 'ROC-AUC en validación cruzada', 'ROC-AUC')]):
    ax.bar(x - ancho / 2, A.loc[orden, f'{col}_media'], ancho,
           yerr=A.loc[orden, f'{col}_std'], capsize=3,
           label='A: variables numéricas', color='#4C72B0')
    ax.bar(x + ancho / 2, B.loc[orden, f'{col}_media'], ancho,
           yerr=B.loc[orden, f'{col}_std'], capsize=3,
           label='B: + departamento', color='#DD8452')
    ax.set_xticks(x)
    ax.set_xticklabels(orden, rotation=35, ha='right')
    ax.set_ylabel(etiqueta)
    ax.set_ylim(0, 1)
    ax.set_title(titulo, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='lower right')

plt.suptitle('Comparación equitativa de clasificación (CV 5x5, mismas variables para todos)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_IMG, dpi=130)
print(f"\nGráfica guardada en: {RUTA_IMG}")
plt.show()