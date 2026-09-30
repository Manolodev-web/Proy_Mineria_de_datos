# ============================================================
# 18_Distribucion_Real_Clasificacion_V4.py
# ¿Qué pasa con la clasificación en la distribución REAL (90/10)?
#   - Se evalúa SIEMPRE sobre datos con la proporción original
#     (verano ~90 %, invierno ~10 %), sin balancear el conjunto de prueba
#   - Se comparan 3 estrategias de entrenamiento:
#       1. Sin balanceo
#       2. Pesos de clase (no descarta ningún registro)
#       3. Submuestreo (igual que la versión V3, solo en entrenamiento)
#   - Métricas adecuadas para clases desbalanceadas: PR-AUC, precisión, recall
# Dataset: 0901_Produccion_Arroz.xlsx (original, sin balancear)
# Imagen:  ../03_imagenes/43_distribucion_real_clf_v4.png
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score
)
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# 1. Rutas
BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_DATOS = BASE_DIR / "datos" / "0901_Produccion_Arroz.xlsx"
RUTA_IMG = BASE_DIR / "03_imagenes" / "43_distribucion_real_clf_v4.png"
RUTA_IMG.parent.mkdir(exist_ok=True)

# 2. Cargar datos originales (mismo filtrado de nulos que preparar_datos_v2_v3.py)
df = pd.read_excel(RUTA_DATOS)
df = df.dropna(subset=["SUPERFICIE (Ha)", "PRODUCCION (qq)", "CAMPAÑA AGRICOLA"]).copy()
df['target'] = df['CAMPAÑA AGRICOLA'].str.lower().map({'verano': 0, 'invierno': 1})
y = df['target'].values

COL_NUM = ['SUPERFICIE (Ha)', 'PRODUCCION (qq)', 'RENDIMIENTO (Kg/Ha)']
COL_CAT = ['DEPARTAMENTO']
X = df[COL_NUM + COL_CAT].copy()
prevalencia = y.mean()

print("=" * 80)
print("CLASIFICACIÓN EN LA DISTRIBUCIÓN REAL V4 (prueba SIN balancear)")
print("=" * 80)
print(f"Registros: {len(df)} | invierno={int(y.sum())} ({prevalencia:.1%}) | "
      f"verano={int((1 - y).sum())} ({1 - prevalencia:.1%})")
print(f"Un modelo que siempre responde 'verano' tendría {1 - prevalencia:.1%} de accuracy,")
print(f"y el PR-AUC de un modelo sin poder discriminante sería ~{prevalencia:.3f}.")

ESTRATEGIAS = ['Sin balanceo', 'Pesos de clase', 'Submuestreo']
MODELOS = ['Regresión logística', 'XGBoost', 'CatBoost']


# 3. Preprocesamiento común (igual que el script 17, escenario B)
def crear_preprocesador():
    return ColumnTransformer([
        ('num', Pipeline([
            ('log', FunctionTransformer(np.log1p)),
            ('esc', StandardScaler())
        ]), COL_NUM),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), COL_CAT),
    ])


# 4. Fábrica de modelos según la estrategia
def crear_modelo(nombre, estrategia, n_neg, n_pos):
    pesos = (estrategia == 'Pesos de clase')
    if nombre == 'Regresión logística':
        return LogisticRegression(
            max_iter=1000, class_weight='balanced' if pesos else None)
    if nombre == 'XGBoost':
        return xgb.XGBClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=4,
            eval_metric='logloss', random_state=42,
            scale_pos_weight=(n_neg / n_pos) if pesos else 1.0)
    return CatBoostClassifier(
        iterations=200, learning_rate=0.05, depth=4, random_seed=42,
        verbose=0, allow_writing_files=False,
        auto_class_weights='Balanced' if pesos else None)


# 5. Validación cruzada estratificada: la prueba conserva la proporción real
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
acumulado = {(m, e): [] for m in MODELOS for e in ESTRATEGIAS}
rng = np.random.RandomState(42)

for idx_tr, idx_te in cv.split(X, y):
    X_tr, X_te = X.iloc[idx_tr], X.iloc[idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]

    for estrategia in ESTRATEGIAS:
        if estrategia == 'Submuestreo':
            # Se descartan registros de verano SOLO del entrenamiento
            pos = np.where(y_tr == 1)[0]
            neg = np.where(y_tr == 0)[0]
            neg_sel = rng.choice(neg, size=len(pos), replace=False)
            sel = np.concatenate([pos, neg_sel])
            X_fit, y_fit = X_tr.iloc[sel], y_tr[sel]
        else:
            X_fit, y_fit = X_tr, y_tr

        prep = crear_preprocesador()
        Xp_fit = prep.fit_transform(X_fit)
        Xp_te = prep.transform(X_te)
        n_pos = int(y_fit.sum())
        n_neg = int(len(y_fit) - n_pos)

        for nombre in MODELOS:
            modelo = crear_modelo(nombre, estrategia, n_neg, n_pos)
            modelo.fit(Xp_fit, y_fit)
            proba = modelo.predict_proba(Xp_te)[:, 1]
            pred = (proba >= 0.5).astype(int)
            acumulado[(nombre, estrategia)].append([
                average_precision_score(y_te, proba),
                roc_auc_score(y_te, proba),
                precision_score(y_te, pred, zero_division=0),
                recall_score(y_te, pred, zero_division=0),
                f1_score(y_te, pred, zero_division=0),
            ])

# 6. Tabla de resultados
filas = []
for (modelo, estrategia), vals in acumulado.items():
    v = np.array(vals)
    filas.append({
        'Modelo': modelo, 'Estrategia': estrategia,
        'PR_media': v[:, 0].mean(), 'PR_std': v[:, 0].std(),
        'AUC_media': v[:, 1].mean(), 'AUC_std': v[:, 1].std(),
        'PRE_media': v[:, 2].mean(), 'PRE_std': v[:, 2].std(),
        'REC_media': v[:, 3].mean(), 'REC_std': v[:, 3].std(),
        'F1_media': v[:, 4].mean(), 'F1_std': v[:, 4].std(),
    })
res = pd.DataFrame(filas)

print("\n" + "=" * 100)
print("RESULTADOS (media ± desv. est. en 15 pliegues, prueba con proporción real 90/10)")
print("Umbral de decisión = 0.5 para precisión, recall y F1")
print("=" * 100)
print(f"{'Modelo':<20} | {'Estrategia':<15} | {'PR-AUC':<14} | {'ROC-AUC':<14} | "
      f"{'Precision inv.':<14} | {'Recall inv.':<14} | {'F1':<14}")
print("-" * 118)
for _, f in res.sort_values(['Modelo', 'PR_media'], ascending=[True, False]).iterrows():
    print(f"{f.Modelo:<20} | {f.Estrategia:<15} | "
          f"{f.PR_media:.3f} ± {f.PR_std:.3f} | {f.AUC_media:.3f} ± {f.AUC_std:.3f} | "
          f"{f.PRE_media:.3f} ± {f.PRE_std:.3f}  | "
          f"{f.REC_media:.3f} ± {f.REC_std:.3f} | {f.F1_media:.3f} ± {f.F1_std:.3f}")

mejor = res.loc[res['PR_media'].idxmax()]
print(f"\nMayor PR-AUC: {mejor.Modelo} con '{mejor.Estrategia}' ({mejor.PR_media:.3f})")
print(f"Referencia de azar (PR-AUC) = prevalencia de invierno = {prevalencia:.3f}")
print("Comparar la PRECISIÓN de este script con la del script 17 muestra cuánto se")
print("sobreestima el desempeño cuando la prueba se balancea artificialmente.")
print("=" * 100)

# 7. Gráfica: PR-AUC, F1 y compromiso precisión/recall
colores = {'Sin balanceo': '#C44E52', 'Pesos de clase': '#55A868', 'Submuestreo': '#4C72B0'}
marcas = {'Regresión logística': 'o', 'XGBoost': 's', 'CatBoost': '^'}
x = np.arange(len(MODELOS))
ancho = 0.26

fig, ejes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (col, titulo) in zip(ejes[:2], [('PR', 'PR-AUC (área precisión-recall)'),
                                         ('F1', 'F1 de invierno (umbral 0.5)')]):
    for i, est in enumerate(ESTRATEGIAS):
        sub = res[res.Estrategia == est].set_index('Modelo').loc[MODELOS]
        ax.bar(x + (i - 1) * ancho, sub[f'{col}_media'], ancho,
               yerr=sub[f'{col}_std'], capsize=3, label=est, color=colores[est])
    if col == 'PR':
        ax.axhline(prevalencia, color='black', linestyle='--', label='Azar (prevalencia)')
    ax.set_xticks(x)
    ax.set_xticklabels(MODELOS)
    ax.set_ylim(0, 1)
    ax.set_title(titulo, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

ax = ejes[2]
for _, f in res.iterrows():
    ax.scatter(f.REC_media, f.PRE_media, s=140, color=colores[f.Estrategia],
               marker=marcas[f.Modelo], edgecolor='black')
for est, c in colores.items():
    ax.scatter([], [], color=c, label=est, s=80)
for mod, m in marcas.items():
    ax.scatter([], [], color='gray', marker=m, label=mod, s=80, edgecolor='black')
ax.axhline(prevalencia, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('Recall de invierno')
ax.set_ylabel('Precisión de invierno')
ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1.02)
ax.set_title('Compromiso precisión / recall', fontweight='bold')
ax.grid(alpha=0.3)
ax.legend(loc='lower left', fontsize=8)

plt.suptitle('Clasificación con la distribución real (prueba 90/10, sin balancear)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_IMG, dpi=130)
print(f"\nGráfica guardada en: {RUTA_IMG}")
plt.show()