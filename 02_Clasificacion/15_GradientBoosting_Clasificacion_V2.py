# ============================================================
# 15_GradientBoosting_Clasificacion_V2.py
# Gradient Boosting (XGBoost vs CatBoost) en Dataset Balanceado
# Dataset: 0901_Produccion_Arroz_V3_Clasificacion.xlsx (Balanceado 50/50)
# Imagen:  ../03_imagenes/38_gb_clf_v2.png
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# 1. Configuración de rutas
BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_DATOS = BASE_DIR / "datos" / "0901_Produccion_Arroz_V3_Clasificacion.xlsx"
RUTA_IMG = BASE_DIR / "03_imagenes" / "38_gb_clf_v2.png"
RUTA_IMG.parent.mkdir(exist_ok=True)

# 2. Cargar datos balanceados
df = pd.read_excel(RUTA_DATOS)
df['target'] = df['CAMPAÑA AGRICOLA'].str.lower().map({'verano': 0, 'invierno': 1})

features = ['DEPARTAMENTO', 'SUPERFICIE (Ha)', 'PRODUCCION (qq)', 'RENDIMIENTO (Kg/Ha)']
X = df[features].copy()
y = df['target'].values

print("=" * 65)
print("GRADIENT BOOSTING CLASIFICACIÓN V2 (DATASET BALANCEADO)")
print("=" * 65)

# 3. Preparación de variables categóricas
X_xgb = pd.get_dummies(X, columns=['DEPARTAMENTO'], drop_first=True)
X_cat = X.copy()
cat_features = ['DEPARTAMENTO']

# Partición estratificada idéntica
X_tr_xgb, X_te_xgb, y_tr, y_te = train_test_split(X_xgb, y, test_size=0.30, random_state=42, stratify=y)
X_tr_cat, X_te_cat, _, _ = train_test_split(X_cat, y, test_size=0.30, random_state=42, stratify=y)

# 4. Entrenamiento de Modelos
# Al estar la base balanceada 1:1, no se requieren pesos artificiales
xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
xgb_clf.fit(X_tr_xgb, y_tr)
pred_xgb = xgb_clf.predict(X_te_xgb)
proba_xgb = xgb_clf.predict_proba(X_te_xgb)[:, 1]

cat_clf = CatBoostClassifier(
    iterations=150,
    learning_rate=0.05,
    depth=4,
    cat_features=cat_features,
    verbose=0,
    random_state=42
)
cat_clf.fit(X_tr_cat, y_tr)
pred_cat = cat_clf.predict(X_te_cat)
proba_cat = cat_clf.predict_proba(X_te_cat)[:, 1]

# 5. Métricas Comparativas
metricas = {
    "XGBoost": {
        "Accuracy": accuracy_score(y_te, pred_xgb),
        "Precision": precision_score(y_te, pred_xgb),
        "Recall": recall_score(y_te, pred_xgb),
        "F1": f1_score(y_te, pred_xgb),
        "AUC": roc_auc_score(y_te, proba_xgb)
    },
    "CatBoost": {
        "Accuracy": accuracy_score(y_te, pred_cat),
        "Precision": precision_score(y_te, pred_cat),
        "Recall": recall_score(y_te, pred_cat),
        "F1": f1_score(y_te, pred_cat),
        "AUC": roc_auc_score(y_te, proba_cat)
    }
}

print(f"{'Métrica (Invierno)':<20} | {'XGBoost':<15} | {'CatBoost':<15}")
print("-" * 55)
for m in ["Accuracy", "Precision", "Recall", "F1", "AUC"]:
    print(f"{m:<20} | {metricas['XGBoost'][m]:<15.4f} | {metricas['CatBoost'][m]:<15.4f}")
print("=" * 65)

print("\nReporte de Clasificación XGBoost:")
print(classification_report(y_te, pred_xgb, target_names=['Verano', 'Invierno']))

print("\nReporte de Clasificación CatBoost:")
print(classification_report(y_te, pred_cat, target_names=['Verano', 'Invierno']))

# 6. Gráficas Diagnósticas (Matrices de Confusión, ROC y Eficiencia de Columnas)
fig = plt.figure(figsize=(16, 11))
nombres = ["Verano", "Invierno"]

# 6.1 Matriz XGBoost
ax1 = plt.subplot(2, 2, 1)
sns.heatmap(confusion_matrix(y_te, pred_xgb), annot=True, fmt='d', cmap='Reds',
            xticklabels=nombres, yticklabels=nombres, ax=ax1)
ax1.set_title("Matriz de Confusión - XGBoost V2", fontsize=12, fontweight='bold')
ax1.set_xlabel("Predicción")
ax1.set_ylabel("Realidad")

# 6.2 Matriz CatBoost
ax2 = plt.subplot(2, 2, 2)
sns.heatmap(confusion_matrix(y_te, pred_cat), annot=True, fmt='d', cmap='Blues',
            xticklabels=nombres, yticklabels=nombres, ax=ax2)
ax2.set_title("Matriz de Confusión - CatBoost V2", fontsize=12, fontweight='bold')
ax2.set_xlabel("Predicción")
ax2.set_ylabel("Realidad")

# 6.3 Curvas ROC Comparadas
ax3 = plt.subplot(2, 2, 3)
fpr_xgb, tpr_xgb, _ = roc_curve(y_te, proba_xgb)
fpr_cat, tpr_cat, _ = roc_curve(y_te, proba_cat)

ax3.plot(fpr_xgb, tpr_xgb, color='crimson', lw=2.5, label=f"XGBoost (AUC = {metricas['XGBoost']['AUC']:.3f})")
ax3.plot(fpr_cat, tpr_cat, color='royalblue', lw=2.5, label=f"CatBoost (AUC = {metricas['CatBoost']['AUC']:.3f})")
ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Azar (AUC = 0.500)")
ax3.set_xlabel("Tasa de Falsos Positivos (FPR)")
ax3.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
ax3.set_title("Comparación de Curvas ROC", fontsize=12, fontweight='bold')
ax3.legend(loc="lower right")
ax3.grid(alpha=0.3)

# 6.4 Comparación de Eficiencia de Columnas (One-Hot vs Nativo)
ax4 = plt.subplot(2, 2, 4)
modelos_bar = ['XGBoost\n(One-Hot)', 'CatBoost\n(Nativo)']
columnas_bar = [X_xgb.shape[1], X_cat.shape[1]]
barras = ax4.bar(modelos_bar, columnas_bar, color=['crimson', 'royalblue'], width=0.5)
ax4.set_title("Columnas en Matriz de Entrada", fontsize=12, fontweight='bold')
ax4.set_ylabel("Cantidad de Columnas")

for b in barras:
    h = b.get_height()
    ax4.text(b.get_x() + b.get_width()/2, h + 0.15, int(h), ha='center', fontweight='bold', fontsize=12)

plt.suptitle("Gradient Boosting V2 - Clasificación con Dataset Balanceado", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_IMG, dpi=130)
print(f"\nGráfica guardada en: {RUTA_IMG}")
plt.show()