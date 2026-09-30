# ============================================================
# 14_Boosting_Clasificacion_V2.py
# Ensambles Boosting (AdaBoost vs LightGBM) en Dataset Balanceado
# Dataset: 0901_Produccion_Arroz_V3_Clasificacion.xlsx (Balanceado 50/50)
# Imagen:  ../03_imagenes/36_boosting_clf_v2.png
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
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
RUTA_IMG = BASE_DIR / "03_imagenes" / "36_boosting_clf_v2.png"
RUTA_IMG.parent.mkdir(exist_ok=True)

# 2. Cargar datos balanceados
df = pd.read_excel(RUTA_DATOS)
df['target_campaña'] = df['CAMPAÑA AGRICOLA'].str.lower().map({'verano': 0, 'invierno': 1})

features = ['SUPERFICIE (Ha)', 'PRODUCCION (qq)', 'RENDIMIENTO (Kg/Ha)']
X = df[features].values
y = df['target_campaña'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print("=" * 65)
print("ENSAMBLES BOOSTING CLASIFICACIÓN V2 (DATASET BALANCEADO)")
print("=" * 65)

# 3. Entrenamiento
# 3.1 AdaBoost
ada_clf = AdaBoostClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
ada_clf.fit(X_train, y_train)
pred_ada = ada_clf.predict(X_test)
proba_ada = ada_clf.predict_proba(X_test)[:, 1]

# 3.2 LightGBM (con dataset balanceado ya no requiere forzar class_weight)
lgb_clf = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, num_leaves=15, random_state=42, verbose=-1)
lgb_clf.fit(X_train, y_train)
pred_lgb = lgb_clf.predict(X_test)
proba_lgb = lgb_clf.predict_proba(X_test)[:, 1]

# 4. Métricas Comparativas
metricas = {
    "AdaBoost": {
        "Accuracy": accuracy_score(y_test, pred_ada),
        "Precision": precision_score(y_test, pred_ada),
        "Recall": recall_score(y_test, pred_ada),
        "F1": f1_score(y_test, pred_ada),
        "AUC": roc_auc_score(y_test, proba_ada)
    },
    "LightGBM": {
        "Accuracy": accuracy_score(y_test, pred_lgb),
        "Precision": precision_score(y_test, pred_lgb),
        "Recall": recall_score(y_test, pred_lgb),
        "F1": f1_score(y_test, pred_lgb),
        "AUC": roc_auc_score(y_test, proba_lgb)
    }
}

print(f"{'Métrica (Invierno)':<20} | {'AdaBoost':<15} | {'LightGBM':<15}")
print("-" * 55)
for m in ["Accuracy", "Precision", "Recall", "F1", "AUC"]:
    print(f"{m:<20} | {metricas['AdaBoost'][m]:<15.4f} | {metricas['LightGBM'][m]:<15.4f}")
print("=" * 65)

print("\nReporte de Clasificación AdaBoost:")
print(classification_report(y_test, pred_ada, target_names=['Verano', 'Invierno']))

print("\nReporte de Clasificación LightGBM:")
print(classification_report(y_test, pred_lgb, target_names=['Verano', 'Invierno']))

# 5. Gráficas Diagnósticas (Matrices de Confusión y Curvas ROC)
fig = plt.figure(figsize=(16, 10))
nombres = ["Verano", "Invierno"]

# 5.1 Matriz de Confusión AdaBoost
ax1 = plt.subplot(2, 2, 1)
sns.heatmap(confusion_matrix(y_test, pred_ada), annot=True, fmt='d', cmap='Oranges',
            xticklabels=nombres, yticklabels=nombres, ax=ax1)
ax1.set_title("Matriz de Confusión - AdaBoost V2", fontsize=12, fontweight='bold')
ax1.set_xlabel("Predicción")
ax1.set_ylabel("Realidad")

# 5.2 Matriz de Confusión LightGBM
ax2 = plt.subplot(2, 2, 2)
sns.heatmap(confusion_matrix(y_test, pred_lgb), annot=True, fmt='d', cmap='Blues',
            xticklabels=nombres, yticklabels=nombres, ax=ax2)
ax2.set_title("Matriz de Confusión - LightGBM V2", fontsize=12, fontweight='bold')
ax2.set_xlabel("Predicción")
ax2.set_ylabel("Realidad")

# 5.3 Curvas ROC Comparadas
ax3 = plt.subplot(2, 1, 2)
fpr_ada, tpr_ada, _ = roc_curve(y_test, proba_ada)
fpr_lgb, tpr_lgb, _ = roc_curve(y_test, proba_lgb)

ax3.plot(fpr_ada, tpr_ada, color='darkorange', lw=2.5, label=f"AdaBoost (AUC = {metricas['AdaBoost']['AUC']:.3f})")
ax3.plot(fpr_lgb, tpr_lgb, color='teal', lw=2.5, label=f"LightGBM (AUC = {metricas['LightGBM']['AUC']:.3f})")
ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Azar (AUC = 0.500)")
ax3.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=11)
ax3.set_ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=11)
ax3.set_title("Comparación de Curvas ROC - Ensambles Boosting V2", fontsize=12, fontweight='bold')
ax3.legend(loc="lower right", fontsize=11)
ax3.grid(alpha=0.3)

plt.suptitle("Boosting V2 - Clasificación de Campaña con Dataset Balanceado", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_IMG, dpi=130)
print(f"\nGráfica guardada en: {RUTA_IMG}")
plt.show()