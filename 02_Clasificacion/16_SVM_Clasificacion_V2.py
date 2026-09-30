# ============================================================
# 16_SVM_Clasificacion_V2.py
# Support Vector Classifier (SVC) en Dataset Balanceado V2
# Dataset: 0901_Produccion_Arroz_V3_Clasificacion.xlsx (Balanceado 50/50)
# Imagen:  ../03_imagenes/40_svm_clf_v2.png
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# 1. Pagur-ongan dagiti files
BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_DATOS = BASE_DIR / "datos" / "0901_Produccion_Arroz_V3_Clasificacion.xlsx"
RUTA_IMG = BASE_DIR / "03_imagenes" / "40_svm_clf_v2.png"
RUTA_IMG.parent.mkdir(exist_ok=True)

# 2. Panangikarga kadagiti nabalanse a datos
df = pd.read_excel(RUTA_DATOS)
df['target'] = df['CAMPAÑA AGRICOLA'].str.lower().map({'verano': 0, 'invierno': 1})

features = ['SUPERFICIE (Ha)', 'PRODUCCION (qq)', 'RENDIMIENTO (Kg/Ha)']
X = df[features].values
y = df['target'].values

# Panangbingay ken panang-scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("=" * 65)
print("SUPPORT VECTOR CLASSIFIER (SVC) V2 (DATASET BALANCEADO)")
print("=" * 65)

# 3. Panagsapul iti kasayaatan a C ken gamma babaen ti GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 50],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'kernel': ['rbf']
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(SVC(probability=True, random_state=42), param_grid, cv=cv, scoring='f1', n_jobs=-1)
grid.fit(X_train_scaled, y_train)

best_svc = grid.best_estimator_
y_pred = best_svc.predict(X_test_scaled)
y_proba = best_svc.predict_proba(X_test_scaled)[:, 1]

# 4. Rukod ti Panagpatingga
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"Mejores Hiperparámetros : {grid.best_params_}")
print(f"Accuracy Global        : {acc * 100:.2f} %")
print(f"Precisión (Invierno)   : {prec:.4f}")
print(f"Sensibilidad / Recall  : {rec:.4f}")
print(f"F1-Score (Invierno)    : {f1:.4f}")
print(f"ROC-AUC                : {auc:.4f}")
print("-" * 65)
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['Verano', 'Invierno']))
print("=" * 65)

# 5. Grapiko ti Panangsukimat (Matriz, Curva ROC ken Fronteras via PCA)
fig = plt.figure(figsize=(16, 10))
nombres = ["Verano", "Invierno"]

# 5.1 Matriz de Confusión
ax1 = plt.subplot(2, 2, 1)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Purples',
            xticklabels=nombres, yticklabels=nombres, ax=ax1)
ax1.set_title("Matriz de Confusión - SVC V2", fontsize=12, fontweight='bold')
ax1.set_xlabel("Predicción")
ax1.set_ylabel("Realidad")

# 5.2 Curva ROC
ax2 = plt.subplot(2, 2, 2)
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax2.plot(fpr, tpr, color='purple', lw=2.5, label=f"SVC (AUC = {auc:.3f})")
ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Azar (AUC = 0.500)")
ax2.set_xlabel("Tasa de Falsos Positivos (FPR)")
ax2.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
ax2.set_title("Curva ROC - SVC V2", fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# 5.3 Fronteras de Decisión en 2D babaen ti PCA
ax3 = plt.subplot(2, 1, 2)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)

model_pca = SVC(kernel="rbf", C=grid.best_params_['C'], gamma=grid.best_params_['gamma'], random_state=42)
model_pca.fit(X_train_pca, y_train)

x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
Z = model_pca.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

ax3.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
ax3.scatter(X_train_pca[y_train == 0, 0], X_train_pca[y_train == 0, 1],
            c='blue', s=30, edgecolor='k', label='Verano (Entrenamiento)')
ax3.scatter(X_train_pca[y_train == 1, 0], X_train_pca[y_train == 1, 1],
            c='red', s=30, edgecolor='k', label='Invierno (Entrenamiento)')

# Support Vectors
ax3.scatter(model_pca.support_vectors_[:, 0], model_pca.support_vectors_[:, 1],
            s=80, facecolors='none', edgecolors='yellow', linewidths=1.2, label='Vectores de Soporte')

ax3.set_title("Fronteras de Decisión RBF y Vectores de Soporte (Proyección PCA 2D)", fontsize=12, fontweight='bold')
ax3.set_xlabel("Componente Principal 1")
ax3.set_ylabel("Componente Principal 2")
ax3.legend()

plt.suptitle("Support Vector Classifier V2 - Clasificación con Dataset Balanceado", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_IMG, dpi=130)
print(f"\nGráfica guardada en: {RUTA_IMG}")
plt.show()