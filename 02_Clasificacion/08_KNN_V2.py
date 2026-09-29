# ============================================================
# 08_KNN_V2.py
# KNN - Clasificación (Campaña Agrícola: Verano vs Invierno)
# Versión 2: Métricas completas + GridSearch + sin CSV
# ============================================================
# MEJORAS RESPECTO A V1 (01_KNN.py):
#   [V2-1] Rutas robustas con pathlib (imágenes a 03_imagenes/)
#   [V2-2] Métricas completas: accuracy, precision, recall, F1, ROC-AUC
#   [V2-3] GridSearchCV sobre K y weights
#   [V2-4] Validación cruzada estratificada
#   [V2-5] Matriz de confusión + curvas ROC + fronteras en una sola imagen
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# 0. Rutas robustas
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_IMAGENES = BASE_DIR / "03_imagenes"
RUTA_DATOS = BASE_DIR / "datos"
RUTA_IMAGENES.mkdir(exist_ok=True)

# ------------------------------------------------------------
# 1. Cargar datos
# ------------------------------------------------------------
df = pd.read_excel(RUTA_DATOS / '0901_Produccion_Arroz.xlsx', sheet_name='Hoja1')

print("="*65)
print("📊 KNN V2 - CLASIFICACIÓN (Campaña: Verano vs Invierno)")
print("="*65)

# Target binario
df['target_campaña'] = df['CAMPAÑA AGRICOLA'].map({'verano': 0, 'invierno': 1})
df = df.dropna(subset=['SUPERFICIE (Ha)', 'PRODUCCION (qq)', 'target_campaña'])

print("\n📌 Balance de clases:")
print(df['CAMPAÑA AGRICOLA'].value_counts())
print(f"Proporción invierno: {df['target_campaña'].mean():.2%}\n")

# ------------------------------------------------------------
# 2. Preparar datos
# ------------------------------------------------------------
X = df[['SUPERFICIE (Ha)', 'PRODUCCION (qq)']].values
y = df['target_campaña'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# ------------------------------------------------------------
# 3. GridSearchCV para K y weights
# ------------------------------------------------------------
param_grid = {
    'n_neighbors': range(1, 21),
    'weights': ['uniform', 'distance']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

knn_base = KNeighborsClassifier()   # KNN no acepta class_weight
grid = GridSearchCV(knn_base, param_grid, cv=cv,
                    scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)

k_final = grid.best_params_['n_neighbors']
w_final = grid.best_params_['weights']
print(f"✅ Mejores hiperparámetros: K={k_final}, weights={w_final}")
print(f"   F1 (CV) = {grid.best_score_:.4f}\n")

# ------------------------------------------------------------
# 4. Entrenar modelo final
# ------------------------------------------------------------
knn = grid.best_estimator_
y_pred = knn.predict(X_test)
y_proba = knn.predict_proba(X_test)[:, 1]

# ------------------------------------------------------------
# 5. Métricas completas
# ------------------------------------------------------------
metricas = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred, zero_division=0),
    'Recall': recall_score(y_test, y_pred, zero_division=0),
    'F1': f1_score(y_test, y_pred, zero_division=0),
    'ROC-AUC': roc_auc_score(y_test, y_proba),
}

print("📊 MÉTRICAS EN TEST")
print("-"*65)
for k, v in metricas.items():
    print(f"  {k:12s}: {v:.4f}")
print("-"*65)
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                            target_names=['Verano', 'Invierno'],
                            zero_division=0))

# ------------------------------------------------------------
# 6. Gráfica compuesta: CM + ROC + Fronteras
# ------------------------------------------------------------
fig = plt.figure(figsize=(16, 10))

# 6.1 Matriz de confusión
ax1 = plt.subplot(2, 2, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Verano', 'Invierno'],
            yticklabels=['Verano', 'Invierno'], ax=ax1)
ax1.set_title(f'Matriz de Confusión (K={k_final}, w={w_final})')
ax1.set_xlabel('Predicción')
ax1.set_ylabel('Realidad')

# 6.2 Curva ROC
ax2 = plt.subplot(2, 2, 2)
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax2.plot(fpr, tpr, color='#1565C0', linewidth=2,
         label=f'KNN (AUC={metricas["ROC-AUC"]:.3f})')
ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax2.set_xlabel('Falsos Positivos (FPR)')
ax2.set_ylabel('Verdaderos Positivos (TPR)')
ax2.set_title('Curva ROC')
ax2.legend()
ax2.grid(alpha=0.3)

# 6.3 Fronteras de decisión
ax3 = plt.subplot(2, 1, 2)
x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

ax3.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlGn)
ax3.scatter(X_test[y_test==0][:, 0], X_test[y_test==0][:, 1],
            c='red', label='Verano', edgecolor='k', s=30, alpha=0.7)
ax3.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1],
            c='green', label='Invierno', edgecolor='k', s=60, alpha=0.9)
ax3.set_xlabel('Superficie (Ha) - Escalado')
ax3.set_ylabel('Producción (qq) - Escalado')
ax3.set_title(f'Fronteras de Decisión KNN V2')
ax3.legend()

plt.suptitle('KNN V2 - Clasificación Campaña Agrícola',
             fontsize=14, fontweight='bold')
plt.tight_layout()

ruta_img = RUTA_IMAGENES / "26_knn_clasificacion_v2.png"
plt.savefig(ruta_img, dpi=120, bbox_inches='tight')
print(f"\n✅ Imagen guardada en: {ruta_img}")
plt.show()