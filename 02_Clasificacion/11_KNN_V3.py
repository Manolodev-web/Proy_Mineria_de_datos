# ============================================================
# 11_KNN_V3.py
# Clasificación KNN con Dataset Balanceado (Versión Final)
# Dataset: 0901_Produccion_Arroz_V3_Clasificacion.xlsx (Balanceado 50/50)
# Imagen:  ../03_imagenes/31_knn_v3.png
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

# 1. Configuración de rutas
BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_DATOS = BASE_DIR / "datos" / "0901_Produccion_Arroz_V3_Clasificacion.xlsx"
RUTA_IMG = BASE_DIR / "03_imagenes" / "31_knn_v3.png"
RUTA_IMG.parent.mkdir(exist_ok=True)

# 2. Cargar datos balanceados
df = pd.read_excel(RUTA_DATOS)
df['target_campaña'] = df['CAMPAÑA AGRICOLA'].str.lower().map({'verano': 0, 'invierno': 1})
df = df.dropna(subset=['SUPERFICIE (Ha)', 'PRODUCCION (qq)', 'target_campaña'])

print("=" * 65)
print("KNN V3 (DATASET BALANCEADO 140 vs 140)")
print("=" * 65)

# 3. Preparación y escalado
X = df[['SUPERFICIE (Ha)', 'PRODUCCION (qq)']].values
y = df['target_campaña'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42, stratify=y
)

# 4. Búsqueda de K y función de pesos óptimos
param_grid = {
    'n_neighbors': range(1, 25),
    'weights': ['uniform', 'distance']
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=cv, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)

best_knn = grid.best_estimator_
k_opt = grid.best_params_['n_neighbors']
w_opt = grid.best_params_['weights']

y_pred = best_knn.predict(X_test)
y_proba = best_knn.predict_proba(X_test)[:, 1]

# 5. Evaluación de métricas
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"Mejores Hiperparámetros : K = {k_opt}, weights = '{w_opt}'")
print(f"Accuracy Global        : {acc * 100:.2f} %")
print(f"Precisión (Invierno)   : {prec:.4f}")
print(f"Sensibilidad / Recall  : {rec:.4f}")
print(f"F1-Score (Invierno)    : {f1:.4f}")
print(f"ROC-AUC                : {auc:.4f}")
print("-" * 65)
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['Verano', 'Invierno']))
print("=" * 65)

# 6. Gráficas diagnósticas (Matriz, ROC, Fronteras)
fig = plt.figure(figsize=(15, 10))

# 6.1 Matriz de Confusión
ax1 = plt.subplot(2, 2, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Verano', 'Invierno'],
            yticklabels=['Verano', 'Invierno'], ax=ax1)
ax1.set_title(f'Matriz de Confusión KNN V3 (K={k_opt})', fontsize=12, fontweight='bold')
ax1.set_xlabel('Predicción')
ax1.set_ylabel('Realidad')

# 6.2 Curva ROC
ax2 = plt.subplot(2, 2, 2)
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax2.plot(fpr, tpr, color='#1565C0', lw=2.5, label=f'KNN (AUC = {auc:.3f})')
ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Azar (AUC = 0.500)')
ax2.set_xlabel('Tasa de Falsos Positivos (FPR)')
ax2.set_ylabel('Tasa de Verdaderos Positivos (TPR)')
ax2.set_title('Curva ROC - KNN V3', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# 6.3 Fronteras de Decisión en Espacio Estandarizado
ax3 = plt.subplot(2, 1, 2)
x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = best_knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

ax3.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlGn)
ax3.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1],
            c='red', label='Verano (Real)', edgecolor='k', s=40, alpha=0.7)
ax3.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1],
            c='green', label='Invierno (Real)', edgecolor='k', s=60, alpha=0.9)
ax3.set_xlabel('Superficie (Ha) [Estandarizado]')
ax3.set_ylabel('Producción (qq) [Estandarizado]')
ax3.set_title(f'Fronteras de Decisión KNN V3 (K={k_opt}, {w_opt})', fontsize=12, fontweight='bold')
ax3.legend()

plt.suptitle('KNN V3 - Diagnóstico con Dataset Balanceado', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_IMG, dpi=130)
print(f"\nGráfica guardada en: {RUTA_IMG}")
plt.show()