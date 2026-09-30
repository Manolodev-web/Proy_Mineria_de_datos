# ============================================================
# 13_Arbol_Decision_Clasificacion_V2.py
# Árbol de Clasificación de Campaña Agrícola (Verano vs Invierno) V2
# Dataset: 0901_Produccion_Arroz_V3_Clasificacion.xlsx (Balanceado)
# Imagen:  ../03_imagenes/34_dt_clasificacion_v2.png
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
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
RUTA_IMG = BASE_DIR / "03_imagenes" / "34_dt_clasificacion_v2.png"
RUTA_IMG.parent.mkdir(exist_ok=True)

# 2. Cargar datos balanceados
df = pd.read_excel(RUTA_DATOS)
df['target_campaña'] = df['CAMPAÑA AGRICOLA'].str.lower().map({'verano': 0, 'invierno': 1})

# En clasificación de temporada, Superficie, Producción y Rendimiento son variables descriptivas válidas
features = ['SUPERFICIE (Ha)', 'PRODUCCION (qq)', 'RENDIMIENTO (Kg/Ha)']
X = df[features].values
y = df['target_campaña'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# 3. Optimización de hiperparámetros con GridSearchCV
param_grid = {
    'max_depth': [2, 3, 4, 5],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=cv, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)

best_dt = grid.best_estimator_
y_pred = best_dt.predict(X_test)
y_proba = best_dt.predict_proba(X_test)[:, 1]

# 4. Métricas
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("=" * 65)
print("ÁRBOL DE DECISIÓN CLASIFICACIÓN V2 (DATASET BALANCEADO)")
print("=" * 65)
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

# 5. Gráficas diagnósticas (Matriz, Curva ROC y Estructura del Árbol)
fig = plt.figure(figsize=(16, 11))

# 5.1 Matriz de Confusión
ax1 = plt.subplot(2, 2, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Verano', 'Invierno'],
            yticklabels=['Verano', 'Invierno'], ax=ax1)
ax1.set_title("Matriz de Confusión - Árbol Clasificación V2", fontsize=12, fontweight='bold')
ax1.set_xlabel("Predicción")
ax1.set_ylabel("Realidad")

# 5.2 Curva ROC
ax2 = plt.subplot(2, 2, 2)
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax2.plot(fpr, tpr, color='#1565C0', lw=2.5, label=f'Árbol (AUC = {auc:.3f})')
ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Azar (AUC = 0.500)')
ax2.set_xlabel('Tasa de Falsos Positivos (FPR)')
ax2.set_ylabel('Tasa de Verdaderos Positivos (TPR)')
ax2.set_title('Curva ROC - Árbol Clasificación V2', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# 5.3 Estructura del Árbol
ax3 = plt.subplot(2, 1, 2)
plot_tree(best_dt, feature_names=['Superficie (Ha)', 'Producción (qq)', 'Rendimiento (Kg/Ha)'],
          class_names=['Verano', 'Invierno'], filled=True, rounded=True, fontsize=9, ax=ax3)
ax3.set_title(f"Estructura del Árbol de Clasificación (max_depth={grid.best_params_['max_depth']})", fontsize=12, fontweight='bold')

plt.suptitle("Árbol de Decisión V2 - Clasificación Campaña Agrícola", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_IMG, dpi=130)
print(f"\nGráfica guardada en: {RUTA_IMG}")
plt.show()