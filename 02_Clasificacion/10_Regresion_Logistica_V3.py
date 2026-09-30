# ============================================================
# 10_Regresion_Logistica_V3.py
# Clasificación de Campaña Agrícola (Verano vs Invierno)
# Dataset: 0901_Produccion_Arroz_V3_Clasificacion.xlsx (Balanceado 50/50)
# Imagen:  ../03_imagenes/30_logistica_v3.png
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# 1. Rutas
BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_DATOS = BASE_DIR / "datos" / "0901_Produccion_Arroz_V3_Clasificacion.xlsx"
RUTA_IMG = BASE_DIR / "03_imagenes" / "30_logistica_v3.png"
RUTA_IMG.parent.mkdir(exist_ok=True)

# 2. Cargar datos balanceados
df = pd.read_excel(RUTA_DATOS)
df['target_campaña'] = df['CAMPAÑA AGRICOLA'].str.lower().map({'verano': 0, 'invierno': 1})
df = df.dropna(subset=['SUPERFICIE (Ha)', 'PRODUCCION (qq)', 'target_campaña'])

print("=" * 65)
print("REGRESIÓN LOGÍSTICA V3 (DATASET BALANCEADO 140 vs 140)")
print("=" * 65)
print("Balance de clases en dataset:")
print(df['CAMPAÑA AGRICOLA'].value_counts())
print("-" * 65)

# 3. Preparación de datos y escalado
X = df[['SUPERFICIE (Ha)', 'PRODUCCION (qq)']].values
y = df['target_campaña'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42, stratify=y
)

# 4. Optimización con GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear']
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(LogisticRegression(max_iter=5000), param_grid, cv=cv, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)

logreg = grid.best_estimator_
y_pred = logreg.predict(X_test)
y_proba = logreg.predict_proba(X_test)[:, 1]

# 5. Métricas
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"Mejor C encontrado     : {grid.best_params_['C']}")
print(f"Accuracy Global        : {acc * 100:.2f} %")
print(f"Precisión (Invierno)   : {prec:.4f}")
print(f"Sensibilidad / Recall  : {rec:.4f}")
print(f"F1-Score (Invierno)    : {f1:.4f}")
print(f"ROC-AUC                : {auc:.4f}")
print("-" * 65)
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['Verano', 'Invierno']))

# Coeficientes
print("Coeficientes en escala estandarizada:")
print(f"  Superficie : {logreg.coef_[0][0]:+.4f}")
print(f"  Producción : {logreg.coef_[0][1]:+.4f}")
print(f"  Intercepto : {logreg.intercept_[0]:+.4f}")
print("=" * 65)

# 6. Gráfica de diagnóstico (Matriz de confusión, ROC y Fronteras)
fig = plt.figure(figsize=(15, 10))

# 6.1 Matriz de Confusión
ax1 = plt.subplot(2, 2, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Verano', 'Invierno'],
            yticklabels=['Verano', 'Invierno'], ax=ax1)
ax1.set_title(f'Matriz de Confusión Logística V3 (C={grid.best_params_["C"]})', fontsize=12, fontweight='bold')
ax1.set_xlabel('Predicción')
ax1.set_ylabel('Realidad')

# 6.2 Curva ROC
ax2 = plt.subplot(2, 2, 2)
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax2.plot(fpr, tpr, color='#2E7D32', lw=2.5, label=f'LogReg (AUC = {auc:.3f})')
ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Azar (AUC = 0.500)')
ax2.set_xlabel('Tasa de Falsos Positivos (FPR)')
ax2.set_ylabel('Tasa de Verdaderos Positivos (TPR)')
ax2.set_title('Curva ROC - Capacidad Discriminativa', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# 6.3 Fronteras de Decisión en Espacio Estandarizado
ax3 = plt.subplot(2, 1, 2)
x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

ax3.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlGn)
ax3.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1],
            c='red', label='Verano (Real)', edgecolor='k', s=40, alpha=0.7)
ax3.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1],
            c='green', label='Invierno (Real)', edgecolor='k', s=60, alpha=0.9)
ax3.set_xlabel('Superficie (Ha) [Estandarizado]')
ax3.set_ylabel('Producción (qq) [Estandarizado]')
ax3.set_title('Frontera Lineal de Separación de Campañas (V3)', fontsize=12, fontweight='bold')
ax3.legend()

plt.suptitle('Regresión Logística V3 - Diagnóstico con Dataset Balanceado', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_IMG, dpi=130)
print(f"\nGráfica guardada en: {RUTA_IMG}")
plt.show()