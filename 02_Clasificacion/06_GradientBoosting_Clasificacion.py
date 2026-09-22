import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb
from catboost import CatBoostClassifier

# 1. Cargar datos
df = pd.read_excel('../datos/0901_Produccion_Arroz.xlsx', sheet_name='Hoja1')
df['target'] = df['CAMPAÑA AGRICOLA'].map({'verano': 0, 'invierno': 1})

X = df[['DEPARTAMENTO', 'SUPERFICIE (Ha)', 'PRODUCCION (qq)', 'RENDIMIENTO (Kg/Ha)']]
y = df['target']

print("=" * 60)
print("GRADIENT BOOSTING - CLASIFICACIÓN (XGBoost vs CatBoost)")
print("=" * 60)

# 2. Preparación de datos y balanceo de pesos
X_xgb = pd.get_dummies(X, columns=['DEPARTAMENTO'])
X_train_xgb, X_test_xgb, y_train, y_test = train_test_split(X_xgb, y, test_size=0.30, random_state=42, stratify=y)
X_train_cat, X_test_cat, _, _ = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

# Calcular ratio para balancear XGBoost (CatBoost lo hace automático con 'Balanced')
ratio_clases = y_train.value_counts()[0] / y_train.value_counts()[1]

# 3. Entrenar Modelos
xgb_clf = xgb.XGBClassifier(n_estimators=100, scale_pos_weight=ratio_clases, random_state=42)
xgb_clf.fit(X_train_xgb, y_train)
pred_xgb = xgb_clf.predict(X_test_xgb)

cat_clf = CatBoostClassifier(iterations=100, cat_features=['DEPARTAMENTO'], auto_class_weights='Balanced', verbose=0, random_state=42)
cat_clf.fit(X_train_cat, y_train)
pred_cat = cat_clf.predict(X_test_cat)

print(f"XGBoost  - Accuracy: {accuracy_score(y_test, pred_xgb):.4f}")
print(f"CatBoost - Accuracy: {accuracy_score(y_test, pred_cat):.4f}")

# 4. Gráfico 1: Matrices de Confusión (Imagen 19)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
nombres = ["Verano", "Invierno"]

sns.heatmap(confusion_matrix(y_test, pred_xgb), annot=True, fmt="d", cmap="Reds", ax=axes[0], xticklabels=nombres, yticklabels=nombres)
axes[0].set_title("Matriz de Confusión - XGBoost (Balanceado)")
axes[0].set_xlabel("Predicción")
axes[0].set_ylabel("Real")

sns.heatmap(confusion_matrix(y_test, pred_cat), annot=True, fmt="d", cmap="Blues", ax=axes[1], xticklabels=nombres, yticklabels=nombres)
axes[1].set_title("Matriz de Confusión - CatBoost (Balanceado)")
axes[1].set_xlabel("Predicción")

plt.tight_layout()
plt.savefig("../03_imagenes/19_gb_clf_cm.png", dpi=150)
plt.show()

# 5. Gráfico 2: Comparación de Memoria/Columnas (Imagen 20)
modelos = ['XGBoost\n(One-Hot Encoding)', 'CatBoost\n(Nativo)']
columnas = [X_xgb.shape[1], X.shape[1]]

plt.figure(figsize=(6, 5))
barras = plt.bar(modelos, columnas, color=['crimson', 'royalblue'])
plt.title('Diferencia en Columnas Entregadas al Modelo\n(Eficiencia de Memoria)')
plt.ylabel('Cantidad de Columnas')

for bar in barras:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, int(yval), ha='center', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig("../03_imagenes/20_gb_clf_columnas.png", dpi=150)
plt.show()