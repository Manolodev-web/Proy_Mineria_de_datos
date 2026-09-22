import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score

# 1. Cargar datos
df = pd.read_excel('../datos/0901_Produccion_Arroz.xlsx', sheet_name='Hoja1')
df['target'] = df['CAMPAÑA AGRICOLA'].map({'verano': 0, 'invierno': 1})

X = df[['SUPERFICIE (Ha)', 'PRODUCCION (qq)', 'RENDIMIENTO (Kg/Ha)']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

print("=" * 60)
print("ENSAMBLES BOOSTING - CLASIFICACIÓN (Campaña Agrícola)")
print("=" * 60)

# 2. Entrenar Modelos (Añadimos class_weight='balanced' a LightGBM por el desbalance)
ada_clf = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_clf.fit(X_train, y_train)
pred_ada = ada_clf.predict(X_test)
proba_ada = ada_clf.predict_proba(X_test)[:, 1]

lgb_clf = LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=42, verbose=-1)
lgb_clf.fit(X_train, y_train)
pred_lgb = lgb_clf.predict(X_test)
proba_lgb = lgb_clf.predict_proba(X_test)[:, 1]

print(f"AdaBoost - Accuracy: {accuracy_score(y_test, pred_ada):.4f}")
print(f"LightGBM - Accuracy: {accuracy_score(y_test, pred_lgb):.4f}")

# 3. Gráfico 1: Matrices de Confusión (Imagen 15)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
nombres_clases = ["Verano", "Invierno"]

sns.heatmap(confusion_matrix(y_test, pred_lgb), annot=True, fmt="d", cmap="GnBu", ax=axes[1], xticklabels=nombres_clases, yticklabels=nombres_clases)
axes[0].set_title("Matriz de Confusión - AdaBoost")
axes[0].set_xlabel("Predicción")
axes[0].set_ylabel("Real")

sns.heatmap(confusion_matrix(y_test, pred_lgb), annot=True, fmt="d", cmap="GnBu", ax=axes[1], xticklabels=nombres_clases, yticklabels=nombres_clases)
axes[1].set_title("Matriz de Confusión - LightGBM (Balanceado)")
axes[1].set_xlabel("Predicción")

plt.tight_layout()
plt.savefig("../03_imagenes/15_boosting_clf_cm.png", dpi=150)
plt.show()

# 4. Gráfico 2: Curvas ROC (Imagen 16)
plt.figure(figsize=(7, 6))

fpr_ada, tpr_ada, _ = roc_curve(y_test, proba_ada)
auc_ada = roc_auc_score(y_test, proba_ada)
plt.plot(fpr_ada, tpr_ada, color='orange', lw=2, label=f"AdaBoost (AUC = {auc_ada:.3f})")

fpr_lgb, tpr_lgb, _ = roc_curve(y_test, proba_lgb)
auc_lgb = roc_auc_score(y_test, proba_lgb)
plt.plot(fpr_lgb, tpr_lgb, color='teal', lw=2, label=f"LightGBM (AUC = {auc_lgb:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Azar (AUC = 0.5)")
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curvas ROC: AdaBoost vs LightGBM")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("../03_imagenes/16_boosting_clf_roc.png", dpi=150)
plt.show()