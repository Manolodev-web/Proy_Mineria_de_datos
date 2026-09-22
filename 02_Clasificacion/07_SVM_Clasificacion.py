import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

# 1. Cargar datos
df = pd.read_excel('../datos/0901_Produccion_Arroz.xlsx', sheet_name='Hoja1')
df['target'] = df['CAMPAÑA AGRICOLA'].map({'verano': 0, 'invierno': 1})

X = df[['SUPERFICIE (Ha)', 'PRODUCCION (qq)', 'RENDIMIENTO (Kg/Ha)']]
y = df['target']

print("=" * 60)
print("SUPPORT VECTOR MACHINE (SVC) - CLASIFICACIÓN (Campaña)")
print("=" * 60)

# 2. Dividir y Escalar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Entrenar SVC con balanceo
modelo_svc = SVC(kernel="rbf", C=1, gamma="scale", class_weight="balanced", probability=True, random_state=42)
modelo_svc.fit(X_train_scaled, y_train)

pred_svc = modelo_svc.predict(X_test_scaled)
print(f"Accuracy General: {accuracy_score(y_test, pred_svc):.4f}")
print("\nReporte de Clasificación:")
print(classification_report(y_test, pred_svc, target_names=["Verano", "Invierno"]))

# 4. Gráfico 1: Matriz de Confusión (Imagen 22)
cm = confusion_matrix(y_test, pred_svc)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=["Verano", "Invierno"], yticklabels=["Verano", "Invierno"])
plt.title("Matriz de Confusión - SVC (Balanceado)")
plt.xlabel("Predicción")
plt.ylabel("Valor real")
plt.tight_layout()
plt.savefig("../03_imagenes/22_svc_clf_cm.png", dpi=150)
plt.show()

# 5. Gráfico 2: Fronteras de Decisión usando PCA a 2D (Imagen 23)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)

modelo_pca = SVC(kernel="rbf", C=1, gamma="scale", class_weight="balanced")
modelo_pca.fit(X_train_pca, y_train)

x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
Z = modelo_pca.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
plt.scatter(X_train_pca[y_train == 0, 0], X_train_pca[y_train == 0, 1], c='blue', s=20, edgecolor='k', label='Verano')
plt.scatter(X_train_pca[y_train == 1, 0], X_train_pca[y_train == 1, 1], c='red', s=20, edgecolor='k', label='Invierno')

# Vectores de soporte en el espacio 2D
plt.scatter(modelo_pca.support_vectors_[:, 0], modelo_pca.support_vectors_[:, 1], 
            s=80, facecolors='none', edgecolors='yellow', linewidths=1, label='Vectores de Soporte')

plt.title("SVC - Fronteras de Decisión Curvas (RBF) vía PCA")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend()
plt.tight_layout()
plt.savefig("../03_imagenes/23_svc_clf_fronteras.png", dpi=150)
plt.show()