import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# 1. Cargar datos
df = pd.read_excel('../datos/0901_Produccion_Arroz.xlsx', sheet_name='Hoja1')

print("=" * 60)
print("NAIVE BAYES - REGRESIÓN ADAPTADA (Discretización de Producción)")
print("=" * 60)

# Discretizar variable continua para Naive Bayes
est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
df['prod_categoria'] = est.fit_transform(df[['PRODUCCION (qq)']].values).ravel()

X = df[['SUPERFICIE (Ha)', 'RENDIMIENTO (Kg/Ha)']]
y = df['prod_categoria']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# 2. Entrenar Gaussian Naive Bayes
nb_reg = GaussianNB()
nb_reg.fit(X_train, y_train)

pred = nb_reg.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f"Accuracy de categorías (Regresión binned): {acc:.4f}")

# 3. Gráfico: Matriz de Confusión para Regresión Binned (Imagen 09)
cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=['Baja', 'Media', 'Alta'], yticklabels=['Baja', 'Media', 'Alta'])
plt.title("Matriz de Confusión - Naive Bayes Regresión (Categorizada)")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig("../03_imagenes/09_nb_regresion_cm.png", dpi=150)
plt.show()