import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 1. Cargar datos
df = pd.read_excel('../datos/0901_Produccion_Arroz.xlsx', sheet_name='Hoja1')
df['target'] = df['CAMPAÑA AGRICOLA'].map({'verano': 0, 'invierno': 1})

print("=" * 60)
print("NAIVE BAYES - CLASIFICACIÓN (Campaña Agrícola)")
print("=" * 60)

# 2. One-Hot Encoding para MultinomialNB
X = pd.get_dummies(df[['DEPARTAMENTO', 'PROVINCIA']], drop_first=False)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

# 3. Entrenar Multinomial Naive Bayes
nb_clf = MultinomialNB()
nb_clf.fit(X_train, y_train)

pred = nb_clf.predict(X_test)

# 4. Gráfico: Matriz de Confusión (Imagen 12)
cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=["Verano", "Invierno"], yticklabels=["Verano", "Invierno"])
plt.title("Matriz de Confusión - Naive Bayes Clasificación")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig("../03_imagenes/12_nb_clasificacion_cm.png", dpi=150)
plt.show()