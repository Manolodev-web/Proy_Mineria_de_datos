import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Cargar datos
df = pd.read_excel('../datos/0901_Produccion_Arroz.xlsx', sheet_name='Hoja1')
df['target'] = df['CAMPAÑA AGRICOLA'].map({'verano': 0, 'invierno': 1})

X = df[['SUPERFICIE (Ha)', 'PRODUCCION (qq)', 'RENDIMIENTO (Kg/Ha)']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

print("=" * 60)
print("ÁRBOL DE DECISIÓN - CLASIFICACIÓN (Campaña Agrícola)")
print("=" * 60)

# 2. Entrenar con balanceo de clases
dt_clf = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42)
dt_clf.fit(X_train, y_train)

pred = dt_clf.predict(X_test)

# 3. Gráfico 1: Matriz de Confusión (Imagen 10)
cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Verano", "Invierno"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", values_format="d")
plt.title("Matriz de Confusión - Árbol de Clasificación")
plt.tight_layout()
plt.savefig("../03_imagenes/10_dt_clasificacion_cm.png", dpi=150)
plt.show()

# 4. Gráfico 2: Estructura del Árbol (Imagen 11)
plt.figure(figsize=(16, 8))
plot_tree(dt_clf, feature_names=X.columns, class_names=["Verano", "Invierno"], filled=True, rounded=True, fontsize=9)
plt.title("Estructura del Árbol de Clasificación")
plt.tight_layout()
plt.savefig("../03_imagenes/11_dt_clasificacion_arbol.png", dpi=150, bbox_inches="tight")
plt.show()