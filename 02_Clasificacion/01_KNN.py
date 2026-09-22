import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# === 1. CARGAR DATOS ===
# Cambia la ruta si tu archivo Excel está en otra carpeta
df = pd.read_excel('../datos/0901_Produccion_Arroz.xlsx', sheet_name='Hoja1')

print("="*60)
print("📊 KNN - CLASIFICACIÓN (Campaña Agrícola: Verano vs Invierno)")
print("="*60)

# Convertir la variable de texto a números (0 = Verano, 1 = Invierno)
df['target_campaña'] = df['CAMPAÑA AGRICOLA'].map({'verano': 0, 'invierno': 1})

# === 2. PREPARAR DATOS ===
VARIABLE_X = 'SUPERFICIE (Ha)'
VARIABLE_Y = 'PRODUCCION (qq)'
TARGET = 'target_campaña'

X = df[[VARIABLE_X, VARIABLE_Y]]
y = df[TARGET]

# === 3. ESCALAR TODAS LAS VARIABLES ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 4. DIVIDIR ENTRENAMIENTO Y PRUEBA ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# === 5. ENCONTRAR EL MEJOR K ===
k_values = range(1, 21)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(knn, X_scaled, y, cv=cv, scoring='accuracy')
    cv_scores.append(scores.mean())

k_final = k_values[np.argmax(cv_scores)]
print(f"✅ Mejor K encontrado: {k_final}\n")

# === 6. ENTRENAR MODELO FINAL ===
knn = KNeighborsClassifier(n_neighbors=k_final)
knn.fit(X_train, y_train)

# === 7. PREDICCIONES Y MÉTRICAS ===
y_test_pred = knn.predict(X_test)
acc_test = accuracy_score(y_test, y_test_pred)
cm = confusion_matrix(y_test, y_test_pred)

print(f"📊 RESULTADOS - Precisión en PRUEBA: {acc_test:.2%}\n")

# === 8. GRÁFICO 1: MATRIZ DE CONFUSIÓN ===
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Verano (0)', 'Invierno (1)'],
            yticklabels=['Verano (0)', 'Invierno (1)'])
plt.title(f'Matriz de Confusión KNN (K={k_final})')
plt.xlabel('Predicción')
plt.ylabel('Realidad')
plt.tight_layout()
plt.savefig('knn_matriz_arroz.png', dpi=300)
plt.show()

# === 9. GRÁFICO 2: FRONTERAS DE DECISIÓN ===
x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlGn)
plt.scatter(X_test[y_test==0][:, 0], X_test[y_test==0][:, 1], c='red', label='Prueba - Verano', edgecolor='k')
plt.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1], c='green', label='Prueba - Invierno', edgecolor='k')
plt.xlabel('Superficie (Ha) - Escalado')
plt.ylabel('Producción (qq) - Escalado')
plt.title(f'Fronteras de Decisión KNN - Campaña Agrícola (K={k_final})')
plt.legend()
plt.tight_layout()
plt.savefig('knn_fronteras_arroz.png', dpi=300)
plt.show()