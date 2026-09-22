import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# === 1. CARGA Y PREPARACIÓN DE DATOS ===
df = pd.read_excel('../datos/0901_Produccion_Arroz.xlsx', sheet_name='Hoja1')

# Variable objetivo: 0 = Verano, 1 = Invierno
df['target_campaña'] = df['CAMPAÑA AGRICOLA'].map({'verano': 0, 'invierno': 1})

# X tendrá 3 variables predictoras, y es el target
X = df[['SUPERFICIE (Ha)', 'PRODUCCION (qq)', 'RENDIMIENTO (Kg/Ha)']]
y = df['target_campaña']

print("="*60)
print("📈 REGRESIÓN LOGÍSTICA (Campaña Agrícola: Verano vs Invierno)")
print("="*60)

# === 2. DIVISIÓN Y ESCALADO ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 3. ENTRENAMIENTO MULTIVARIADO ===
modelo = LogisticRegression()
modelo.fit(X_train_scaled, y_train)

# === 4. EVALUACIÓN Y MATRIZ DE CONFUSIÓN ===
y_pred = modelo.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Precisión Global (Accuracy): {acc*100:.2f}%\n")

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Verano", "Invierno"], yticklabels=["Verano", "Invierno"])
plt.ylabel("Realidad")
plt.xlabel("Predicción del Modelo")
plt.title(f"Matriz de Confusión - Accuracy: {acc*100:.2f}%")
plt.show()

# === 5. MODELO UNIVARIADO PARA CURVA S-SHAPED ===
# Usaremos el Rendimiento para graficar la probabilidad de que sea Invierno
X_single = df[["RENDIMIENTO (Kg/Ha)"]]
X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(X_single, y, test_size=0.2, random_state=42)

model_1d = LogisticRegression()
model_1d.fit(X_tr_s, y_tr_s)

x_range = np.linspace(X_single["RENDIMIENTO (Kg/Ha)"].min(), X_single["RENDIMIENTO (Kg/Ha)"].max(), 300).reshape(-1, 1)
probs = model_1d.predict_proba(x_range)[:, 1]

plt.figure(figsize=(8, 5))
plt.scatter(X_te_s, y_te_s, c=y_te_s, cmap="coolwarm", alpha=0.6, label="Datos Reales (0=Verano, 1=Invierno)")
plt.plot(x_range, probs, color="black", linewidth=2, label="Curva Logística (Probabilidad de Invierno)")
plt.title("Rendimiento (Kg/Ha) vs Probabilidad de Campaña de Invierno")
plt.xlabel("Rendimiento (Kg/Ha)")
plt.ylabel("Probabilidad Calculada")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()