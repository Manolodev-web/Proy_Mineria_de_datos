# ============================================================
# 17_Errores_por_Grupo_Regresion_V4.py
# ¿DÓNDE se equivocan los modelos de regresión?
#   - Predicciones fuera de muestra (validación cruzada de 5 pliegues)
#   - Error por tamaño de comunidad (rangos de superficie)
#   - Error por departamento
#   - Se compara la Regresión lineal con LightGBM (mismas variables)
# Dataset: 0901_Produccion_Arroz_V2_Regresion.xlsx
# Imagen:  ../03_imagenes/44_errores_por_grupo_reg_v4.png
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# 1. Rutas
BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_DATOS = BASE_DIR / "datos" / "0901_Produccion_Arroz_V2_Regresion.xlsx"
RUTA_IMG = BASE_DIR / "03_imagenes" / "44_errores_por_grupo_reg_v4.png"
RUTA_IMG.parent.mkdir(exist_ok=True)

# 2. Cargar datos y preparar (log1p en superficie y producción, como en scripts previos)
df = pd.read_excel(RUTA_DATOS).reset_index(drop=True)
X = np.log1p(df[['SUPERFICIE (Ha)']].values)
y = df['PRODUCCION (qq)'].values
y_log = np.log1p(y)

print("=" * 75)
print("ANÁLISIS DE ERRORES POR GRUPO V4 (predicciones fuera de muestra, 5 pliegues)")
print("=" * 75)

# 3. Predicciones fuera de muestra: cada registro se predice sin haber sido
#    usado para entrenar (ningún dato se evalúa con un modelo que ya lo vio)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
modelos = {
    'Regresión lineal': LinearRegression(),
    'LightGBM': lgb.LGBMRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=4, num_leaves=15,
        random_state=42, verbose=-1),
}
for nombre, modelo in modelos.items():
    df[f'pred_{nombre}'] = np.expm1(cross_val_predict(modelo, X, y_log, cv=cv))

# 4. Grupos: rangos de superficie
limites = [0, 5, 20, 100, np.inf]
etiquetas = ['< 5 ha', '5 a 20 ha', '20 a 100 ha', '≥ 100 ha']
df['rango_sup'] = pd.cut(df['SUPERFICIE (Ha)'], bins=limites,
                         labels=etiquetas, right=False)


def resumen_grupo(g, col_pred):
    real, pred = g['PRODUCCION (qq)'].values, g[col_pred].values
    ape = np.abs(pred - real) / real * 100
    return pd.Series({
        'n': len(g),
        'MAE (qq)': np.abs(pred - real).mean(),
        'MAPE (%)': ape.mean(),
        'Mediana APE (%)': np.median(ape),
        'Sesgo (%)': ((pred - real) / real * 100).mean(),
    })


def tabla_por(columna):
    partes = []
    for nombre in modelos:
        t = df.groupby(columna, observed=True).apply(
            lambda g: resumen_grupo(g, f'pred_{nombre}'))
        t.insert(0, 'Modelo', nombre)
        partes.append(t.reset_index())
    return pd.concat(partes)


por_rango = tabla_por('rango_sup')
por_depto = tabla_por('DEPARTAMENTO')

# 5. Impresión
for titulo, tabla, col in [
        ("POR TAMAÑO DE COMUNIDAD (superficie)", por_rango, 'rango_sup'),
        ("POR DEPARTAMENTO", por_depto, 'DEPARTAMENTO')]:
    print("\n" + "=" * 90)
    print(f"ERROR {titulo}")
    print("=" * 90)
    print(f"{'Grupo':<14} | {'Modelo':<17} | {'n':<5} | {'MAE (qq)':<9} | "
          f"{'MAPE (%)':<9} | {'Mediana APE':<12} | {'Sesgo (%)':<9}")
    print("-" * 90)
    for _, f in tabla.sort_values([col, 'Modelo']).iterrows():
        print(f"{str(f[col]):<14} | {f['Modelo']:<17} | {int(f['n']):<5} | "
              f"{f['MAE (qq)']:<9.1f} | {f['MAPE (%)']:<9.1f} | "
              f"{f['Mediana APE (%)']:<12.1f} | {f['Sesgo (%)']:<+9.1f}")

print("\nInterpretación guiada:")
print(" - MAPE muy superior a la mediana de APE => pocos casos con error enorme.")
print(" - Sesgo > 0 => el modelo tiende a SOBREESTIMAR la producción del grupo.")
print(" - Departamentos con n < 15 (p. ej. Tarija y Chuquisaca) NO permiten conclusiones.")

# 6. Gráfica
fig, ejes = plt.subplots(1, 3, figsize=(19, 6))
colores = {'Regresión lineal': '#4C72B0', 'LightGBM': '#DD8452'}
ancho = 0.38

ax = ejes[0]
x = np.arange(len(etiquetas))
for i, nombre in enumerate(modelos):
    sub = por_rango[por_rango.Modelo == nombre].set_index('rango_sup').loc[etiquetas]
    ax.bar(x + (i - 0.5) * ancho, sub['MAPE (%)'], ancho,
           label=nombre, color=colores[nombre])
n_rango = por_rango[por_rango.Modelo == 'LightGBM'].set_index('rango_sup').loc[etiquetas, 'n']
ax.set_xticks(x)
ax.set_xticklabels([f"{e}\n(n={int(n)})" for e, n in zip(etiquetas, n_rango)])
ax.set_ylabel('MAPE (%)')
ax.set_title('Error relativo por tamaño de comunidad', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.legend()

ax = ejes[1]
deptos = (df['DEPARTAMENTO'].value_counts()[lambda s: s >= 15].index.tolist())
x = np.arange(len(deptos))
for i, nombre in enumerate(modelos):
    sub = por_depto[por_depto.Modelo == nombre].set_index('DEPARTAMENTO').loc[deptos]
    ax.bar(x + (i - 0.5) * ancho, sub['MAPE (%)'], ancho,
           label=nombre, color=colores[nombre])
n_dep = df['DEPARTAMENTO'].value_counts()[deptos]
ax.set_xticks(x)
ax.set_xticklabels([f"{d}\n(n={n})" for d, n in zip(deptos, n_dep)])
ax.set_ylabel('MAPE (%)')
ax.set_title('Error relativo por departamento (n ≥ 15)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.legend()

ax = ejes[2]
ax.scatter(y, df['pred_LightGBM'], alpha=0.35, s=15, color='#DD8452')
lim = [y.min(), y.max()]
ax.plot(lim, lim, 'k--', label='Predicción perfecta')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Producción real (qq, escala log)')
ax.set_ylabel('Producción predicha (qq, escala log)')
ax.set_title('LightGBM: real vs predicho (fuera de muestra)', fontweight='bold')
ax.grid(alpha=0.3)
ax.legend()

plt.suptitle('¿Dónde se equivocan los modelos de regresión?',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RUTA_IMG, dpi=130)
print(f"\nGráfica guardada en: {RUTA_IMG}")
plt.show()