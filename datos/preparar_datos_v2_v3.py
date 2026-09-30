# ============================================================
# preparar_datos_v2_v3.py
# Generación de Datasets Especializados:
#   1. V2 para Regresión (Filtro de Outliers > 3000 Ha)
#   2. V3 para Clasificación (Submuestreo Balanceado 140 vs 140)
# ============================================================

import pandas as pd
from pathlib import Path

# 1. Definición de rutas relativas
DIR_DATOS = Path(__file__).resolve().parent
RUTA_ORIGINAL = DIR_DATOS / "0901_Produccion_Arroz.xlsx"
RUTA_SALIDA_V2 = DIR_DATOS / "0901_Produccion_Arroz_V2_Regresion.xlsx"
RUTA_SALIDA_V3 = DIR_DATOS / "0901_Produccion_Arroz_V3_Clasificacion.xlsx"

print("=" * 65)
print("PROCESAMIENTO DE DATOS: GENERACIÓN DE VERSIONES V2 Y V3")
print("=" * 65)

# 2. Cargar dataset crudo
df = pd.read_excel(RUTA_ORIGINAL)
total_original = len(df)
print(f"Registros originales cargados: {total_original}")

# ------------------------------------------------------------
# 3. Generar V2: Datos limpios para Regresión
# ------------------------------------------------------------
# Se excluyen registros con Superficie >= 3000 Ha (solo 2 casos atípicos extremos)
# y se descartan valores nulos en variables clave.
df_v2 = df[df["SUPERFICIE (Ha)"] < 3000].dropna(
    subset=["SUPERFICIE (Ha)", "PRODUCCION (qq)"]
).copy()

df_v2.to_excel(RUTA_SALIDA_V2, index=False)
print(f"\n[V2 REGRESIÓN CREADO]")
print(f" -> Ruta: {RUTA_SALIDA_V2.name}")
print(f" -> Filas conservadas: {len(df_v2)} (Excluidos {total_original - len(df_v2)} outliers)")

# ------------------------------------------------------------
# 4. Generar V3: Datos balanceados para Clasificación
# ------------------------------------------------------------
# Contamos cuántos registros de invierno existen y tomamos una muestra
# aleatoria idéntica de verano para balance 50/50.
df_cls = df.dropna(subset=["SUPERFICIE (Ha)", "PRODUCCION (qq)", "CAMPAÑA AGRICOLA"]).copy()
df_invierno = df_cls[df_cls["CAMPAÑA AGRICOLA"].str.lower() == "invierno"].copy()
conteo_invierno = len(df_invierno)

df_verano_muestreado = df_cls[
    df_cls["CAMPAÑA AGRICOLA"].str.lower() == "verano"
].sample(n=conteo_invierno, random_state=42).copy()

df_v3 = pd.concat([df_invierno, df_verano_muestreado]).sample(
    frac=1.0, random_state=42
).reset_index(drop=True)

df_v3.to_excel(RUTA_SALIDA_V3, index=False)
print(f"\n[V3 CLASIFICACIÓN CREADO]")
print(f" -> Ruta: {RUTA_SALIDA_V3.name}")
print(f" -> Balance: {conteo_invierno} Verano vs {conteo_invierno} Invierno (Total: {len(df_v3)} filas)")
print("=" * 65)
print("Proceso completado exitosamente.")