# ========================================================================
# generar_muestras_3.py
# ------------------------------------------------------------------------
# Genera muestras estratificadas mixtas con refuerzo manual para casos
# especiales (ej. "Vías urbanas") en el conjunto de entrenamiento.
#
# @author   Mauricio Viera-Torres <ronnyvieramt95@gmail.com>
# @address  La Tola (Ecuador)
# @version  1.0.0 (2025-07-01)
# ========================================================================

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data_loader import cargar_shapefile, muestrear_mixto_area_y_num_poligonos

# --- Rutas ---
shp_train = "../DATA/OUTPUT/CORREGIDOS/suelo_permeable_train_c.shp"
shp_val   = "../DATA/OUTPUT/CORREGIDOS/suelo_permeable_val_c.shp"
shp_test  = "../DATA/OUTPUT/CORREGIDOS/suelo_permeable_test_c.shp"

columna_clase = "TIPO_SUELO"
columna_cob   = "COB"

# --- Áreas objetivo ---
area_objetivo_train = {
    "Permeable": 3_000_000,       # sube 50%
    "No permeable": 6_000_000     # sube 3x para incluir vías
}
area_objetivo_val = {
    "Permeable": 750_000,
    "No permeable": 1_500_000
}

area_objetivo_test = {
    "Permeable": 750_000,
    "No permeable": 1_500_000
}


# --- Cargar shapefiles ---
gdf_train = cargar_shapefile(shp_train)
gdf_val   = cargar_shapefile(shp_val)
gdf_test  = cargar_shapefile(shp_test)

# --- Funciones auxiliares ---
def reforzar_vias_urbanas(gdf_base, muestra_base):
    vias = gdf_base[(gdf_base[columna_clase] == "No permeable") & (gdf_base[columna_cob].str.lower().str.contains("vias urbanas"))]
    if not vias.empty:
        muestra_final = gpd.GeoDataFrame(pd.concat([muestra_base, vias.sample(1)]), crs=gdf_base.crs)
        return muestra_final
    return muestra_base

# --- Generar muestras ---
muestra_train = muestrear_mixto_area_y_num_poligonos(gdf_train, columna_clase, area_objetivo_train, min_poligonos_por_clase=100)
muestra_train = reforzar_vias_urbanas(gdf_train, muestra_train)

muestra_val   = muestrear_mixto_area_y_num_poligonos(gdf_val, columna_clase, area_objetivo_val, min_poligonos_por_clase=100)
muestra_val   = reforzar_vias_urbanas(gdf_val, muestra_val)

muestra_test  = muestrear_mixto_area_y_num_poligonos(gdf_test, columna_clase, area_objetivo_test, min_poligonos_por_clase=100)
muestra_test  = reforzar_vias_urbanas(gdf_test, muestra_test)

# --- Visualización Train ---
muestra_train["area_m2"] = muestra_train.geometry.area
print("\n→ TRAIN")
print(muestra_train[columna_clase].value_counts())
print(muestra_train.groupby(columna_clase)["area_m2"].sum().round(2))

# --- Visualización VALIDACION ---
muestra_val["area_m2"] = muestra_val.geometry.area
print("\n→ VALIDACION")
print(muestra_val[columna_clase].value_counts())
print(muestra_val.groupby(columna_clase)["area_m2"].sum().round(2))

# --- Visualización TEST ---
muestra_test["area_m2"] = muestra_test.geometry.area
print("\n→ TEST")
print(muestra_test[columna_clase].value_counts())
print(muestra_test.groupby(columna_clase)["area_m2"].sum().round(2))

# --- Gráficos
muestra_train[columna_clase].value_counts().plot(kind='bar', title="Polígonos por clase (Train)")
plt.ylabel("Cantidad")
plt.grid(True)
plt.tight_layout()
plt.show()

muestra_train.groupby(columna_clase)["area_m2"].sum().plot(kind='bar', title="Área por clase (m²) - Train", color=['gray', 'green'])
plt.ylabel("Área [m²]")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Guardar resultados ---
muestra_train = gpd.GeoDataFrame(muestra_train, geometry="geometry", crs=gdf_train.crs)
muestra_val   = gpd.GeoDataFrame(muestra_val,   geometry="geometry", crs=gdf_val.crs)
muestra_test  = gpd.GeoDataFrame(muestra_test,  geometry="geometry", crs=gdf_test.crs)


muestra_train.to_file("../DATA/OUTPUT/MUESTRAS/muestra_train_area_3.shp")
muestra_val.to_file("../DATA/OUTPUT/MUESTRAS/muestra_val_area_3.shp")
muestra_test.to_file("../DATA/OUTPUT/MUESTRAS/muestra_test_area_3.shp")
