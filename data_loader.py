# ========================================================================
# data_loader.py
# ------------------------------------------------------------------------
# Funciones para carga de datos y muestreo
# ------------------------------------------------------------------------
# @author   Mauricio Viera-Torres <ronnyvieramt95@gmail.com>
# @address  La Tola (Ecuador)
# @version  1.0.0 (2025-06-23)
# ========================================================================


import geopandas as gpd
import pandas as pd
import os
import numpy as np

def cargar_shapefile(path):
    """
    Carga un shapefile con geopandas y retorna el GeoDataFrame.
    """
    gdf = gpd.read_file(path)
    print(f"📁 Archivo leído: {path}")
    print("📌 Columnas disponibles:", list(gdf.columns))
    print("📏 Total de registros:", len(gdf))

    if 'TIPO_SUELO' in gdf.columns:
        conteo = gdf['TIPO_SUELO'].value_counts()
        print("\n📊 Conteo por clase:\n", conteo)
    else:
        print("⚠️ La columna 'TIPO_SUELO' no se encuentra en el shapefile.")

    return gdf


def muestrear_por_area_balanceado(gdf, columna_clase, area_objetivo_por_clase):
    """
    Retorna un subconjunto balanceado del GeoDataFrame basado en el área por clase,
    sin sobrepasar el área disponible por clase.

    Args:
        gdf (GeoDataFrame): GeoDataFrame original con geometría y clase.
        columna_clase (str): Nombre de la columna de clases.
        area_objetivo_por_clase (dict): Área objetivo por clase.

    Returns:
        GeoDataFrame con la muestra por clase.
    """
    gdf = gdf.copy()
    gdf["area_m2"] = gdf.geometry.area
    muestras = []

    for clase, grupo in gdf.groupby(columna_clase):
        grupo = grupo.sort_values("area_m2", ascending=False).reset_index(drop=True)
        suma_area = 0.0
        seleccionados = []

        area_deseada = area_objetivo_por_clase.get(clase, 0)
        area_disponible = grupo["area_m2"].sum()

        if area_disponible < area_deseada:
            print(f"⚠️ Clase '{clase}': sólo hay {area_disponible:.2f} m² disponibles (objetivo era {area_deseada})")
            area_deseada = area_disponible  # tomamos lo máximo

        for _, row in grupo.iterrows():
            if suma_area + row["area_m2"] > area_deseada:
                continue
            seleccionados.append(row)
            suma_area += row["area_m2"]
            if suma_area >= area_deseada:
                break

        muestras.append(gpd.GeoDataFrame(seleccionados, geometry="geometry", crs=gdf.crs))

    gdf_muestra = pd.concat(muestras, ignore_index=True)
    print(f"\n✅ Muestra creada con: {len(gdf_muestra)} polígonos")
    return gdf_muestra


def muestrear_mixto_area_y_num_poligonos(gdf, columna_clase, area_objetivo_por_clase, min_poligonos_por_clase=100):
    """
    Muestreo mixto por área y cantidad mínima de polígonos. Asegura al menos `min_poligonos_por_clase`,
    y recorta el total para no superar el área deseada.

    Returns:
        GeoDataFrame con la muestra por clase.
    """
    gdf = gdf.copy()
    gdf["area_m2"] = gdf.geometry.area
    muestras = []

    for clase, grupo in gdf.groupby(columna_clase):
        grupo = grupo.sort_values("area_m2", ascending=True).reset_index(drop=True)

        seleccionados = grupo.iloc[:min_poligonos_por_clase].copy()
        suma_area = seleccionados["area_m2"].sum()

        # Si ya superamos el área, cortamos aquí
        area_deseada = area_objetivo_por_clase.get(clase, np.inf)

        if suma_area > area_deseada:
            # Recortar al área deseada
            acumulado = 0
            recorte = []
            for _, row in seleccionados.iterrows():
                if acumulado + row["area_m2"] > area_deseada:
                    break
                recorte.append(row)
                acumulado += row["area_m2"]
            seleccionados = gpd.GeoDataFrame(recorte, geometry="geometry", crs=gdf.crs)
        else:
            # Añadir más hasta alcanzar área objetivo sin duplicar polígonos
            resto = grupo.iloc[min_poligonos_por_clase:]
            for _, row in resto.iterrows():
                if suma_area + row["area_m2"] > area_deseada:
                    break
                seleccionados = pd.concat([seleccionados, row.to_frame().T], ignore_index=True)
                suma_area += row["area_m2"]

        muestras.append(seleccionados)

    gdf_muestra = pd.concat(muestras, ignore_index=True)
    print(f"✅ Muestra creada con: {len(gdf_muestra)} polígonos")
    return gdf_muestra
