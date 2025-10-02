# ========================================================================
# feature_extraction.py
# ------------------------------------------------------------------------
# Función para romper la espacialidad de los datos ráster
# ------------------------------------------------------------------------
# @author   Mauricio Viera-Torres <ronnyvieramt95@gmail.com>
# @address  La Tola (Ecuador)
# @version  1.0.0 (2025-06-23)
# ========================================================================


import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping

def extraer_atributos_desde_muestra(muestra, raster_path, clase_col, clase_dict):
    """
    Extrae atributos espectrales y etiquetas desde un raster y un GeoDataFrame de muestra.

    Args:
        muestra (GeoDataFrame): Muestra con geometría y columna de clase.
        raster_path (str): Ruta al archivo raster.
        clase_col (str): Nombre de la columna con clases ('TIPO_SUELO').
        clase_dict (dict): Diccionario con codificación de clases.

    Returns:
        X (np.array): Matriz de atributos (n muestras x d bandas).
        y (np.array): Vector de etiquetas.
    """
    X = np.zeros((0, ), dtype=np.float32).reshape(0, 3)  # Suponiendo RGB
    y = np.zeros((0, ), dtype=int)

    with rasterio.open(raster_path) as src:
        bandas = src.count

        for _, row in muestra.iterrows():
            geom = [mapping(row['geometry'])]
            clase = row[clase_col]

            try:
                clip, _ = mask(src, geom, crop=True)
                d, h, w = clip.shape
                pix = clip.reshape(d, -1).T
                pix = [p for p in pix if not all(v == 0 for v in p)]
                X = np.concatenate((X, np.array(pix)))
                y = np.concatenate((y, np.repeat(clase_dict[clase], len(pix))))
            except Exception as e:
                print(f"⚠️ Error al extraer de un polígono: {e}")
                continue

    print(f"✅ Atributos extraídos: {X.shape[0]} muestras, {X.shape[1]} bandas")
    return X, y
