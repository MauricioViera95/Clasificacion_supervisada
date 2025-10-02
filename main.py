# ========================================================================
# main.py
# ------------------------------------------------------------------------
# Módulo principal que coordina la ejecución de los distintos algoritmos
# propuestos de clasificación supervisada
# ------------------------------------------------------------------------
# @author   Mauricio Viera-Torres <ronnyvieramt95@gmail.com>
# @address  La Tola (Ecuador)
# @version  1.0.0 (2025-06-23)
# ========================================================================

# main.py

from data_loader import cargar_shapefile
from config import shp_train, shp_val, shp_test
import geopandas as gpd
import numpy as np

# Cargar shapefiles
gdf_train = cargar_shapefile(shp_train)
gdf_val = cargar_shapefile(shp_val)
gdf_test = cargar_shapefile(shp_test)


# Cargar shapefile --- Train
from data_loader import cargar_shapefile
gdf_train = cargar_shapefile("../DATA/OUTPUT/CORREGIDOS/suelo_permeable_train_c.shp")

# Área total por clase
area_total_por_clase = gdf_train.groupby("TIPO_SUELO")["Shape_Area"].sum()
area_total_por_clase_m2 = area_total_por_clase.round(2)

print("📐 Área total disponible por clase (en m²):")
print(area_total_por_clase_m2)


# Cargar shapefile --- Val
from data_loader import cargar_shapefile
gdf_train = cargar_shapefile("../DATA/OUTPUT/CORREGIDOS/suelo_permeable_val_c.shp")

# Área total por clase
area_total_por_clase = gdf_train.groupby("TIPO_SUELO")["Shape_Area"].sum()
area_total_por_clase_m2 = area_total_por_clase.round(2)

print("📐 Área total disponible por clase (en m²):")
print(area_total_por_clase_m2)


# Cargar shapefile --- Test
from data_loader import cargar_shapefile
gdf_train = cargar_shapefile("../DATA/OUTPUT/CORREGIDOS/suelo_permeable_test_c.shp")

# Área total por clase
area_total_por_clase = gdf_train.groupby("TIPO_SUELO")["Shape_Area"].sum()
area_total_por_clase_m2 = area_total_por_clase.round(2)

print("📐 Área total disponible por clase (en m²):")
print(area_total_por_clase_m2)



#######################################3
# Área total a muestrear por zona
area_total_train = 4_000_000
area_total_val = 1_000_000
area_total_test = 1_000_000


######################### Generación del dataset -  Extraer muestras del raster
import pandas as pd
import geopandas as gpd
import numpy as np
from config import raster_path_1, raster_path_2
from feature_extraction import extraer_atributos_desde_muestra
from knn_model import expandir_atributos_rgb

# Leer muestras generadas previamente
muestra_train = gpd.read_file("../DATA/OUTPUT/MUESTRAS/muestra_train_area_3.shp")
muestra_val   = gpd.read_file("../DATA/OUTPUT/MUESTRAS/muestra_val_area_3.shp")
muestra_test  = gpd.read_file("../DATA/OUTPUT/MUESTRAS/muestra_test_area_3.shp")


# --- Definir clases
clases = sorted(muestra_train["TIPO_SUELO"].unique())
clase_dict = {clase: i for i, clase in enumerate(clases)}
print("🔢 Diccionario de clases:", clase_dict)

from feature_extraction import extraer_atributos_desde_muestra
from config import raster_path_1, raster_path_2

# --- Extraer atributos
##################### 1. Train ##############################################
# Para entrenamiento (solo se superpone al raster 1)
X_train, y_train = extraer_atributos_desde_muestra(muestra_train, raster_path_1, "TIPO_SUELO", clase_dict)

## Comprobar
import pandas as pd

unique, counts = np.unique(y_train, return_counts=True)
for clase, count in zip(unique, counts):
    print(f"Clase {clase} → {count:,} píxeles")

df_preview = pd.DataFrame(X_train, columns=[f"Banda_{i+1}" for i in range(X_train.shape[1])])
df_preview["Clase"] = y_train
print(df_preview.sample(5))

##################### 2. Val ##############################################
# Para validación (se superpone a ambos → combinar resultados)
X_val_1, y_val_1 = extraer_atributos_desde_muestra(muestra_val, raster_path_1, "TIPO_SUELO", clase_dict)
X_val_2, y_val_2 = extraer_atributos_desde_muestra(muestra_val, raster_path_2, "TIPO_SUELO", clase_dict)

X_val = np.concatenate((X_val_1, X_val_2), axis=0)
y_val = np.concatenate((y_val_1, y_val_2), axis=0)

# Verificar muestreo balanceado
unique, counts = np.unique(y_val, return_counts=True)
for clase, count in zip(unique, counts):
    print(f"Clase {clase} → {count:,} píxeles")


df_preview_2 = pd.DataFrame(X_val, columns=[f"Banda_{i+1}" for i in range(X_val.shape[1])])
df_preview_2["Clase"] = y_val
print(df_preview_2.sample(5))


##################### 3. Test ##############################################
# Para test (solo se superpone al raster 2)
X_test, y_test = extraer_atributos_desde_muestra(muestra_test, raster_path_2, "TIPO_SUELO", clase_dict)

# Comprobar
unique, counts = np.unique(y_test, return_counts=True)
for clase, count in zip(unique, counts):
    print(f"Clase {clase} → {count:,} píxeles")


df_preview_3 = pd.DataFrame(X_test, columns=[f"Banda_{i+1}" for i in range(X_test.shape[1])])
df_preview_3["Clase"] = y_test
print(df_preview_3.sample(5))

##### Guardar los datos
# Guardar arrays
np.savez_compressed("../DATA/OUTPUT/NUMPY/Xy_train_3.npz", X=X_train, y=y_train)
np.savez_compressed("../DATA/OUTPUT/NUMPY/Xy_val_3.npz", X=X_val, y=y_val)
np.savez_compressed("../DATA/OUTPUT/NUMPY/Xy_test_3.npz", X=X_test, y=y_test)

print("✅ Arrays guardados en formato .npz")

# Cargar arrays
train_data = np.load("../DATA/OUTPUT/NUMPY/Xy_train_3.npz")
X_train = train_data["X"]
y_train = train_data["y"]

val_data = np.load("../DATA/OUTPUT/NUMPY/Xy_val_3.npz")
X_val = val_data["X"]
y_val = val_data["y"]

test_data = np.load("../DATA/OUTPUT/NUMPY/Xy_test_3.npz")
X_test = test_data["X"]
y_test = test_data["y"]

print("✅ Datos cargados desde archivo .npz")

# Expandir y escalar atributos
X_train_exp = expandir_atributos_rgb(X_train)
X_val_exp   = expandir_atributos_rgb(X_val)
X_test_exp  = expandir_atributos_rgb(X_test)

# Añadir atributos adicionales: Excess Green y CIVE
def agregar_atributos_extras(X):
    R = X[:, 0]
    G = X[:, 1]
    B = X[:, 2]
    exg = 2 * G - R - B
    cive = 0.441 * R - 0.811 * G + 0.385 * B + 18.78745
    return np.column_stack([X, exg, cive])

X_train_exp = agregar_atributos_extras(X_train_exp)
X_val_exp   = agregar_atributos_extras(X_val_exp)
X_test_exp  = agregar_atributos_extras(X_test_exp)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_exp = scaler.fit_transform(X_train_exp)
X_val_exp = scaler.transform(X_val_exp)
X_test_exp = scaler.transform(X_test_exp)

# 🔁 K-Folds con datos de entrenamiento + validación
X_kfold = np.concatenate([X_train_exp, X_val_exp])
y_kfold = np.concatenate([y_train, y_val])

# Sin expandir atributos 
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train_exp = scaler.fit_transform(X_train)s
# X_val_exp = scaler.transform(X_val)
# X_test_exp = scaler.transform(X_test)

# # 🔁 K-Folds con datos de entrenamiento + validación
# X_kfold = np.concatenate([X_train_exp, X_val_exp])
# y_kfold = np.concatenate([y_train, y_val])


############### Dimensiones de los datasets a utilizar

# --- 0) Formas brutas (RGB sin expandir)
print("Formas originales (RGB):")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape},   y_val:   {y_val.shape}")
print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")

# --- 1) Expansión de atributos (R,G,B -> + (R-G,R-B,G-B) = 6)

print("\nTras expandir (RGB + diferencias = 6 variables):")
print(f"X_train_exp: {X_train_exp.shape}")
print(f"X_val_exp:   {X_val_exp.shape}")
print(f"X_test_exp:  {X_test_exp.shape}")

# --- 2) Índices espectrales (ExG y CIVE) => 8 variables finales
print("\nTras añadir ExG y CIVE (8 variables):")
print(f"X_train_exp: {X_train_exp.shape}")
print(f"X_val_exp:   {X_val_exp.shape}")
print(f"X_test_exp:  {X_test_exp.shape}")

# --- 3) Escalado (no cambia dimensiones)

print("\nTras escalar (mismas dimensiones):")
print(f"X_train_exp: {X_train_exp.shape}")
print(f"X_val_exp:   {X_val_exp.shape}")
print(f"X_test_exp:  {X_test_exp.shape}")

# --- 4) Conjunto para KFold (train+val) – rama “todas las variables”
print("\nKFold (todas las variables):")
print(f"X_kfold: {X_kfold.shape}, y_kfold: {y_kfold.shape}")

############################# PCA ###################################################

from pca_utils import aplicar_pca

# Aplicar PCA
X_train_pca, X_val_pca, X_test_pca, X_kfold_pca, pca = aplicar_pca(
    X_train_exp, X_val_exp, X_test_exp, X_kfold,
    y_train=y_train,
    n_components=0.95,
    plot_varianza=True,
    plot_2d=False,
    plot_3d=False,
    save_image=False  # solo se activa si plot_2d = True
)

print("\nTras PCA (95% varianza):")
print(f"n_components_: {pca.n_components_}")
print(f"X_train_pca:  {X_train_pca.shape}")
print(f"X_val_pca:    {X_val_pca.shape}")
print(f"X_test_pca:   {X_test_pca.shape}")
print(f"X_kfold_pca:  {X_kfold_pca.shape}")

########################### 1. KNN ##################################################


from knn_optim import optimizar_knn_kfold, entrenar_knn, evaluar_knn, clasificar_ventana_raster_knn, explorar_k_vecinos

explorar_k_vecinos(X_train, y_train, X_val, y_val, k_values=[3, 5, 7, 9], sample_frac=0.03)

# -----------------------------------------
# 1. Optimización de hiperparámetros (KFold)
# -----------------------------------------

best_params = optimizar_knn_kfold(X_kfold, y_kfold, n_trials=15, n_splits=3,sample_frac=0.3)
# con sample_frac=0.03 {'n_neighbors': 20, 'weights': 'uniform'}
# con sample_frac=0.3 
best_params = {'n_neighbors': 20, 'weights': 'distance'}


# -----------------------------------------
# 2. Entrenamiento final con mejores params
# -----------------------------------------

clf_knn = entrenar_knn(X_kfold, y_kfold, best_params=best_params)

# -----------------------------------------
# 3. Evaluación en test
# -----------------------------------------

feature_names = ["R", "G", "B", "R-G", "R-B", "G-B", "ExG", "CIVE"]
evaluar_knn(clf_knn, X_test_exp, y_test, clase_dict)

# -----------------------------------------
# 4. Clasificación en ventana del raster
# -----------------------------------------

import rasterio
from rasterio.windows import Window

# Parámetros de la ventana (ajústalos si lo deseas)
col_off, row_off = 15000, 15000  # píxeles de desplazamiento desde la esquina superior izquierda
width, height = 1500, 1500       # tamaño de la ventana

# Leer ventana desde raster
with rasterio.open(raster_path_1) as src:
    print(f"{src.count} bandas | Tamaño: {src.width} x {src.height}")
    raster_array = src.read(window=Window(col_off, row_off, width, height))

clasificar_ventana_raster_knn(clf_knn, scaler, raster_array, clase_dict)

# -----------------------------------------
# 0. Aplicar PCA
# -----------------------------------------
# 1. Optimización de hiperparámetros (KFold)
# -----------------------------------------
best_params = optimizar_knn_kfold(X_kfold_pca, y_kfold, n_trials=15, n_splits=3, sample_frac=0.3)
best_params = {'n_neighbors': 20, 'weights': 'uniform'}

# -----------------------------------------
# 2. Entrenamiento final con mejores params
# -----------------------------------------
clf_knn = entrenar_knn(X_kfold_pca, y_kfold, best_params=best_params)

# -----------------------------------------
# 3. Evaluación en test
# -----------------------------------------
evaluar_knn(clf_knn, X_test_pca, y_test, clase_dict)

# -----------------------------------------
# 4. Clasificación en ventana del raster
# -----------------------------------------
import rasterio
from rasterio.windows import Window

col_off, row_off = 15000, 15000
width, height = 1500, 1500

with rasterio.open(raster_path_1) as src:
    print(f"{src.count} bandas | Tamaño: {src.width} x {src.height}")
    raster_array = src.read(window=Window(col_off, row_off, width, height))

# PCA debe aplicarse a los datos raster también
# Esto implica modificar la función clasificar_ventana_raster_knn para aceptar `pca`

clasificar_ventana_raster_knn(clf_knn, scaler, raster_array, clase_dict, pca=pca)

########################### 2. Arboles de decisión ##################################

from decision_tree_optim import (
    entrenar_y_seleccionar_mejor_arbol,
    optimizar_arbol_con_kfold,
    entrenar_arbol_final,
    evaluar_arbol,
    clasificar_ventana_raster_dt
)

entrenar_y_seleccionar_mejor_arbol(X_train, y_train, X_val, y_val, max_depth_range=range(2, 10))
# max_depth = 8

# -----------------------------------------
# 1. Optimización de hiperparámetros (KFold)
# -----------------------------------------
best_params = optimizar_arbol_con_kfold(X_kfold, y_kfold, n_trials=15, n_splits=3)
best_params = {'max_depth': 18, 'min_samples_split': 8, 'min_samples_leaf': 7}

# -----------------------------------------
# 2. Entrenamiento final con mejores params
# -----------------------------------------
modelo_dt = entrenar_arbol_final(X_kfold, y_kfold, best_params)

# -----------------------------------------
# 3. Evaluación en test
# -----------------------------------------
evaluar_arbol(modelo_dt, X_test_exp, y_test, clase_dict, nombre_set="Test")

# -----------------------------------------
# 4. Clasificación en ventana del raster
# -----------------------------------------
import rasterio
from rasterio.windows import Window

# Parámetros de la ventana (ajústalos si lo deseas)
col_off, row_off = 15000, 15000  # píxeles de desplazamiento desde la esquina superior izquierda
width, height = 1500, 1500       # tamaño de la ventana

# Leer ventana desde raster
with rasterio.open(raster_path_1) as src:
    print(f"{src.count} bandas | Tamaño: {src.width} x {src.height}")
    raster_array = src.read(window=Window(col_off, row_off, width, height))

clasificar_ventana_raster_dt(modelo_dt, scaler, raster_array, clase_dict)

# -----------------------------------------
# 0. Aplicar PCA
# -----------------------------------------
# 1. Optimización de hiperparámetros (KFold)
# -----------------------------------------
best_params = optimizar_arbol_con_kfold(X_kfold_pca, y_kfold, n_trials=15, n_splits=3)
# 🔍 Mejores hiperparámetros encontrados: 
best_params = {'max_depth': 25, 'min_samples_split': 2, 'min_samples_leaf': 13}

# -----------------------------------------
# 2. Entrenamiento final con mejores params
# -----------------------------------------
modelo_dt = entrenar_arbol_final(X_kfold_pca, y_kfold, best_params)

# -----------------------------------------
# 3. Evaluación en test
# -----------------------------------------
evaluar_arbol(modelo_dt, X_test_pca, y_test, clase_dict, nombre_set="Test")

# -----------------------------------------
# 4. Clasificación en ventana del raster
# -----------------------------------------
import rasterio
from rasterio.windows import Window

col_off, row_off = 15000, 15000
width, height = 1500, 1500

with rasterio.open(raster_path_1) as src:
    print(f"{src.count} bandas | Tamaño: {src.width} x {src.height}")
    raster_array = src.read(window=Window(col_off, row_off, width, height))

clasificar_ventana_raster_dt(modelo_dt, scaler, raster_array, clase_dict, pca=pca)

########################### 3. Random Forest ########################################
from random_forest_optim import (
    optimizar_random_forest_kfold,
    entrenar_random_forest,
    evaluar_random_forest,
    clasificar_ventana_raster_rf
)

# -----------------------------------------
# 1. Optimización de hiperparámetros (KFold)
# -----------------------------------------
best_params = optimizar_random_forest_kfold(X_kfold, y_kfold, n_trials=15, n_splits=3, sample_frac=0.15)
# {'n_estimators': 120, 'max_depth': 15, 'min_samples_split': 12, 'min_samples_leaf': 8, 'max_features': 'sqrt'}
best_params = {'n_estimators': 120, 'max_depth': 15, 'min_samples_split': 12, 'min_samples_leaf': 8, 'max_features': 'sqrt'}

# -----------------------------------------
# 2. Entrenamiento final con mejores params
# -----------------------------------------
clf_rf = entrenar_random_forest(X_kfold, y_kfold, best_params=best_params, sample_frac=0.15) # Puede aumentar la fracción
# -----------------------------------------
# 3. Evaluación en test
# ----------------------------------------- 
# = ["R", "G", "B", "R-G", "R-B", "G-B", "ExG", "CIVE"]

evaluar_random_forest(clf_rf, X_test_exp, y_test, clase_dict, nombre_set="Test")

# -----------------------------------------
# 4. Clasificación en ventana del raster
# -----------------------------------------
import rasterio
from rasterio.windows import Window

# Parámetros de la ventana
col_off, row_off = 15000, 15000  # píxeles de desplazamiento desde la esquina superior izquierda
width, height = 1500, 1500       # tamaño de la ventana

# Leer ventana desde raster
with rasterio.open(raster_path_1) as src:
    print(f"{src.count} bandas | Tamaño: {src.width} x {src.height}")
    raster_array = src.read(window=Window(col_off, row_off, width, height))

clasificar_ventana_raster_rf(clf_rf, scaler, raster_array, clase_dict)

# -----------------------------------------
# 0. Limpiar temporales
from limpieza import limpiar_temporales
limpiar_temporales()
# -----------------------------------------

# -----------------------------------------
# 0. Aplicar PCA
# -----------------------------------------
# 1. Optimización de hiperparámetros (KFold)
# -----------------------------------------
best_params = optimizar_random_forest_kfold(X_kfold_pca, y_kfold, n_trials=15, n_splits=3, sample_frac=0.15)
# {'n_estimators': 168, 'max_depth': 18, 'min_samples_split': 14, 'min_samples_leaf': 1, 'max_features': 'log2'}
best_params = {'n_estimators': 168, 'max_depth': 18, 'min_samples_split': 14, 'min_samples_leaf': 1, 'max_features': 'log2'}

# -----------------------------------------
# 2. Entrenamiento final con mejores params
# -----------------------------------------
clf_rf = entrenar_random_forest(X_train_pca, y_train, best_params=best_params, sample_frac=0.15)

# -----------------------------------------
# 3. Evaluación en test
# -----------------------------------------
evaluar_random_forest(clf_rf, X_kfold_pca, y_kfold, clase_dict, nombre_set="Test")

# -----------------------------------------
# 4. Clasificación en ventana del raster
# -----------------------------------------
import rasterio
from rasterio.windows import Window

col_off, row_off = 15000, 15000
width, height = 1500, 1500

with rasterio.open(raster_path_1) as src:
    print(f"{src.count} bandas | Tamaño: {src.width} x {src.height}")
    raster_array = src.read(window=Window(col_off, row_off, width, height))

clasificar_ventana_raster_rf(clf_rf, scaler, raster_array, clase_dict, pca=pca)

########################### 4. XGBOOST ##############################################
from xgboost_model import (
    optimizar_xgboost_con_kfold,
    entrenar_xgboost,
    evaluar_xgboost,
    clasificar_ventana_raster_xgb
)

# ===============================
# 1. Optimización de hiperparámetros
# ===============================
best_params = optimizar_xgboost_con_kfold(X_kfold, y_kfold, n_trials=30, n_splits=3)

#---- 15 trials 
# best_params = {'max_depth': 14, 'learning_rate': 0.059239920288551226, 'n_estimators': 382, 'gamma': 3.4936110505596436, 'min_child_weight': 4, 'subsample': 0.7290635275808393, 'colsample_bytree': 0.6301144278209232}
#---- 20 trials
# {'max_depth': 13, 'learning_rate': 0.13915317865144, 'n_estimators': 161, 'gamma': 4.2052404633881695, 'min_child_weight': 5, 'subsample': 0.9988894746908422, 'colsample_bytree': 0.6742066769057147}
#---- Sin variables adicionales (solo RGB)
# {'max_depth': 14, 'learning_rate': 0.2954483960193896, 'n_estimators': 218, 'gamma': 1.3235914976042777, 'min_child_weight': 5, 'subsample': 0.5063984391111775, 'colsample_bytree': 0.9549236625204564}
#---- 30 trials
best_params = best_params = {'max_depth': 15, 'learning_rate': 0.015393548526718703, 'n_estimators': 487, 'gamma': 2.2861927733513525, 'min_child_weight': 4, 'subsample': 0.8739758494721231, 'colsample_bytree': 0.836102204957935}

# ===============================
# 2. Entrenamiento final
# ===============================
clf_xgb = entrenar_xgboost(X_kfold, y_kfold, best_params=best_params)

# ===============================
# 3. Evaluación final en test
# ===============================
feature_names = ["R", "G", "B", "R-G", "R-B", "G-B", "ExG", "CIVE"]
evaluar_xgboost(clf_xgb, X_test_exp, y_test, clase_dict, feature_names=feature_names)

# -----------------------------------------
# Clasificación en ventana del raster
# -----------------------------------------
import os
import rasterio
from rasterio.windows import Window

# ===============================
# 4. Clasificación raster (SIN PCA)
# ===============================
# Parámetros de la ventana
col_off, row_off = 15000, 15000
width, height = 1500, 1500

# Leer ventana desde raster y obtener transform/crs correctos
with rasterio.open(raster_path_1) as src:
    print(f"{src.count} bandas | Tamaño: {src.width} x {src.height}")
    window = Window(col_off, row_off, width, height)
    raster_array = src.read(window=window)
    transform_win = src.window_transform(window)  # 🔹 transform de la ventana
    crs = src.crs

# Carpeta salida
out_dir = "../DATA/OUTPUT/SHP/"
os.makedirs(out_dir, exist_ok=True)
shp_path = os.path.join(out_dir, "clasificacion_ventana_xgb.shp")

# Clasificación y exportación a SHP
Y_pred_img = clasificar_ventana_raster_xgb(
    modelo=clf_xgb,
    scaler=scaler,
    raster_array=raster_array,
    clase_dict=clase_dict,
    pca=None,                          # sin PCA
    transform=transform_win,           # 🔹 necesario para georreferenciar la ventana
    crs=crs,                           # 🔹 CRS original
    shp_out=shp_path,                  # 🔹 ruta de salida Shapefile
    min_area=2.0                       # opcional: elimina polígonos < 2 m² si CRS es métrico
)
print(f"✅ Shapefile exportado en: {shp_path}")

# -----------------------------------------
# 0. Aplicar PCA
# -----------------------------------------
# 1. Optimización de hiperparámetros (KFold)
# -----------------------------------------
best_params = optimizar_xgboost_con_kfold(
    X_kfold_pca, y_kfold,
    n_trials=30, n_splits=3

)
# ---- 30 trials
# {'max_depth': 12, 'learning_rate': 0.1952054940174675, 'n_estimators': 178, 'gamma': 2.1358104174146004, 'min_child_weight': 4, 'subsample': 0.6087337378533421, 'colsample_bytree': 0.8070975942306197}
best_params = {'max_depth': 15, 'learning_rate': 0.18054608569965644, 'n_estimators': 382, 'gamma': 0.060482829276031325, 'min_child_weight': 4, 'subsample': 0.9463971570037508, 'colsample_bytree': 0.7804792354309387}

# -----------------------------------------
# 2. Entrenamiento final con mejores params
# -----------------------------------------
clf_xgb = entrenar_xgboost(X_train_pca, y_train, best_params=best_params)

# -----------------------------------------
# 3. Evaluación en test
# -----------------------------------------
evaluar_xgboost(clf_xgb, X_test_pca, y_test, clase_dict, feature_names=[f"PC{i+1}" for i in range(X_test_pca.shape[1])])

# ===============================
# 4. Clasificación raster (CON PCA)
# ===============================
with rasterio.open(raster_path_1) as src:
    window = Window(col_off, row_off, width, height)
    raster_array = src.read(window=window)
    transform_win = src.window_transform(window)
    crs = src.crs

shp_path_pca = os.path.join(out_dir, "clasificacion_ventana_xgb_pca.shp")

Y_pred_img_pca = clasificar_ventana_raster_xgb(
    modelo=clf_xgb,                    
    scaler=scaler,                     
    raster_array=raster_array,
    clase_dict=clase_dict,
    pca=pca,                           
    transform=transform_win,
    crs=crs,
    shp_out=shp_path_pca,
    min_area=2.0
)
print(f"✅ Shapefile (PCA) exportado en: {shp_path_pca}")


########################### 5. Light GBM ############################################
from lightgbm_model import (
    optimizar_lightgbm_kfold,
    entrenar_lightgbm,
    evaluar_lightgbm,
    clasificar_ventana_raster_lgb
)

# -----------------------------------------
# 0. Limpiar temporales
from limpieza import limpiar_temporales
limpiar_temporales()
# -----------------------------------------

# ----------------------------
# 1. Optimizar hiperparámetros con Optuna
# ----------------------------
best_params = optimizar_lightgbm_kfold(X_kfold, y_kfold, num_trials=25, num_folds=3)
best_params = {'learning_rate': 0.24614483982991803, 'max_depth': 7, 'num_leaves': 74, 'min_child_samples': 61, 'subsample': 0.9108765013712062, 'colsample_bytree': 0.8260117066804222}

# ----------------------------
# 2. Entrenar modelo con los mejores hiperparámetros
# ----------------------------
clf_lgb = entrenar_lightgbm(X_kfold, y_kfold, best_params=best_params)

# ----------------------------
# 3. Evaluar el modelo en test
# ----------------------------
feature_names = ["R", "G", "B", "R-G", "R-B", "G-B", "ExG", "CIVE"]
evaluar_lightgbm(clf_lgb, X_test_exp, y_test, clase_dict, feature_names=feature_names)

# ----------------------------
# 4. Clasificar ventana del raster
# ----------------------------
import rasterio
from rasterio.windows import Window

col_off, row_off = 15000, 15000
width, height = 1500, 1500

with rasterio.open(raster_path_1) as src:
    print(f"{src.count} bandas | Tamaño: {src.width} x {src.height}")
    raster_array = src.read(window=Window(col_off, row_off, width, height))

clasificar_ventana_raster_lgb(clf_lgb, scaler, raster_array, clase_dict)


# -----------------------------------------
# 0. Aplicar PCA
# -----------------------------------------
# 1. Optimizar hiperparámetros con PCA
# ----------------------------
best_params = optimizar_lightgbm_kfold(X_kfold_pca, y_kfold, num_trials=25, num_folds=3)
best_params = {'learning_rate': 0.1514449876320673, 'max_depth': 7, 'num_leaves': 44, 'min_child_samples': 83, 'subsample': 0.8420536269553477, 'colsample_bytree': 0.9495319914250514}

# ----------------------------
# 2. Entrenar modelo con los mejores hiperparámetros
# ----------------------------
clf_lgb = entrenar_lightgbm(X_kfold_pca, y_kfold, best_params=best_params)

# ----------------------------
# 3. Evaluar el modelo en test (usa X_test_pca)
# ----------------------------
evaluar_lightgbm(clf_lgb, X_test_pca, y_test, clase_dict, feature_names=[f"PC{i+1}" for i in range(X_test_pca.shape[1])])

# ----------------------------
# 4. Clasificar ventana del raster
# ----------------------------
import rasterio
from rasterio.windows import Window

col_off, row_off = 15000, 15000
width, height = 1500, 1500

with rasterio.open(raster_path_1) as src:
    print(f"{src.count} bandas | Tamaño: {src.width} x {src.height}")
    raster_array = src.read(window=Window(col_off, row_off, width, height))

Y_pred_img = clasificar_ventana_raster_lgb(clf_lgb, scaler, raster_array, clase_dict, pca=pca)

########################### 6. Regresión logistica ##################################

from logistic_regression import (
    explorar_valores_C_logreg,
    optimizar_logreg_kfold,
    entrenar_logreg,
    evaluar_logreg,
    clasificar_ventana_raster_logreg
)

# ----------------------------
# 1. Optimización manual explorando C
# ----------------------------

modelo, best_C = explorar_valores_C_logreg(X_train_exp, y_train, X_val_exp, y_val)

# ----------------------------
# 2. Alternativa: Optuna con KFold sobre training+val
# ----------------------------
best_params = optimizar_logreg_kfold(X_kfold, y_kfold, n_trials=30)
best_params =  {'C': 0.004536054306746058}

modelo = entrenar_logreg(X_kfold, y_kfold, best_params)

# ----------------------------
# 3. Evaluación en test
# ----------------------------

evaluar_logreg(modelo, X_test_exp, y_test, clase_dict, nombre_set="Test")

# ----------------------------
# 4. Clasificación raster
# ----------------------------
import rasterio
from rasterio.windows import Window

col_off, row_off = 15000, 15000
width, height = 1500, 1500

with rasterio.open(raster_path_1) as src:
    print(f"{src.count} bandas | Tamaño: {src.width} x {src.height}")
    raster_array = src.read(window=Window(col_off, row_off, width, height))

Y_pred_img = clasificar_ventana_raster_logreg(modelo, scaler, raster_array, clase_dict)

# -----------------------------------------
# 0. Aplicar PCA
# -----------------------------------------
# 1. Optimización con Optuna usando datos PCA
best_params = optimizar_logreg_kfold(X_kfold_pca, y_kfold, n_trials=30)
best_params = {'C': 7.972281627890308}
# -----------------------------------------

# -----------------------------------------
# 2. Entrenamiento final
# -----------------------------------------

modelo = entrenar_logreg(X_kfold_pca, y_kfold, best_params)

# -----------------------------------------
# 3. Evaluación
# -----------------------------------------

evaluar_logreg(modelo, X_test_pca, y_test, clase_dict, nombre_set="Test")

# -----------------------------------------
# 4. Clasificación en ventana
# -----------------------------------------
import rasterio
from rasterio.windows import Window

col_off, row_off = 15000, 15000
width, height = 1500, 1500

with rasterio.open(raster_path_1) as src:
    print(f"{src.count} bandas | Tamaño: {src.width} x {src.height}")
    raster_array = src.read(window=Window(col_off, row_off, width, height))

Y_pred_img = clasificar_ventana_raster_logreg(modelo, scaler, raster_array, clase_dict, pca=pca)

########################### 7. Naive Bayes ##########################################

from naive_bayes import (
    entrenar_y_evaluar_naive_bayes,
    validar_naive_bayes_kfold,
    evaluar_naive_bayes,
    clasificar_ventana_raster_nb
)

# -------------------------------
# 1. Entrenamiento y evaluación en validación
# -------------------------------
modelo_nb = entrenar_y_evaluar_naive_bayes(X_kfold, y_kfold)

# -------------------------------
# 2. Validación cruzada
# -------------------------------
_ = validar_naive_bayes_kfold(X_kfold, y_kfold, n_splits=5, sample_frac=0.5)

# -------------------------------
# 3. Evaluación en test
# -------------------------------
evaluar_naive_bayes(modelo_nb, X_test_exp, y_test, clase_dict)

# -------------------------------
# 4. Clasificación en ventana del raster
# -------------------------------
import rasterio
from rasterio.windows import Window

col_off, row_off = 15000, 15000
width, height = 1500, 1500

with rasterio.open(raster_path_1) as src:
    print(f"{src.count} bandas | Tamaño: {src.width} x {src.height}")
    raster_array = src.read(window=Window(col_off, row_off, width, height))

Y_pred_img = clasificar_ventana_raster_nb(modelo_nb, scaler, raster_array, clase_dict)


# -----------------------------------------
# 0. Aplicar PCA
# -----------------------------------------
# 1. Entrenamiento y evaluación en validación con PCA
# -------------------------------
modelo_nb = entrenar_y_evaluar_naive_bayes(X_kfold_pca, y_kfold)

# -------------------------------
# 2. Validación cruzada con PCA
# -------------------------------
_ = validar_naive_bayes_kfold(X_kfold_pca, y_kfold, n_splits=5, sample_frac=0.5)

# -------------------------------
# 3. Evaluación en test
# -------------------------------
evaluar_naive_bayes(modelo_nb, X_test_pca, y_test, clase_dict)

# -------------------------------
# 4. Clasificación raster con PCA
# -------------------------------
import rasterio
from rasterio.windows import Window

col_off, row_off = 15000, 15000
width, height = 1500, 1500

with rasterio.open(raster_path_1) as src:
    print(f"{src.count} bandas | Tamaño: {src.width} x {src.height}")
    raster_array = src.read(window=Window(col_off, row_off, width, height))

Y_pred_img = clasificar_ventana_raster_nb(modelo_nb, scaler, raster_array, clase_dict, pca=pca)

########################### 8. Gradient Boosting ####################################
from gradient_boosting import (
    explorar_n_estimators,
    optimizar_gb_kfold,
    entrenar_gradient_boosting,
    evaluar_gradient_boosting,
    clasificar_ventana_raster_gb
)

explorar_n_estimators(X_train, y_train, X_val, y_val, n_range=range(50, 301, 25), sample_frac=0.2)

# ---------------------------------------
# 1. Optimización de hiperparámetros
# ---------------------------------------
best_params = optimizar_gb_kfold(X_kfold, y_kfold, n_trials=20, n_splits=3,sample_frac=0.005)
## --- sample 0.001
# {'learning_rate': 0.0861869634442195, 'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 2}
## --- sample 0.005
best_params = {'learning_rate': 0.15540322106059468, 'max_depth': 3, 'min_samples_split': 3, 'min_samples_leaf': 2}

# ---------------------------------------
# 2. Entrenamiento final con mejores hiperparámetros
# ---------------------------------------
clf_gb = entrenar_gradient_boosting(X_kfold, y_kfold, best_params, sample_frac=0.05, verbose=True)

# ---------------------------------------
# 3. Evaluación en test
# ---------------------------------------
feature_names = ["R", "G", "B", "R-G", "R-B", "G-B", "ExG", "CIVE"]
evaluar_gradient_boosting(clf_gb, X_test_exp, y_test, clase_dict, nombre_set="Test")

# ---------------------------------------
# 4. Clasificación de ventana raster
# ---------------------------------------
import rasterio
from rasterio.windows import Window

col_off, row_off = 15000, 15000
width, height = 1500, 1500

with rasterio.open(raster_path_1) as src:
    print(f"{src.count} bandas | Tamaño: {src.width} x {src.height}")
    raster_array = src.read(window=Window(col_off, row_off, width, height))

Y_pred_img = clasificar_ventana_raster_gb(clf_gb, scaler, raster_array, clase_dict)


# -----------------------------------------
# 0. Aplicar PCA
# -----------------------------------------
# ---------------------------------------
# 1. Optimización con PCA
# ---------------------------------------
best_params = optimizar_gb_kfold(X_kfold_pca, y_kfold, n_trials=20, n_splits=3, sample_frac=0.05)
best_params = {'learning_rate': 0.15969892832889018, 'max_depth': 4, 'min_samples_split': 3, 'min_samples_leaf': 2}

# ---------------------------------------
# 2. Entrenamiento final con PCA
# ---------------------------------------
clf_gb = entrenar_gradient_boosting(X_kfold_pca, y_kfold, best_params,sample_frac=0.05, verbose=True)

# ---------------------------------------
# 3. Evaluación en test con PCA
# ---------------------------------------
evaluar_gradient_boosting(clf_gb, X_test_pca, y_test, clase_dict, nombre_set="Test")

# ---------------------------------------
# 4. Clasificación en raster con PCA
# ---------------------------------------
import rasterio
from rasterio.windows import Window

col_off, row_off = 15000, 15000
width, height = 1500, 1500

with rasterio.open(raster_path_1) as src:
    print(f"{src.count} bandas | Tamaño: {src.width} x {src.height}")
    raster_array = src.read(window=Window(col_off, row_off, width, height))

Y_pred_img = clasificar_ventana_raster_gb(clf_gb, scaler, raster_array, clase_dict, pca=pca)

########################### 9. SVM ##################################################
from svm_model import (
    explorar_valores_c,
    optimizar_svm_con_kfold,
    entrenar_svm_final,
    evaluar_svm,
    clasificar_ventana_raster_svm,
)

explorar_valores_c(X_train, y_train, X_val, y_val, sample_frac=0.01)

# ---------------------------------------
# 1. Exploración visual (opcional)
# ---------------------------------------
# best_model, best_C = explorar_valores_c(X_train_exp, y_train, X_val_exp, y_val, sample_frac=0.25)

# ---------------------------------------
# 2. Optimización con KFold
# ---------------------------------------
best_params = optimizar_svm_con_kfold(X_kfold, y_kfold, n_trials=15, n_splits=3, sample_frac=0.01)
# --- frac + 0.005 {'C': 0.236709308152586, 'gamma': 0.13127538330020944}
# --- frac 0.01 {'C': 0.19139334285921508, 'gamma': 0.08829169932676562}
best_params = {'C': 0.19139334285921508, 'gamma': 0.08829169932676562}

# ---------------------------------------
# 3. Entrenamiento final
# ---------------------------------------
clf_svm = entrenar_svm_final(X_kfold, y_kfold, best_params, sample_frac=0.0005, verbose=True)

# ---------------------------------------
# 4. Evaluación en test
# ---------------------------------------
evaluar_svm(clf_svm, X_test_exp, y_test, clase_dict, nombre_set="Test", sample_frac=0.0005, batch_size=5000)

# ---------------------------------------
# 5. Clasificación en raster
# ---------------------------------------
import rasterio
from rasterio.windows import Window

col_off, row_off = 15000, 15000
width, height = 1500, 1500

with rasterio.open(raster_path_1) as src:
    print(f"{src.count} bandas | Tamaño: {src.width} x {src.height}")
    raster_array = src.read(window=Window(col_off, row_off, width, height))

Y_pred_img = clasificar_ventana_raster_svm(clf_svm, scaler, raster_array, clase_dict, batch_size=50000)


# -----------------------------------------
# 0. Aplicar PCA
# -----------------------------------------
# 1. Optimización con KFold y PCA
# ---------------------------------------
best_params = optimizar_svm_con_kfold(X_kfold_pca, y_kfold, n_trials=15, n_splits=3, sample_frac=0.01)
best_params ={'C': 0.10928463102179635, 'gamma': 0.17867297028690834}

# ---------------------------------------
# 2. Entrenamiento final con PCA
# ---------------------------------------
clf_svm = entrenar_svm_final(X_kfold_pca, y_kfold, best_params, sample_frac=0.0005, verbose=True)

# ---------------------------------------
# 3. Evaluación con PCA
# ---------------------------------------
evaluar_svm(clf_svm, X_test_pca, y_test, clase_dict, nombre_set="Test", sample_frac=0.0005, batch_size=5000)

# ---------------------------------------
# 4. Clasificación raster con PCA
# ---------------------------------------
import rasterio
from rasterio.windows import Window

col_off, row_off = 15000, 15000
width, height = 1500, 1500

with rasterio.open(raster_path_1) as src:
    print(f"{src.count} bandas | Tamaño: {src.width} x {src.height}")
    raster_array = src.read(window=Window(col_off, row_off, width, height))

Y_pred_img = clasificar_ventana_raster_svm(clf_svm, scaler, raster_array, clase_dict, pca=pca)
