# ========================================================================
# xgboost_model.py
# ------------------------------------------------------------------------
# Entrenamiento y evaluación de XGBoost para clasificar
# suelo permeable y no permeable usando ortofotos y datos vectoriales.
# Incluye gráficas de validación y curva de aprendizaje.
#
# @author   Mauricio Viera-Torres <ronnyvieramt95@gmail.com>
# @address  La Tola (Ecuador)
# @version  1.0.0 (2025-06-29)
# ========================================================================

import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import optuna
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    cohen_kappa_score, f1_score, roc_curve, auc
)
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from sklearn.model_selection import StratifiedKFold
from optuna.samplers import TPESampler


def optimizar_xgboost_con_kfold(X, y, n_trials=30, n_splits=3):
    def objective(trial):
        pos_weight = len(y[y == 0]) / len(y[y == 1])
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'device': 'cuda',
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'scale_pos_weight': pos_weight
        }

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]

            clf = xgb.XGBClassifier(**params)
            clf.fit(
                X_train_cv, y_train_cv,
                eval_set=[(X_val_cv, y_val_cv)],
                verbose=False
            )
            preds = clf.predict(X_val_cv)
            acc = accuracy_score(y_val_cv, preds)
            scores.append(acc)
            print(f"🌀 Fold {fold_idx + 1}/{n_splits} - Accuracy: {acc:.3f}")

        mean_acc = np.mean(scores)
        print(f"📊 Promedio de accuracy: {mean_acc:.3f}")
        return mean_acc

    study = optuna.create_study(direction='maximize')
    study = optuna.create_study(
                                direction='maximize',
                                sampler=TPESampler(seed=42)
                                )       
    study.optimize(objective, n_trials=n_trials)

    print("\n🔍 Mejores hiperparámetros encontrados:")
    print(study.best_params)
    return study.best_params


def entrenar_xgboost(X_train, y_train, best_params=None):
    if best_params is None:
        best_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'gamma': 0,
            'min_child_weight': 1,
            'subsample': 1.0,
            'colsample_bytree': 1.0
        }

    pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    best_params['scale_pos_weight'] = pos_weight

    clf = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        device='cuda',
        random_state=42,
        **best_params
    )

    clf.fit(X_train, y_train, verbose=True)
    return clf


import matplotlib.ticker as mtick
from time import time
from tqdm import tqdm

def evaluar_xgboost(modelo, X, y, clase_dict, feature_names=None, nombre_set="Test"):
    print(f"\n🔎 Evaluando modelo sobre {nombre_set}...")
    t0 = time()

    y_pred_prob = modelo.predict_proba(X)[:, 1]

    thresholds = np.arange(0.3, 0.7, 0.01)
    f1_scores = [f1_score(y, (y_pred_prob > t).astype(int)) for t in thresholds]
    best_thresh = thresholds[np.argmax(f1_scores)]
    print(f"🔧 Threshold óptimo (según F1): {best_thresh:.2f}")

    y_pred = (y_pred_prob > best_thresh).astype(int)
    t1 = time()
    print(f"✅ Evaluación completada en {t1 - t0:.2f} segundos.")

    print(f"\n📋 Reporte de clasificación - {nombre_set}:")
    print(classification_report(y, y_pred, target_names=list(clase_dict.keys())))

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensibilidad = tp / (tp + fn)
    especificidad = tn / (tn + fp)

    def format_miles(x):
        return f'{x:,}'.replace(',', '\u202F')

    cmap = sns.color_palette("YlGnBu", as_cmap=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = sns.heatmap(
        cm,
        annot=[[format_miles(v) for v in row] for row in cm],
        fmt='',
        cmap=cmap,
        linewidths=0.6,
        linecolor='gray',
        xticklabels=clase_dict.keys(),
        yticklabels=clase_dict.keys(),
        cbar_kws={"format": mtick.FuncFormatter(lambda x, _: f"{int(x/1e6)}") if cm.max() > 1e6 else None}
    )

    ax.set_title(f"Matriz de confusión - {nombre_set}\nSensibilidad: {sensibilidad:.2%} | Especificidad: {especificidad:.2%}")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")

    if cm.max() > 1e6:
        colorbar = im.collections[0].colorbar
        colorbar.set_label("Escala ×10⁶", fontsize=10)

    plt.tight_layout()
    plt.show()

    acc = accuracy_score(y, y_pred)
    kappa = cohen_kappa_score(y, y_pred)
    print(f"\n✅ Accuracy: {acc:.3f}")
    print(f"✅ Kappa: {kappa:.3f}")

    # 🎯 Curva ROC
    fpr, tpr, _ = roc_curve(y, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}', linewidth=2.5)
    ax.fill_between(fpr, tpr, alpha=0.2)
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
    ax.set_xlabel("Tasa de falsos positivos", fontsize=11)
    ax.set_ylabel("Tasa de verdaderos positivos", fontsize=11)
    ax.set_title(f"Curva ROC - {nombre_set}", fontsize=13)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # 📊 Importancia de variables (estética impactante)
    booster = modelo.get_booster()
    fmap = {f"f{i}": name for i, name in enumerate(feature_names)} if feature_names else None
    importances = booster.get_score(importance_type='gain')
    if fmap:
        importances = {fmap.get(k, k): v for k, v in importances.items()}

    if importances:
        keys, values = zip(*sorted(importances.items(), key=lambda x: x[1], reverse=True))
        cmap_bar = plt.cm.plasma(np.linspace(0.2, 0.85, len(keys)))  # 🌈 barra con impacto

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(keys, values, color=cmap_bar, edgecolor='black', alpha=0.8, linewidth=0.5)
        ax.invert_yaxis()  # El más importante arriba

        ax.set_title("Importancia de atributos (XGBoost)", fontsize=13)
        ax.set_xlabel("Ganancia promedio", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        plt.tight_layout()
        plt.show()

# --- Librerías ---
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import geopandas as gpd
from shapely.geometry import shape
from rasterio.features import shapes


def _array_to_shapefile(class_array, transform, crs, shp_out,
                        class_map={0: "No permeable", 1: "Permeable"},
                        remove_holes=True, min_area=0.0):
    """
    Convierte un array 2D de clases (enteros) a polígonos y exporta a Shapefile.
    - class_array: np.ndarray 2D con valores de clase (int)
    - transform: affine del ráster original (rasterio.transform.Affine)
    - crs: CRS del ráster original (e.g., "EPSG:25830" o dict)
    - shp_out: ruta de salida .shp
    - class_map: diccionario opcional {valor_entero: nombre_clase}
    - remove_holes: si True, elimina hoyos interiores en los polígonos
    - min_area: filtra polígonos con área menor a este umbral (en unidades del CRS)
    """
    # shapes() itera (geom, value) en coordenadas del sistema del ráster
    geoms = []
    vals  = []
    for geom, val in shapes(class_array.astype(np.int32), mask=None, transform=transform):
        if val is None:
            continue
        geom_shape = shape(geom)
        if remove_holes:
            geom_shape = geom_shape.buffer(0)  # limpia geometrías (quita hoyos pequeños)
        if min_area > 0 and geom_shape.area < min_area:
            continue
        geoms.append(geom_shape)
        vals.append(int(val))

    if not geoms:
        # Si no hay geometrías (p.ej., ventana vacía)
        gpd.GeoDataFrame({"class_val": [], "class_name": []},
                         geometry=[], crs=crs).to_file(shp_out, driver="ESRI Shapefile")
        return

    df = gpd.GeoDataFrame(
        {
            "class_val": vals,
            "class_name": [class_map.get(v, f"Clase {v}") for v in vals],
        },
        geometry=geoms,
        crs=crs
    )

    df.to_file(shp_out, driver="ESRI Shapefile")


def clasificar_ventana_raster_xgb(modelo, scaler, raster_array, clase_dict,
                                  pca=None, transform=None, crs=None,
                                  shp_out=None, min_area=0.0):
    """
    Clasifica una ventana ráster con XGBoost y opcionalmente exporta el resultado a Shapefile.

    Parámetros clave:
    - raster_array: np.ndarray (C, H, W) con bandas en el orden [R, G, B, ...]
    - transform: affine del ráster original (OBLIGATORIO si shp_out no es None)
    - crs: CRS del ráster original (OBLIGATORIO si shp_out no es None)
    - shp_out: ruta de salida .shp para exportar polígonos de la clasificación
    - min_area: elimina polígonos más pequeños que este umbral (unidades CRS)

    Devuelve:
    - Y_pred_img: np.ndarray (H, W) con etiquetas predichas (0/1)
    """
    d, h, w = raster_array.shape
    X_window = raster_array.reshape([d, h * w]).T

    # Componentes RGB y atributos derivados
    R, G, B = X_window[:, 0], X_window[:, 1], X_window[:, 2]
    excess_green = 2 * G - R - B
    cive = 0.441 * R - 0.811 * G + 0.385 * B + 18.78745

    # Ensamble de features: ajusta si usas más bandas/atributos
    X_ext = np.column_stack([R, G, B, R - G, R - B, G - B, excess_green, cive])

    # Escalado y PCA (opcional)
    X_scaled = scaler.transform(X_ext)
    if pca is not None:
        X_scaled = pca.transform(X_scaled)

    # Predicción
    y_pred_prob = modelo.predict_proba(X_scaled)[:, 1]
    y_pred = (y_pred_prob > 0.5).astype(int)
    Y_pred_img = y_pred.reshape((h, w))

    # --- Visualización compacta y legible ---
    class_labels = list(clase_dict.keys())
    # Asegurar orden consistente: clave -> valor  (p.ej., {"Permeable":1,"No permeable":0})
    inv_map = {v: k for k, v in clase_dict.items()}  # 0/1 -> nombre
    class_colors = {inv_map.get(1, "Permeable"): "green",
                    inv_map.get(0, "No permeable"): "lightgray"}
    color_list = [class_colors[cls] for cls in class_labels]
    custom_cmap = ListedColormap(color_list)
    legend_patches = [mpatches.Patch(color=class_colors[cl], label=cl) for cl in class_labels]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5.6), dpi=140)
    axs[0].imshow(Y_pred_img, cmap=custom_cmap)
    axs[0].set_title("Clasificación XGBoost", fontsize=14, fontweight='bold', pad=8)
    axs[0].axis('off')
    axs[0].legend(handles=legend_patches, loc='lower left', fontsize=12,
                  frameon=True, facecolor='white')

    rgb_img = np.transpose(raster_array[[0, 1, 2]], (1, 2, 0)).astype(np.uint8)
    axs[1].imshow(rgb_img)
    axs[1].set_title("Imagen RGB original", fontsize=14, fontweight='bold', pad=8)
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

    # --- Exportar a Shapefile (opcional) ---
    if shp_out is not None:
        if transform is None or crs is None:
            raise ValueError("Para exportar a SHP debes proporcionar 'transform' y 'crs' del ráster original.")
        # Mapa de clases: 0/1 -> nombre
        class_map = {val: key for key, val in clase_dict.items()}
        _array_to_shapefile(
            class_array=Y_pred_img,
            transform=transform,
            crs=crs,
            shp_out=shp_out,
            class_map=class_map,
            remove_holes=True,
            min_area=min_area
        )

    return Y_pred_img