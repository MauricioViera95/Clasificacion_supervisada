# ========================================================================
# lightgbm_model.py
# ------------------------------------------------------------------------
# Entrenamiento y evaluación de LightGBM para clasificar suelo permeable
# y no permeable a partir de ortofotos y datos vectoriales procesados.
# Incluye búsqueda de hiperparámetros, análisis visual y aplicación
# del modelo sobre una ventana del raster.
#
# @author   Mauricio Viera-Torres <ronnyvieramt95@gmail.com>
# @address  La Tola (Ecuador)
# @version  1.0.0 (2025-06-30)
# ========================================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import optuna
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score, f1_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


# ===============================
# Optimización con validación cruzada y Optuna
# ===============================
def optimizar_lightgbm_kfold(X, y, num_trials=30, num_folds=3):
    def objective(trial):
        try:
            pos_weight = len(y[y == 0]) / len(y[y == 1])

            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "verbosity": -1,
                "device": "gpu",
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "num_leaves": trial.suggest_int("num_leaves", 15, 100),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "scale_pos_weight": pos_weight
            }

            skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
            scores = []

            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_train_cv, X_val_cv = X[train_idx], X[val_idx]
                y_train_cv, y_val_cv = y[train_idx], y[val_idx]

                clf = lgb.LGBMClassifier(**params)
                clf.fit(X_train_cv, y_train_cv)

                preds = clf.predict(X_val_cv)
                acc = accuracy_score(y_val_cv, preds)
                scores.append(acc)

                print(f"🌀 Fold {fold + 1}/{num_folds} - Accuracy: {acc:.3f}")

            avg_score = np.mean(scores)
            print(f"📊 Promedio de accuracy: {avg_score:.3f}")
            return avg_score

        except Exception as e:
            print(f"⚠️ Error en trial: {e}")
            raise optuna.exceptions.TrialPruned()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=num_trials)

    print("\n🔍 Mejores hiperparámetros encontrados:")
    print(study.best_params)
    return study.best_params


# ===============================
# Entrenamiento final con mejores hiperparámetros
# ===============================
def entrenar_lightgbm(X_train, y_train, best_params=None):
    if best_params is None:
        best_params = {
            "learning_rate": 0.1,
            "max_depth": 6,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }

    pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    best_params["scale_pos_weight"] = pos_weight

    clf = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        boosting_type="gbdt",
        device="gpu",
        verbosity=-1,
        random_state=42,
        **best_params
    )

    clf.fit(X_train, y_train)
    return clf


# ===============================
# Evaluación del modelo con visualizaciones mejoradas
# ===============================
import matplotlib.ticker as mtick
from time import time
from tqdm import tqdm

def evaluar_lightgbm(modelo, X, y, clase_dict, feature_names=None, nombre_set="Test"):
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

    # 📊 Importancia de atributos (estética impactante)
    if feature_names is not None:
        importances = modelo.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        keys = [feature_names[i] for i in sorted_idx]
        values = importances[sorted_idx]
        cmap_bar = plt.cm.plasma(np.linspace(0.2, 0.85, len(keys)))

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(keys, values, color=cmap_bar, edgecolor='black', alpha=0.8, linewidth=0.5)
        ax.invert_yaxis()
        ax.set_title("Importancia de atributos (LightGBM)", fontsize=13)
        ax.set_xlabel("Ganancia relativa", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        plt.tight_layout()
        plt.show()



# ===============================
# Clasificación sobre una ventana del raster
# ===============================
def clasificar_ventana_raster_lgb(modelo, scaler, raster_array, clase_dict, pca=None):
    d, h, w = raster_array.shape
    X_window = raster_array.reshape([d, h * w]).T

    # Atributos RGB y derivados
    R, G, B = X_window[:, 0], X_window[:, 1], X_window[:, 2]
    excess_green = 2 * G - R - B
    cive = 0.441 * R - 0.811 * G + 0.385 * B + 18.78745

    # Atributos extendidos
    X_ext = np.column_stack([R, G, B, R - G, R - B, G - B, excess_green, cive])

    # Escalar
    X_scaled = scaler.transform(X_ext)

    # 🔸 Aplicar PCA si se proporciona
    if pca is not None:
        X_scaled = pca.transform(X_scaled)

    # Predicción
    y_pred_prob = modelo.predict_proba(X_scaled)[:, 1]
    y_pred = (y_pred_prob > 0.5).astype(int)
    Y_pred_img = y_pred.reshape((h, w))

    # Visualización
    class_labels = list(clase_dict.keys())
    class_colors = {
        "Permeable": "green",
        "No permeable": "lightgray"
    }
    custom_cmap = ListedColormap([class_colors[clase] for clase in class_labels])
    legend_patches = [mpatches.Patch(color=class_colors[clase], label=clase) for clase in class_labels]

    fig, axs = plt.subplots(1, 2, figsize=(14, 8))
    axs[0].imshow(Y_pred_img, cmap=custom_cmap)
    axs[0].set_title("Clasificación LightGBM")
    axs[0].axis('off')
    axs[0].legend(handles=legend_patches, loc='lower left', fontsize=10)

    rgb_img = np.transpose(raster_array[[0, 1, 2]], (1, 2, 0)).astype(np.uint8)
    axs[1].imshow(rgb_img)
    axs[1].set_title("Imagen RGB original")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

    return Y_pred_img