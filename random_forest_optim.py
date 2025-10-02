# ========================================================================
# random_forest_optim.py
# ------------------------------------------------------------------------
# Entrenamiento y evaluación de Random Forest para clasificar
# suelo permeable y no permeable usando ortofotos y datos vectoriales.
# Incluye búsqueda de hiperparámetros con Optuna, evaluación gráfica
# detallada y aplicación a una ventana del raster georreferenciado.
#
# @author   Mauricio Viera-Torres <ronnyvieramt95@gmail.com>
# @address  La Tola (Ecuador)
# @version  1.0.0 (2025-07-14)
# ========================================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn import tree
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import optuna
from tqdm import tqdm
import matplotlib.ticker as mtick
from time import time


def entrenar_y_seleccionar_mejor_arbol(X_train, y_train, X_val, y_val, max_depth_range=range(2, 26), sample_frac=None):
    if sample_frac is not None and 0 < sample_frac < 1.0:
        n = int(len(X_train) * sample_frac)
        idx = np.random.choice(len(X_train), n, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
        print(f"📉 Submuestreo activado: {n:,} muestras")

    scores_train, scores_val = [], []

    for md in tqdm(max_depth_range, desc="🔍 Explorando profundidades"):
        clf = tree.DecisionTreeClassifier(max_depth=md, random_state=42)
        clf.fit(X_train, y_train)
        y_pred_train = clf.predict(X_train)
        y_pred_val = clf.predict(X_val)

        scores_train.append(accuracy_score(y_train, y_pred_train))
        scores_val.append(accuracy_score(y_val, y_pred_val))

    best_idx = np.argmax(scores_val)
    best_depth = max_depth_range[best_idx]
    best_clf = tree.DecisionTreeClassifier(max_depth=best_depth, random_state=42)
    best_clf.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))

    plt.figure()
    plt.plot(max_depth_range, scores_train, label="Train")
    plt.plot(max_depth_range, scores_val, label="Validation")
    plt.axvline(x=best_depth, label=f'Mejor profundidad: {best_depth}', color='red')
    plt.title("Exactitud en función de la profundidad")
    plt.xlabel("Profundidad del árbol")
    plt.ylabel("Exactitud")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return best_clf, best_depth


def optimizar_random_forest_kfold(X, y, n_trials=30, n_splits=3, sample_frac=1.0):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 25),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"])
        }

        if sample_frac < 1.0:
            n_samples = int(X.shape[0] * sample_frac)
            idx = np.random.choice(X.shape[0], n_samples, replace=False)
            X_sub, y_sub = X[idx], y[idx]
        else:
            X_sub, y_sub = X, y

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in skf.split(X_sub, y_sub):
            X_train_cv, X_val_cv = X_sub[train_idx], X_sub[val_idx]
            y_train_cv, y_val_cv = y_sub[train_idx], y_sub[val_idx]

            clf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
            clf.fit(X_train_cv, y_train_cv)
            preds = clf.predict(X_val_cv)
            scores.append(accuracy_score(y_val_cv, preds))

        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("\n🔍 Mejores hiperparámetros encontrados:")
    print(study.best_params)
    return study.best_params


def entrenar_random_forest(X_train, y_train, best_params, sample_frac=None):
    if sample_frac is not None and 0 < sample_frac < 1.0:
        n_samples = int(len(X_train) * sample_frac)
        idx = np.random.choice(len(X_train), n_samples, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
        print(f"📉 Submuestreo activado: {n_samples:,} muestras")

    clf = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf


def predict_proba_batched(modelo, X, batch_size=10000):
    y_prob = []
    for i in tqdm(range(0, len(X), batch_size), desc="🔍 Probabilidades"):
        X_batch = X[i:i + batch_size]
        proba_batch = modelo.predict_proba(X_batch)[:, 1]
        y_prob.extend(proba_batch)
    return np.array(y_prob)


def evaluar_random_forest(modelo, X, y, clase_dict, nombre_set="Test", batch_size=10000):
    print(f"\n🔎 Evaluando modelo sobre {nombre_set}...")
    t0 = time()

    y_pred = []
    for i in tqdm(range(0, len(X), batch_size), desc="📊 Clasificando"):
        y_pred.extend(modelo.predict(X[i:i + batch_size]))
    y_pred = np.array(y_pred)

    t1 = time()
    print(f"✅ Evaluación completada en {t1 - t0:.2f} segundos.")

    y_prob = predict_proba_batched(modelo, X, batch_size) if hasattr(modelo, "predict_proba") else None

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

    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y, y_prob)
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


def clasificar_ventana_raster_rf(modelo, scaler, raster_array, clase_dict, pca=None):
    d, h, w = raster_array.shape
    X_window = raster_array.reshape([d, h * w]).T

    R, G, B = X_window[:, 0], X_window[:, 1], X_window[:, 2]
    excess_green = 2 * G - R - B
    cive = 0.441 * R - 0.811 * G + 0.385 * B + 18.78745

    X_ext = np.column_stack([R, G, B, R - G, R - B, G - B, excess_green, cive])
    X_scaled = scaler.transform(X_ext)

    if pca:
        X_scaled = pca.transform(X_scaled)

    y_pred = modelo.predict(X_scaled)
    Y_pred_img = y_pred.reshape((h, w))

    class_labels = list(clase_dict.keys())
    class_colors = {"Permeable": "green", "No permeable": "lightgray"}
    color_list = [class_colors[clase] for clase in class_labels]
    custom_cmap = ListedColormap(color_list)
    legend_patches = [mpatches.Patch(color=class_colors[clase], label=clase) for clase in class_labels]

    fig, axs = plt.subplots(1, 2, figsize=(14, 8))
    axs[0].imshow(Y_pred_img, cmap=custom_cmap)
    axs[0].set_title("Clasificación Random Forest")
    axs[0].axis('off')
    axs[0].legend(handles=legend_patches, loc='lower left', fontsize=10)

    rgb_img = np.transpose(raster_array[[0, 1, 2]], (1, 2, 0)).astype(np.uint8)
    axs[1].imshow(rgb_img)
    axs[1].set_title("Imagen RGB original")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

    return Y_pred_img
