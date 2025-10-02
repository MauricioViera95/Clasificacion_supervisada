# ========================================================================
# logistic_regression.py
# ------------------------------------------------------------------------
# Entrenamiento, evaluación y aplicación de Regresión Logística para
# clasificar suelo permeable y no permeable. Incluye búsqueda de C con
# validación cruzada, optimización con Optuna, evaluación gráfica y
# aplicación sobre raster. Integración con curvas ROC y métricas clave.
#
# @author   Mauricio Viera-Torres <ronnyvieramt95@gmail.com>
# @address  La Tola (Ecuador)
# @version  1.0.0 (2025-07-06)
# ========================================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import gc
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    cohen_kappa_score, f1_score, roc_curve, auc
)
from sklearn.model_selection import StratifiedKFold
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


# ------------------------------
def explorar_valores_C_logreg(X_train, y_train, X_val, y_val, c_values=None, sample_frac=None):
    if c_values is None:
        c_values = np.logspace(-3, 2, 20)

    if sample_frac is not None and 0 < sample_frac < 1.0:
        n = int(len(X_train) * sample_frac)
        idx = np.random.choice(len(X_train), n, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
        print(f"\U0001F4C9 Submuestreo aplicado: {n:,} muestras")

    scores_train, scores_val = [], []
    for C in tqdm(c_values, desc="\U0001F50D Explorando valores de C (LogReg)"):
        clf = LogisticRegression(C=C, max_iter=1000, solver='lbfgs')
        clf.fit(X_train, y_train)
        scores_train.append(accuracy_score(y_train, clf.predict(X_train)))
        scores_val.append(accuracy_score(y_val, clf.predict(X_val)))

    best_idx = np.argmax(scores_val)
    best_C = c_values[best_idx]

    plt.figure()
    plt.semilogx(c_values, scores_train, label="Train")
    plt.semilogx(c_values, scores_val, label="Validation")
    plt.axvline(best_C, color='red', linestyle='--', label=f"Mejor C: {best_C:.3f}")
    plt.xlabel("C (Regularización inversa)")
    plt.ylabel("Accuracy")
    plt.title("Búsqueda del mejor C (Regresión Logística)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    best_clf = LogisticRegression(C=best_C, max_iter=1000, solver='lbfgs')
    best_clf.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))

    return best_clf, best_C


# ------------------------------
def optimizar_logreg_kfold(X, y, n_trials=20, n_splits=3, sample_frac=None):
    def objective(trial):
        C = trial.suggest_float("C", 1e-3, 1e2, log=True)
        X_, y_ = (X, y)
        if sample_frac and 0 < sample_frac < 1.0:
            n = int(len(X) * sample_frac)
            idx = np.random.choice(len(X), n, replace=False)
            X_, y_ = X[idx], y[idx]

        clf = LogisticRegression(C=C, max_iter=1000, solver='lbfgs')
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return np.mean([
            accuracy_score(y[val], clf.fit(X[train], y[train]).predict(X[val]))
            for train, val in skf.split(X_, y_)
        ])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("\n\U0001F50D Mejores hiperparámetros encontrados:")
    print(study.best_params)
    return study.best_params


# ------------------------------
def entrenar_logreg(X_train, y_train, best_params):
    clf = LogisticRegression(**best_params, max_iter=1000, solver='lbfgs')
    clf.fit(X_train, y_train)
    return clf


# ------------------------------
def evaluar_logreg(modelo, X, y, clase_dict, nombre_set="Test"):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    from time import time
    from sklearn.metrics import (
        confusion_matrix, classification_report, accuracy_score,
        cohen_kappa_score, roc_curve, auc, f1_score
    )

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

    # 🎯 Curva ROC con estilo validado
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



# ------------------------------
def clasificar_ventana_raster_logreg(modelo, scaler, raster_array, clase_dict, pca=None):
    d, h, w = raster_array.shape
    X_window = raster_array.reshape([d, h * w]).T
    R, G, B = X_window[:, 0], X_window[:, 1], X_window[:, 2]
    exg = 2 * G - R - B
    cive = 0.441 * R - 0.811 * G + 0.385 * B + 18.78745
    X_ext = np.column_stack([R, G, B, R - G, R - B, G - B, exg, cive])
    X_scaled = scaler.transform(X_ext)
    if pca: X_scaled = pca.transform(X_scaled)
    y_pred = modelo.predict(X_scaled)
    Y_pred_img = y_pred.reshape((h, w))

    class_labels = list(clase_dict.keys())
    class_colors = {"Permeable": "green", "No permeable": "lightgray"}
    custom_cmap = ListedColormap([class_colors[c] for c in class_labels])
    legend_patches = [mpatches.Patch(color=class_colors[c], label=c) for c in class_labels]

    fig, axs = plt.subplots(1, 2, figsize=(14, 8))
    axs[0].imshow(Y_pred_img, cmap=custom_cmap)
    axs[0].set_title("Clasificación Regresión Logística")
    axs[0].axis('off')
    axs[0].legend(handles=legend_patches, loc='lower left', fontsize=10)

    rgb_img = np.transpose(raster_array[[0, 1, 2]], (1, 2, 0)).astype(np.uint8)
    axs[1].imshow(rgb_img)
    axs[1].set_title("Imagen RGB original")
    axs[1].axis('off')
    plt.tight_layout()
    plt.show()

    return Y_pred_img
