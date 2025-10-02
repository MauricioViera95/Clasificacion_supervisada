# ========================================================================
# naive_bayes.py
# ------------------------------------------------------------------------
# Entrenamiento, evaluación y aplicación de Naive Bayes para clasificar
# suelo permeable y no permeable. Incluye validación cruzada, métricas
# detalladas, curva ROC y clasificación sobre raster georreferenciado.
#
# @author   Mauricio Viera-Torres <ronnyvieramt95@gmail.com>
# @address  La Tola (Ecuador)
# @version  1.0.0 (2025-07-06)
# ========================================================================
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


def entrenar_y_evaluar_naive_bayes(X_train, y_train, X_val=None, y_val=None):
    clf = GaussianNB()
    clf.fit(X_train, y_train)

    acc_train = accuracy_score(y_train, clf.predict(X_train))
    print(f"\n📊 Accuracy Train: {acc_train:.3f}")

    if X_val is not None and y_val is not None:
        y_pred_val = clf.predict(X_val)
        acc_val = accuracy_score(y_val, y_pred_val)
        print(f"📊 Accuracy Validación: {acc_val:.3f}")

    return clf


def validar_naive_bayes_kfold(X, y, n_splits=5, sample_frac=None):
    if sample_frac is not None and 0 < sample_frac < 1.0:
        n = int(len(X) * sample_frac)
        idx = np.random.choice(len(X), n, replace=False)
        X = X[idx]
        y = y[idx]
        print(f"📉 Submuestreo aplicado: {n:,} muestras")

    accs = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(tqdm(skf.split(X, y), total=n_splits, desc="🔁 K-Folds")):
        X_train_cv, X_val_cv = X[train_idx], X[val_idx]
        y_train_cv, y_val_cv = y[train_idx], y[val_idx]

        clf = GaussianNB()
        clf.fit(X_train_cv, y_train_cv)
        acc = accuracy_score(y_val_cv, clf.predict(X_val_cv))
        accs.append(acc)
        print(f"Fold {fold+1}: Accuracy = {acc:.3f}")

    print(f"\n✅ Accuracy promedio: {np.mean(accs):.3f}")
    return np.mean(accs)


def evaluar_naive_bayes(modelo, X, y, clase_dict, nombre_set="Test"):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    from time import time
    from sklearn.metrics import (
        confusion_matrix, classification_report, accuracy_score,
        cohen_kappa_score, roc_curve, auc
    )

    print(f"\n🔎 Evaluando modelo sobre {nombre_set}...")
    t0 = time()

    y_pred = modelo.predict(X)
    y_prob = modelo.predict_proba(X)[:, 1] if hasattr(modelo, "predict_proba") else None

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



def clasificar_ventana_raster_nb(modelo, scaler, raster_array, clase_dict, pca=None):
    d, h, w = raster_array.shape
    X_window = raster_array.reshape([d, h * w]).T

    R, G, B = X_window[:, 0], X_window[:, 1], X_window[:, 2]
    excess_green = 2 * G - R - B
    cive = 0.441 * R - 0.811 * G + 0.385 * B + 18.78745

    X_ext = np.column_stack([R, G, B, R - G, R - B, G - B, excess_green, cive])
    X_scaled = scaler.transform(X_ext)

    if pca is not None:
        X_scaled = pca.transform(X_scaled)

    y_pred = modelo.predict(X_scaled)
    Y_pred_img = y_pred.reshape((h, w))

    class_labels = list(clase_dict.keys())
    class_colors = {
        "Permeable": "green",
        "No permeable": "lightgray"
    }
    color_list = [class_colors[clase] for clase in class_labels]
    custom_cmap = ListedColormap(color_list)

    legend_patches = [
        mpatches.Patch(color=class_colors[clase], label=clase)
        for clase in class_labels
    ]

    fig, axs = plt.subplots(1, 2, figsize=(14, 8))
    axs[0].imshow(Y_pred_img, cmap=custom_cmap)
    axs[0].set_title("Clasificación Naive Bayes")
    axs[0].axis('off')
    axs[0].legend(handles=legend_patches, loc='lower left', fontsize=10)

    rgb_img = np.transpose(raster_array[[0, 1, 2]], (1, 2, 0)).astype(np.uint8)
    axs[1].imshow(rgb_img)
    axs[1].set_title("Imagen RGB original")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

    return Y_pred_img
