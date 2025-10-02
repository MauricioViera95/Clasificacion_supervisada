# ========================================================================
# svm_model.py
# ------------------------------------------------------------------------
# Entrenamiento, evaluación y aplicación de SVM para clasificar
# suelo permeable y no permeable. Incluye validación cruzada, búsqueda
# de hiperparámetros, métricas detalladas, curva ROC y clasificación
# sobre raster georreferenciado.
#
# @author   Mauricio Viera-Torres <ronnyvieramt95@gmail.com>
# @address  La Tola (Ecuador)
# @version  1.0.0 (2025-07-06)
# ========================================================================

from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

def explorar_valores_c(X_train, y_train, X_val, y_val, c_values=None, sample_frac=None):
    if c_values is None:
        c_values = [0.1, 1, 10]  # simplificado para velocidad

    if sample_frac is not None and 0 < sample_frac < 1.0:
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=sample_frac, stratify=y_train, random_state=42)
        print(f"📉 Submuestreo aplicado: {len(X_train):,} muestras")

    scores_train, scores_val = [], []
    for C in tqdm(c_values, desc="🔍 Explorando valores de C"):
        clf = svm.SVC(C=C, kernel='rbf', probability=False, random_state=42, max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred_train = clf.predict(X_train)
        y_pred_val = clf.predict(X_val)
        scores_train.append(accuracy_score(y_train, y_pred_train))
        scores_val.append(accuracy_score(y_val, y_pred_val))

    best_idx = np.argmax(scores_val)
    best_C = c_values[best_idx]
    best_clf = svm.SVC(C=best_C, kernel='rbf', probability=True, random_state=42, max_iter=1000)
    best_clf.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))

    plt.figure()
    plt.semilogx(c_values, scores_train, label="Train")
    plt.semilogx(c_values, scores_val, label="Validation")
    plt.axvline(x=best_C, label=f'Mejor C: {best_C:.3f}', color='red')
    plt.title("Exactitud en función del parámetro C (SVM)")
    plt.xlabel("C")
    plt.ylabel("Exactitud")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return best_clf, best_C

def optimizar_svm_con_kfold(X, y, n_trials=20, n_splits=3, sample_frac=None):
    def objective(trial):
        C = trial.suggest_float("C", 1e-2, 1e2, log=True)
        gamma = trial.suggest_float("gamma", 1e-4, 1e0, log=True)

        clf = svm.SVC(C=C, gamma=gamma, kernel='rbf', probability=False, random_state=42, max_iter=1000)

        if sample_frac is not None and 0 < sample_frac < 1.0:
            n = int(len(X) * sample_frac)
            idx = np.random.choice(len(X), n, replace=False)
            X_, y_ = X[idx], y[idx]
        else:
            X_, y_ = X, y

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in tqdm(skf.split(X_, y_), total=n_splits, desc="🔁 Validación cruzada"):
            X_train_cv, X_val_cv = X_[train_idx], X_[val_idx]
            y_train_cv, y_val_cv = y_[train_idx], y_[val_idx]

            clf.fit(X_train_cv, y_train_cv)
            preds = clf.predict(X_val_cv)
            acc = accuracy_score(y_val_cv, preds)
            scores.append(acc)

        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("\n🔍 Mejores hiperparámetros encontrados:")
    print(study.best_params)
    return study.best_params

def entrenar_svm_final(X_train, y_train, best_params, sample_frac=None, verbose=True):
    """
    Entrena un modelo SVM con control de avance y submuestreo opcional.

    Parámetros:
        X_train (ndarray): Datos de entrenamiento.
        y_train (ndarray): Etiquetas de entrenamiento.
        best_params (dict): Hiperparámetros optimizados para SVM.
        sample_frac (float): Fracción de muestras a utilizar (0 < sample_frac <= 1). None para usar todos.
        verbose (bool): Si True, muestra progreso.

    Retorna:
        clf (svm.SVC): Modelo entrenado.
    """
    if sample_frac is not None and 0 < sample_frac < 1.0:
        n = int(len(X_train) * sample_frac)
        idx = np.random.choice(len(X_train), n, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
        print(f"📉 Submuestreo aplicado: {n:,} muestras")

    clf = svm.SVC(**best_params, probability=True, random_state=42)

    if verbose:
        print("🚀 Entrenando modelo SVM...")
        with tqdm(total=1, desc="🔧 Entrenamiento SVM", unit="modelo") as pbar:
            clf.fit(X_train, y_train)
            pbar.update(1)
    else:
        clf.fit(X_train, y_train)

    return clf

def evaluar_svm(modelo, X, y, clase_dict, nombre_set="Test", sample_frac=None, batch_size=10000):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    from time import time
    from sklearn.metrics import (
        confusion_matrix, classification_report, accuracy_score,
        cohen_kappa_score, roc_curve, auc
    )
    import numpy as np
    from tqdm import tqdm

    if sample_frac is not None and 0 < sample_frac < 1.0:
        n = int(len(X) * sample_frac)
        idx = np.random.choice(len(X), n, replace=False)
        X = X[idx]
        y = y[idx]
        print(f"📉 Evaluando sobre una fracción de los datos: {n:,} muestras")

    print(f"\n🔎 Evaluando modelo sobre {nombre_set}...")
    t0 = time()

    # Evaluación por lotes
    y_pred = []
    y_prob = []
    for i in tqdm(range(0, len(X), batch_size), desc="⏳ Evaluando"):
        X_batch = X[i:i + batch_size]
        y_pred.extend(modelo.predict(X_batch))
        if hasattr(modelo, "predict_proba"):
            y_prob.extend(modelo.predict_proba(X_batch)[:, 1])

    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob) if y_prob else None

    t1 = time()
    print(f"✅ Evaluación completada en {t1 - t0:.2f} segundos.")

    print(f"\n📋 Reporte de clasificación - {nombre_set}:")
    print(classification_report(y, y_pred, target_names=list(clase_dict.keys())))

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensibilidad = tp / (tp + fn)
    especificidad = tn / (tn + fp)

    def format_miles(x):
        return f'{x:,}'.replace(',', ' ')

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

def clasificar_ventana_raster_svm(modelo, scaler, raster_array, clase_dict, pca=None, batch_size=100000):
    d, h, w = raster_array.shape
    X_window = raster_array.reshape([d, h * w]).T

    # Variables espectrales
    R, G, B = X_window[:, 0], X_window[:, 1], X_window[:, 2]
    excess_green = 2 * G - R - B
    cive = 0.441 * R - 0.811 * G + 0.385 * B + 18.78745
    X_ext = np.column_stack([R, G, B, R - G, R - B, G - B, excess_green, cive])

    # Escalado
    X_scaled = scaler.transform(X_ext)
    if pca is not None:
        X_scaled = pca.transform(X_scaled)

    # Predicción por lotes con barra de progreso
    print("⏳ Clasificando píxeles...")
    y_pred = []
    for i in tqdm(range(0, len(X_scaled), batch_size), desc="🔍 Procesando"):
        X_batch = X_scaled[i:i + batch_size]
        y_pred.extend(modelo.predict(X_batch))

    y_pred = np.array(y_pred)
    Y_pred_img = y_pred.reshape((h, w))

    # Visualización
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
    axs[0].set_title("Clasificación SVM")
    axs[0].axis('off')
    axs[0].legend(handles=legend_patches, loc='lower left', fontsize=10)

    rgb_img = np.transpose(raster_array[[0, 1, 2]], (1, 2, 0)).astype(np.uint8)
    axs[1].imshow(rgb_img)
    axs[1].set_title("Imagen RGB original")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

    return Y_pred_img
