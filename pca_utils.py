# ========================================================================
# pca_utils.py
# ------------------------------------------------------------------------
# Utilidades para aplicar PCA sobre conjuntos de datos geoespaciales,
# manteniendo compatibilidad con flujos existentes de entrenamiento.
# Incluye visualización de varianza explicada y transformaciones.
#
# @author   Mauricio Viera-Torres <ronnyvieramt95@gmail.com>
# @address  La Tola (Ecuador)
# @version  1.0.0 (2025-07-10)
# ========================================================================

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

def aplicar_pca(X_train, X_val, X_test, X_kfold, y_train=None,
                n_components=0.95,
                plot_varianza=True,
                plot_2d=False,
                plot_3d=False,
                save_image=False,
                output_dir="outputs"):
    """
    Aplica PCA a los datos y permite controlar visualizaciones.

    Parámetros:
    - n_components: porcentaje de varianza explicada o número fijo de componentes
    - plot_varianza: bool, muestra el gráfico de varianza explicada
    - plot_2d: bool, muestra visualización PCA 2D
    - plot_3d: bool, muestra visualización PCA 3D
    - save_image: bool, guarda la imagen 2D si plot_2d = True
    """

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca   = pca.transform(X_val)
    X_test_pca  = pca.transform(X_test)
    X_kfold_pca = pca.transform(X_kfold)

    # ----- Gráfico varianza explicada acumulada -----
    if plot_varianza:
        var_cumsum = np.cumsum(pca.explained_variance_ratio_) * 100
        plt.figure()
        plt.plot(var_cumsum, marker='o')
        plt.axhline(y=95, color='gray', linestyle='--')
        plt.xlabel("Número de componentes")
        plt.ylabel("Varianza acumulada (%)")
        plt.title("PCA - Varianza explicada acumulada")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # ----- Visualización 2D -----
    if plot_2d and X_train_pca.shape[1] >= 2:
        plt.figure()
        if y_train is not None:
            sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=y_train, palette='Set1', s=30)
        else:
            plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], alpha=0.5)
        plt.xlabel("Componente principal 1")
        plt.ylabel("Componente principal 2")
        plt.title("PCA - Visualización 2D")
        plt.grid(True)
        plt.tight_layout()

        if save_image:
            os.makedirs(output_dir, exist_ok=True)
            image_path = os.path.join(output_dir, "pca_visualizacion_2d.png")
            print("💾 Guardando imagen 2D...", end="", flush=True)
            for i in range(10):
                time.sleep(0.1)
                print(".", end="", flush=True)
            plt.savefig(image_path, dpi=300)
            print(" ✅")
            print(f"📸 Imagen 2D guardada en: {image_path}")

        plt.show()

    # ----- Visualización 3D -----
    if plot_3d and X_train_pca.shape[1] >= 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if y_train is not None:
            scatter = ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], X_train_pca[:, 2],
                                 c=y_train, cmap='Set1', s=15)
            legend1 = ax.legend(*scatter.legend_elements(), title="Clases")
            ax.add_artist(legend1)
        else:
            ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], X_train_pca[:, 2], alpha=0.6)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title("PCA - Visualización 3D")
        plt.tight_layout()
        plt.show()

    print(f"✅ PCA aplicado: {X_train.shape[1]} → {X_train_pca.shape[1]} componentes principales")
    return X_train_pca, X_val_pca, X_test_pca, X_kfold_pca, pca
