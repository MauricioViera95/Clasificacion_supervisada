# Clasificacion_supervisada
Este repositorio contiene algoritmos de clasificación supervisada aplicados a la distinción de suelo permeable  a partir de ortofotos y datos vectoriales. El flujo de trabajo incluye generación de muestras, extracción de características, PCA, entrenamiento, validación cruzada, optimización de hiperparámetros y aplicación de los modelos.

📂 Datos
El proyecto utiliza una estructura estándar de directorios para organizar insumos y datos procesados:

INSUMOS/   # Archivos de entrada (ortofotos, vectores, etc.)

DATA/      # Datos procesados y salidas de modelos

🔗 Opción 1: Datos ya organizados en Google Drive

Puedes acceder directamente a la estructura completa (INSUMOS y DATA) en el siguiente enlace:

👉 https://drive.google.com/drive/folders/1bAoQoRNwwrQ80NHqpTCJstQLO003U4NS


🔗 Opción 2: Descarga desde fuentes oficiales

Si prefieres armar la estructura desde cero:

* Ortofotos PNOA → disponibles en la web oficial del Instituto Geográfico Nacional:

      https://pnoa.ign.es/pnoa-imagen/productos-a-descarga

* Datos de cobertura del suelo SIOSE → descargables desde el Centro de Descargas del CNIG:

      https://centrodedescargas.cnig.es/CentroDescargas/siose
  

📂 Estructura de archivos
🔧 Configuración y utilidades

* config.py → Define rutas de entrada (ortofotos y shapefiles) y salida

* data_loader.py → Funciones para cargar ortofotos y shapefiles, preparar conjuntos de entrenamiento/validación/test.

* feature_extraction.py → Genera atributos espectrales (bandas, índices como ExG, CIVE, etc.) para cada píxel/muestra.

* pca_utils.py → Funciones para aplicar y guardar PCA sobre las variables predictoras.

* generar_muestras.py → Script para generar y balancear muestras a partir de polígonos de entrenamiento, validación y test.

🏗️ Modelos de clasificación

Cada archivo entrena, valida y aplica un algoritmo específico:

* knn_optim.py → K-Nearest Neighbors con búsqueda de hiperparámetros y evaluación (matriz de confusión, ROC).

* decision_tree_optim.py → Árbol de Decisión con selección de profundidad óptima y visualización.

* random_forest_optim.py → Random Forest con Optuna para búsqueda bayesiana, métricas avanzadas y aplicación a raster

* logistic_regression.py → Regresión Logística con métricas detalladas e importancia de atributos.

* naive_bayes.py → Clasificación Naive Bayes con validación cruzada y visualización de predicción sobre raster

* svm_model.py → Support Vector Machine (SVM) con validación cruzada, búsqueda de hiperparámetros (Optuna) y clasificación raster

* gradient_boosting.py → Gradient Boosting con evaluación y curvas de aprendizaje.

* xgboost_model.py → XGBoost con optimización bayesiana (Optuna), métricas detalladas e importancia de variables

* lightgbm_model.py → LightGBM con entrenamiento, validación y visualización de métricas.

🚀 Ejecución central

* main.py → Punto de entrada principal. Coordina:

  * Carga de datos (data_loader.py).
  
  * Extracción de características (feature_extraction.py).
  
  * Reducción de dimensionalidad (pca_utils.py).
  
  * Entrenamiento y validación de los distintos modelos.
  
  * Predicción sobre ventanas raster.

📦 Dependencias

El proyecto requiere Python 3.10+ y se recomienda usar un entorno virtual (Anaconda/venv).
Librerías principales

* Geoespaciales: rasterio, geopandas, shapely

* Machine Learning: scikit-learn, xgboost, lightgbm, optuna

* Ciencia de datos: numpy, pandas, matplotlib, seaborn, tqdm

▶️ Ejecución

1. Definir rutas en config.py.

2. Generar muestras balanceadas con:
    python generar_muestras.py
3. Entrenar y evaluar un modelo (ejemplo Random Forest):
    python random_forest_optim.py
4. Ejecutar el flujo central:
    python main.py

📊 Salidas

* Reportes de métricas (Accuracy, Kappa, Sensibilidad, Especificidad).

* Gráficos: matrices de confusión, curvas ROC, curvas de validación y aprendizaje.

* Clasificación raster exportada como GeoTIFF.
