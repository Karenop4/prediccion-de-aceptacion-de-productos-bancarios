# 🏦 Predicción de Aceptación de Productos Bancarios con Deep Learning y XAI

> **Un proceso sistemático mediante Redes Neuronales, Modelos de Ensamble y Técnicas de Explicabilidad.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Scikit-Learn](https://img.shields.io/badge/Sklearn-Machine%20Learning-yellow)
![SHAP](https://img.shields.io/badge/XAI-SHAP-green)

## 📋 Descripción del Proyecto

Este proyecto aborda el desafío de predecir si un cliente suscribirá un depósito a plazo fijo basándose en datos de campañas de marketing directo de una institución bancaria portuguesa.

Se sigue la metodología **CRISP-DM**, implementando un flujo de trabajo robusto que incluye limpieza de datos, ingeniería de características, reducción de dimensionalidad (PCA), balanceo de clases (SMOTE) y la comparación de arquitecturas de Machine Learning clásico vs. Deep Learning y Stacking.

**Autores:**
* Andrés Encalada
* Karen Ortiz

## 📊 Dataset

El conjunto de datos proviene del repositorio UCI Machine Learning: **Bank Marketing Dataset**.
* **Instancias:** 41,188
* **Variables:** 20 (Demográficas, Contexto Socioeconómico y Detalles de contacto).
* **Target:** `y` (Suscripción al depósito: 'yes'/'no').

## 🛠️ Tecnologías y Librerías

* **Python 3**
* **Manipulación de Datos:** Pandas, NumPy.
* **Machine Learning:** Scikit-learn (Pipeline, ColumnTransformer, PCA).
* **Balanceo de Datos:** Imbalanced-learn (SMOTE).
* **Deep Learning:** TensorFlow / Keras.
* **Explicabilidad (XAI):** SHAP.
* **Visualización:** Matplotlib, Seaborn.

## ⚙️ Metodología del Pipeline

### 1. Preprocesamiento e Ingeniería de Características
* **Limpieza:** Manejo de valores 'unknown' y eliminación de duplicados.
* **Transformación:**
    * *Numéricas:* Imputación (mediana) y Estandarización (StandardScaler).
    * *Categóricas:* Codificación One-Hot y Ordinal.
* **Reducción de Dimensionalidad:** Aplicación de **PCA** conservando el 95% de la varianza (reducción de 47 a 22 features).
* **Balanceo:** Aplicación de **SMOTE** en el conjunto de entrenamiento para mitigar el desbalance de clases.

### 2. Modelado (Arquitecturas Evaluadas)
Se diseñaron y compararon tres estrategias:
1.  **Línea Base (Baseline):** Comparación entre **KNN** y **Random Forest**. (Ganador: Random Forest).
2.  **Deep Learning (Propuesta):** Red Neuronal Artificial (ANN) con capas densas, activación ReLU y Dropout para regularización.
3.  **Híbrida (Stacking):** Un ensamble que combina las predicciones de RF, KNN y ANN utilizando una Regresión Logística como meta-modelo.

## 🏆 Resultados y Evaluación

El modelo de **Red Neuronal** fue seleccionado como el mejor modelo para producción debido a su capacidad superior de generalización (AUC) y su equilibrio en la matriz de confusión (menor cantidad de clientes potenciales perdidos).

| Modelo | AUC Score | F1-Score | Observación |
| :--- | :---: | :---: | :--- |
| **Red Neuronal (ANN)** | **0.9423** | **0.6305** | **Mejor rendimiento global y operativo.** |
| Random Forest | 0.9357 | 0.6143 | Baseline sólido. |
| Stacking Ensemble | 0.9342 | 0.5756 | Demasiado conservador (muchos Falsos Negativos). |

### Impacto de Negocio (Matriz de Confusión)
La Red Neuronal logró reducir significativamente los **Falsos Negativos** (clientes interesados que el modelo ignoró), captando un **41% más de oportunidades de venta** en comparación con el modelo de Stacking.

## 🔍 Explicabilidad (XAI) con SHAP

Se utilizó **SHAP (SHapley Additive exPlanations)** para interpretar las decisiones del modelo.
* **Variable más influyente:** `duration` (Duración de la llamada). A mayor duración, mayor probabilidad de éxito.
* **Factores Macroeconómicos:** `euribor3m` y `nr.employed` juegan un papel crucial, indicando que el contexto económico afecta la decisión del cliente.

## 🚀 Instalación y Uso

1.  Clona el repositorio:
    ```bash
    git clone [https://github.com/Karenop4/prediccion-de-aceptacion-de-productos-bancarios.git](https://github.com/Karenop4/prediccion-de-aceptacion-de-productos-bancarios.git)
    ```
2.  Instala las dependencias:
    ```bash
    pip install pandas numpy scikit-learn tensorflow shap imbalanced-learn matplotlib seaborn
    ```
3.  Ejecuta el notebook `Interciclo.ipynb` en Jupyter o Google Colab.

## 📄 Licencia
Este proyecto es para fines educativos y académicos.
