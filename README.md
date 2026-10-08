# 🚲 AW-Bikes — Modelo de clasificación con Streamlit

Aplicación desarrollada en **Python + Streamlit** para el Mini Case **AW-Bikes (II): Convirtiendo datos en conocimiento**.

El proyecto utiliza información de clientes para construir un modelo de clasificación capaz de estimar si un cliente puede ser un **posible comprador de bicicletas (`BikeBuyer`)**.

## 🎯 Objetivo

Transformar los datos de clientes en información útil para la toma de decisiones comerciales mediante:

- exploración y preparación de los datos;
- selección de variables predictoras;
- entrenamiento de un modelo de clasificación;
- evaluación mediante matriz de confusión y métricas;
- análisis de la importancia de los predictores;
- predicción interactiva para un nuevo cliente.

## 📁 Estructura del proyecto

```text
aw-bikes-streamlit/
├── app.py
├── datos_actividad2.xlsx
├── requirements.txt
└── README.md
```

### Archivos principales

| Archivo | Descripción |
|---|---|
| `app.py` | Aplicación principal desarrollada con Streamlit. |
| `datos_actividad2.xlsx` | Dataset utilizado para entrenar y evaluar el modelo. |
| `requirements.txt` | Librerías necesarias para ejecutar la aplicación. |
| `README.md` | Documentación del proyecto. |

## 🧠 Variables utilizadas

El modelo utiliza las siguientes variables predictoras:

- `Age`
- `Education`
- `Occupation`
- `Gender`
- `MaritalStatus`
- `HomeOwnerFlag`
- `NumberCarsOwned`
- `NumberChildrenAtHome`
- `TotalChildren`
- `YearlyIncome`

La variable objetivo es:

- `BikeBuyer` — indica si el cliente es comprador de bicicleta (`1`) o no comprador (`0`).

Los identificadores personales o variables que no aportan valor predictivo directo se excluyen del entrenamiento para evitar que el modelo dependa de identificadores en lugar de características del cliente.

## 🤖 Modelo de clasificación

Se utiliza un **árbol de decisión (`DecisionTreeClassifier`)** porque permite:

1. trabajar con relaciones no lineales;
2. combinar variables numéricas y categóricas después del preprocesamiento;
3. interpretar la importancia de las variables;
4. explicar de forma sencilla las decisiones del modelo;
5. generar una herramienta útil para segmentación y priorización comercial.

El flujo de procesamiento incluye:

- imputación de valores faltantes;
- estandarización de variables numéricas;
- codificación One-Hot de variables categóricas;
- división estratificada entre entrenamiento y prueba;
- entrenamiento del árbol de decisión;
- evaluación sobre datos de prueba.

## 📊 Funcionalidades de la aplicación

La aplicación está organizada en cinco secciones:

### 1. Datos

Permite revisar el dataset, conocer su tamaño, visualizar la distribución de `BikeBuyer`, identificar valores faltantes y explorar información básica de los clientes.

### 2. Predictores

Presenta las variables seleccionadas para el modelo y explica por qué cada una puede aportar información para estimar la probabilidad de compra.

### 3. Modelo

Entrena el árbol de decisión y muestra las principales métricas de rendimiento.

### 4. Evaluación

Incluye:

- matriz de confusión;
- verdaderos positivos (TP);
- verdaderos negativos (TN);
- falsos positivos (FP);
- falsos negativos (FN);
- accuracy;
- precision;
- recall/sensibilidad;
- F1-score;
- importancia de los predictores.

### 5. Predicción

Permite introducir las características de un nuevo cliente y obtener una clasificación estimada sobre si podría ser comprador de bicicleta.

## ▶️ Ejecución local

### 1. Clonar el repositorio

```bash
git clone URL_DEL_REPOSITORIO
cd aw-bikes-streamlit
```

### 2. Instalar las dependencias

```bash
pip install -r requirements.txt
```

### 3. Ejecutar Streamlit

```bash
streamlit run app.py
```

La aplicación se abrirá en el navegador, normalmente en una dirección local como `http://localhost:8501`.

## ☁️ Publicación en Streamlit Community Cloud

Para desplegar la aplicación:

1. Crear o utilizar un repositorio de GitHub.
2. Subir `app.py`, `datos_actividad2.xlsx`, `requirements.txt` y `README.md`.
3. Crear una aplicación en Streamlit Community Cloud.
4. Seleccionar el repositorio de GitHub.
5. Seleccionar la rama `main`.
6. Indicar `app.py` como archivo principal.
7. Ejecutar el despliegue.

> **Importante:** `datos_actividad2.xlsx` debe estar en el mismo nivel que `app.py` si se desea que la aplicación utilice automáticamente el dataset incluido en el repositorio.

## 📈 Resultado del modelo de referencia

Con la configuración utilizada en el desarrollo de la actividad:

- división entrenamiento/prueba: **80 % / 20 %**;
- árbol de decisión con profundidad máxima de **5**;
- mínimo de **50 observaciones por hoja**;
- partición estratificada;
- `random_state = 42`.

La evaluación de referencia obtuvo aproximadamente:

| Métrica | Resultado |
|---|---:|
| Accuracy | 79,13 % |
| Precision | 77,79 % |
| Sensibilidad / Recall | 87,01 % |
| Especificidad | 69,44 % |
| F1-score | 82,14 % |

Estos valores corresponden al conjunto de prueba utilizado en el desarrollo de la actividad y pueden variar si se modifican los parámetros del modelo o los datos.

## 🎓 Relación con la actividad académica

La aplicación sirve como soporte práctico para los cinco criterios principales de la actividad:

| Criterio | Evidencia en la aplicación |
|---|---|
| 1. Justificación de predictores | Pestaña **Predictores** y selección de variables. |
| 2. Justificación del modelo | Pestaña **Modelo** y explicación del árbol de decisión. |
| 3. Matriz de confusión | Pestaña **Evaluación** con TP, TN, FP y FN. |
| 4. Predicción | Pestaña **Predicción** con un nuevo perfil de cliente. |
| 5. Utilidad del modelo | Interpretación comercial de los resultados para priorizar clientes. |

## ⚠️ Consideraciones

El modelo es una herramienta de apoyo a la decisión y no garantiza que un cliente compre una bicicleta. Sus resultados deben interpretarse junto con criterios comerciales y conocimiento del negocio.

El rendimiento puede cambiar si se utiliza otro conjunto de datos, se modifican los predictores o se ajustan los parámetros del árbol.

## 🛠️ Tecnologías

- **Python**
- **Streamlit**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**
- **OpenPyXL**

## 📚 Contexto académico

Proyecto desarrollado como parte del análisis del caso **AW-Bikes (II): Convirtiendo datos en conocimiento**, utilizando técnicas de clasificación supervisada para transformar datos de clientes en información útil para la toma de decisiones.
