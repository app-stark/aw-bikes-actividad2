from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="AW-Bikes | Clasificación de clientes",
    page_icon="🚲",
    layout="wide",
)

st.title("🚲 AW-Bikes — Convirtiendo datos en conocimiento")
st.markdown(
    """
    **Modelo de clasificación de clientes mediante Python, Scikit-learn y Streamlit.**

    Esta aplicación permite explorar los datos de clientes de AW-Bikes,
    seleccionar variables predictoras, entrenar un árbol de decisión,
    evaluar su desempeño y realizar predicciones sobre nuevos clientes.
    """
)


# ============================================================
# VARIABLES DEL MODELO
# ============================================================

TARGET = "BikeBuyer"

CATEGORICAL_FEATURES = [
    "Education",
    "Occupation",
    "Gender",
    "MaritalStatus",
]

NUMERIC_FEATURES = [
    "Age",
    "HomeOwnerFlag",
    "NumberCarsOwned",
    "NumberChildrenAtHome",
    "TotalChildren",
    "YearlyIncome",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

DEFAULT_FILE = "datos_actividad2.xlsx"


# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================

@st.cache_data
def load_data(file_source):

    df = pd.read_excel(
        file_source,
        sheet_name="Hoja2"
    )

    # --------------------------------------------------------
    # Conversión de fechas
    # --------------------------------------------------------

    if "BirthDate" in df.columns:
        df["BirthDate"] = pd.to_datetime(
            df["BirthDate"],
            errors="coerce"
        )

    if "LastUpdated" in df.columns:
        df["LastUpdated"] = pd.to_datetime(
            df["LastUpdated"],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Cálculo de edad
    # --------------------------------------------------------

    if (
        "BirthDate" in df.columns
        and "LastUpdated" in df.columns
    ):
        df["Age"] = (
            (
                df["LastUpdated"]
                - df["BirthDate"]
            ).dt.days / 365.25
        ).round(1)

    # --------------------------------------------------------
    # Conversión de variable objetivo
    # --------------------------------------------------------

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    return df


# ============================================================
# CONSTRUCCIÓN DEL MODELO
# ============================================================

def build_model(
    max_depth=5,
    min_samples_leaf=50
):

    # Variables numéricas
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    # Variables categóricas
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            ),
        ]
    )

    # Preprocesamiento
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                NUMERIC_FEATURES
            ),
            (
                "cat",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            ),
        ]
    )

    # Árbol de decisión
    classifier = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    # Pipeline completo
    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                classifier
            ),
        ]
    )

    return model


# ============================================================
# ENTRENAMIENTO
# ============================================================

def train_model(
    df,
    test_size,
    max_depth,
    min_samples_leaf
):

    data = df[
        FEATURES + [TARGET]
    ].copy()

    # Se eliminan solamente los registros sin objetivo
    data = data.dropna(
        subset=[TARGET]
    )

    X = data[FEATURES]

    y = data[TARGET].astype(int)

    # División entrenamiento/prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    # Crear modelo
    model = build_model(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf
    )

    # Entrenar
    model.fit(
        X_train,
        y_train
    )

    # Predicciones
    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # Matriz de confusión
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    # Métricas
    metrics = {

        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "Sensibilidad": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "Especificidad": (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0
        ),

        "F1": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),
    }

    return (
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        predictions,
        probabilities,
        cm,
        metrics,
    )


# ============================================================
# BARRA LATERAL
# ============================================================

st.sidebar.header("⚙️ Configuración del modelo")

uploaded_file = st.sidebar.file_uploader(
    "Cargar otro archivo Excel",
    type=["xlsx"],
    help=(
        "Opcional. Si no seleccionas un archivo, "
        "se utilizará datos_actividad2.xlsx."
    )
)

st.sidebar.markdown("---")

test_size = st.sidebar.slider(
    "Porcentaje de datos para prueba",
    min_value=0.10,
    max_value=0.40,
    value=0.20,
    step=0.05
)

max_depth = st.sidebar.slider(
    "Profundidad máxima del árbol",
    min_value=2,
    max_value=10,
    value=5,
    step=1
)

min_samples_leaf = st.sidebar.slider(
    "Mínimo de observaciones por hoja",
    min_value=10,
    max_value=200,
    value=50,
    step=10
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Configuración utilizada en el documento**

    • Prueba: 20 %

    • Profundidad máxima: 5

    • Mínimo por hoja: 50

    • Random state: 42
    """
)


# ============================================================
# CARGAR ARCHIVO
# ============================================================

if uploaded_file is None:

    if not Path(DEFAULT_FILE).exists():

        st.error(
            f"No se encontró el archivo **{DEFAULT_FILE}**."
        )

        st.info(
            """
            Coloca `datos_actividad2.xlsx` en la misma carpeta
            donde se encuentra `app.py`.
            """
        )

        st.stop()

    data_source = DEFAULT_FILE

else:

    data_source = uploaded_file


# ============================================================
# LEER DATOS
# ============================================================

try:

    df = load_data(
        data_source
    )

except Exception as error:

    st.error(
        f"No fue posible cargar el archivo: {error}"
    )

    st.stop()


# ============================================================
# VALIDAR COLUMNAS
# ============================================================

required_columns = FEATURES + [TARGET]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error(
        "El archivo no contiene las siguientes columnas requeridas:"
    )

    st.write(
        missing_columns
    )

    st.stop()


# ============================================================
# ENTRENAR MODELO
# ============================================================

try:

    (
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        predictions,
        probabilities,
        confusion,
        metrics,
    ) = train_model(
        df,
        test_size=test_size,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf
    )

except Exception as error:

    st.error(
        f"No fue posible entrenar el modelo: {error}"
    )

    st.stop()


# ============================================================
# INDICADORES PRINCIPALES
# ============================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Clientes",
    f"{len(df):,}"
)

col2.metric(
    "Accuracy",
    f"{metrics['Accuracy']:.2%}"
)

col3.metric(
    "Sensibilidad",
    f"{metrics['Sensibilidad']:.2%}"
)

col4.metric(
    "F1",
    f"{metrics['F1']:.2%}"
)


# ============================================================
# PESTAÑAS
# ============================================================

tab_data, tab_predictors, tab_model, tab_evaluation, tab_prediction = st.tabs(
    [
        "📊 Datos",
        "🎯 Predictores",
        "🌳 Modelo",
        "🔎 Evaluación",
        "🔮 Predicción",
    ]
)


# ============================================================
# PESTAÑA 1 — DATOS
# ============================================================

with tab_data:

    st.header(
        "1. Exploración de los datos"
    )

    col_left, col_right = st.columns(2)

    # --------------------------------------------------------
    # Muestra
    # --------------------------------------------------------

    with col_left:

        st.subheader(
            "Muestra de clientes"
        )

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

    # --------------------------------------------------------
    # Distribución BikeBuyer
    # --------------------------------------------------------

    with col_right:

        st.subheader(
            "Distribución de BikeBuyer"
        )

        counts = (
            df[TARGET]
            .value_counts()
            .reindex(
                [0, 1],
                fill_value=0
            )
        )

        fig, ax = plt.subplots(
            figsize=(6, 4)
        )

        ax.bar(
            [
                "No comprador (0)",
                "Comprador (1)"
            ],
            counts.values
        )

        ax.set_ylabel(
            "Número de clientes"
        )

        ax.set_title(
            "Distribución de la variable objetivo"
        )

        fig.tight_layout()

        st.pyplot(
            fig
        )

        plt.close(fig)

    # --------------------------------------------------------
    # Valores faltantes
    # --------------------------------------------------------

    st.subheader(
        "Valores faltantes"
    )

    missing_table = (
        df[
            FEATURES + [TARGET]
        ]
        .isna()
        .sum()
        .reset_index()
    )

    missing_table.columns = [
        "Variable",
        "Valores faltantes"
    ]

    st.dataframe(
        missing_table,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        """
        Los identificadores del cliente y otros datos personales no se
        utilizan como variables predictoras. El objetivo es construir un
        modelo que aprenda patrones de comportamiento, no identificar
        directamente a una persona.
        """
    )


# ============================================================
# PESTAÑA 2 — PREDICTORES
# ============================================================

with tab_predictors:

    st.header(
        "2. Selección de variables predictoras"
    )

    predictor_justification = {

        "Age":
            "Representa la etapa de vida y permite detectar diferentes patrones de consumo.",

        "Education":
            "Aporta información sobre características socioeconómicas del cliente.",

        "Occupation":
            "Permite aproximar el contexto laboral y económico.",

        "Gender":
            "Permite identificar posibles patrones históricos de comportamiento.",

        "MaritalStatus":
            "Aporta información relacionada con el contexto familiar.",

        "HomeOwnerFlag":
            "Puede representar características de la situación patrimonial.",

        "NumberCarsOwned":
            "Puede estar relacionado con movilidad y capacidad de gasto.",

        "NumberChildrenAtHome":
            "Ayuda a representar las necesidades y características del hogar.",

        "TotalChildren":
            "Complementa la caracterización de la estructura familiar.",

        "YearlyIncome":
            "Es un indicador relevante de capacidad económica."
    }

    predictor_table = pd.DataFrame({

        "Variable":
            FEATURES,

        "Tipo":
            (
                ["Categórica"] * len(CATEGORICAL_FEATURES)
                +
                ["Numérica"] * len(NUMERIC_FEATURES)
            ),

        "Justificación":
            [
                predictor_justification[
                    variable
                ]
                for variable in FEATURES
            ]
    })

    st.dataframe(
        predictor_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PESTAÑA 3 — MODELO
# ============================================================

with tab_model:

    st.header(
        "3. Modelo de clasificación"
    )

    st.markdown(
        """
        ### Árbol de Decisión

        Se selecciona un **Decision Tree** porque permite clasificar a los
        clientes como posibles compradores o no compradores y, al mismo
        tiempo, facilita la interpretación de las reglas utilizadas para
        tomar las decisiones.

        El procesamiento incluye:

        - Imputación de valores faltantes.
        - Estandarización de variables numéricas.
        - Codificación One-Hot de variables categóricas.
        - Entrenamiento mediante un árbol de decisión.
        """
    )

    model_object = (
        model
        .named_steps["model"]
    )

    feature_names = (
        model
        .named_steps["preprocessor"]
        .get_feature_names_out()
    )

    importances = (
        model_object
        .feature_importances_
    )

    importance_table = pd.DataFrame({

        "Predictor":
            feature_names,

        "Importancia":
            importances
    })

    importance_table = (
        importance_table
        .sort_values(
            "Importancia",
            ascending=False
        )
        .head(15)
    )

    st.subheader(
        "Importancia de los predictores"
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.barh(
        importance_table["Predictor"][::-1],
        importance_table["Importancia"][::-1]
    )

    ax.set_xlabel(
        "Importancia"
    )

    ax.set_title(
        "Principales predictores utilizados por el modelo"
    )

    fig.tight_layout()

    st.pyplot(
        fig
    )

    plt.close(fig)

    st.dataframe(
        importance_table.reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PESTAÑA 4 — EVALUACIÓN
# ============================================================

with tab_evaluation:

    st.header(
        "4. Matriz de confusión y evaluación"
    )

    col_matrix, col_metrics = st.columns(2)

    # --------------------------------------------------------
    # Matriz de confusión
    # --------------------------------------------------------

    with col_matrix:

        st.subheader(
            "Matriz de confusión"
        )

        fig, ax = plt.subplots(
            figsize=(6, 5)
        )

        image = ax.imshow(
            confusion
        )

        ax.set_xticks(
            [0, 1]
        )

        ax.set_yticks(
            [0, 1]
        )

        ax.set_xticklabels(
            [
                "Predice 0",
                "Predice 1"
            ]
        )

        ax.set_yticklabels(
            [
                "Real 0",
                "Real 1"
            ]
        )

        ax.set_xlabel(
            "Predicción"
        )

        ax.set_ylabel(
            "Valor real"
        )

        ax.set_title(
            "Matriz de confusión"
        )

        for i in range(2):

            for j in range(2):

                ax.text(
                    j,
                    i,
                    str(
                        confusion[i, j]
                    ),
                    ha="center",
                    va="center"
                )

        fig.colorbar(
            image,
            ax=ax
        )

        fig.tight_layout()

        st.pyplot(
            fig
        )

        plt.close(fig)

    # --------------------------------------------------------
    # Métricas
    # --------------------------------------------------------

    with col_metrics:

        st.subheader(
            "Indicadores del modelo"
        )

        metrics_table = pd.DataFrame({

            "Métrica":
                list(
                    metrics.keys()
                ),

            "Resultado":
                [
                    f"{value:.2%}"
                    for value
                    in metrics.values()
                ]
        })

        st.dataframe(
            metrics_table,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # Interpretación
    # --------------------------------------------------------

    tn, fp, fn, tp = confusion.ravel()

    st.subheader(
        "Interpretación de la matriz"
    )

    confusion_table = pd.DataFrame({

        "Resultado": [

            "Verdadero negativo (TN)",

            "Falso positivo (FP)",

            "Falso negativo (FN)",

            "Verdadero positivo (TP)"
        ],

        "Cantidad": [
            tn,
            fp,
            fn,
            tp
        ],

        "Interpretación": [

            "Clientes que no compran y fueron clasificados correctamente como no compradores.",

            "Clientes que no compran pero fueron clasificados como compradores.",

            "Clientes que compran pero fueron clasificados como no compradores.",

            "Clientes que compran y fueron identificados correctamente."
        ]
    })

    st.dataframe(
        confusion_table,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        f"""
        **Hallazgo:** el modelo alcanza una sensibilidad del
        **{metrics['Sensibilidad']:.2%}**, por lo que identifica una proporción
        elevada de los compradores reales.

        Para una estrategia comercial, los **falsos negativos ({fn:,})** son
        especialmente importantes porque representan clientes que podrían
        tener interés en comprar una bicicleta pero que el modelo no detectó.
        """
    )


# ============================================================
# PESTAÑA 5 — PREDICCIÓN
# ============================================================

with tab_prediction:

    st.header(
        "5. Predicción de un nuevo cliente"
    )

    st.markdown(
        """
        Introduce las características de un nuevo cliente. El modelo
        devolverá una clasificación y la probabilidad estimada de pertenecer
        a la categoría de posible comprador.
        """
    )

    # --------------------------------------------------------
    # Opciones categóricas
    # --------------------------------------------------------

    options = {}

    for column in CATEGORICAL_FEATURES:

        options[column] = sorted(
            df[column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    # --------------------------------------------------------
    # Formulario
    # --------------------------------------------------------

    with st.form(
        "prediction_form"
    ):

        col_left, col_right = st.columns(2)

        # ----------------------------------------------------
        # Columna izquierda
        # ----------------------------------------------------

        with col_left:

            age = st.number_input(
                "Edad",
                min_value=15.0,
                max_value=100.0,
                value=35.0,
                step=1.0
            )

            education = st.selectbox(
                "Educación",
                options["Education"],
                index=(
                    options["Education"]
                    .index("Bachelors")
                    if "Bachelors"
                    in options["Education"]
                    else 0
                )
            )

            occupation = st.selectbox(
                "Ocupación",
                options["Occupation"],
                index=(
                    options["Occupation"]
                    .index("Skilled Manual")
                    if "Skilled Manual"
                    in options["Occupation"]
                    else 0
                )
            )

            gender = st.selectbox(
                "Género",
                options["Gender"]
            )

            marital_status = st.selectbox(
                "Estado civil",
                options["MaritalStatus"]
            )

        # ----------------------------------------------------
        # Columna derecha
        # ----------------------------------------------------

        with col_right:

            homeowner = st.selectbox(
                "Propietario de vivienda",
                [0, 1],
                index=1
            )

            cars = st.number_input(
                "Número de vehículos",
                min_value=0,
                max_value=10,
                value=2,
                step=1
            )

            children_home = st.number_input(
                "Número de hijos en casa",
                min_value=0,
                max_value=10,
                value=1,
                step=1
            )

            total_children = st.number_input(
                "Total de hijos",
                min_value=0,
                max_value=10,
                value=2,
                step=1
            )

            income = st.number_input(
                "Ingreso anual",
                min_value=0.0,
                max_value=500000.0,
                value=65000.0,
                step=1000.0
            )

        submit = st.form_submit_button(
            "🔮 Calcular predicción"
        )

    # --------------------------------------------------------
    # Realizar predicción
    # --------------------------------------------------------

    if submit:

        new_customer = pd.DataFrame([{

            "Age":
                age,

            "Education":
                education,

            "Occupation":
                occupation,

            "Gender":
                gender,

            "MaritalStatus":
                marital_status,

            "HomeOwnerFlag":
                homeowner,

            "NumberCarsOwned":
                cars,

            "NumberChildrenAtHome":
                children_home,

            "TotalChildren":
                total_children,

            "YearlyIncome":
                income
        }])

        prediction = int(
            model.predict(
                new_customer
            )[0]
        )

        probability = float(
            model.predict_proba(
                new_customer
            )[0, 1]
        )

        # ----------------------------------------------------
        # Resultado
        # ----------------------------------------------------

        if prediction == 1:

            st.success(
                f"""
                ## 🚲 Posible comprador

                El modelo clasifica al cliente como **posible comprador**.

                **Probabilidad estimada: {probability:.2%}**
                """
            )

        else:

            st.warning(
                f"""
                ## Cliente no clasificado como comprador

                El modelo no identifica al cliente como comprador potencial.

                **Probabilidad estimada de compra: {probability:.2%}**
                """
            )

        # ----------------------------------------------------
        # Perfil evaluado
        # ----------------------------------------------------

        st.subheader(
            "Perfil evaluado"
        )

        st.dataframe(
            new_customer,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# PIE DE PÁGINA
# ============================================================

st.markdown(
    "---"
)

st.caption(
    """
    AW-Bikes — Actividad académica de clasificación de clientes.
    Modelo reproducible con random_state=42.
    La predicción representa un patrón estadístico basado en los datos
    disponibles y no constituye una garantía de compra real.
    """
)