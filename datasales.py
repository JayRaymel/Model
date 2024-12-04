import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Función para cargar y previsualizar los datos
def load_data(uploaded_file):
    try:
        # Cargar el archivo CSV
        data = pd.read_csv(uploaded_file)
        st.success("Archivo cargado exitosamente!")
        st.write("### Vista previa de los datos:")
        st.write(data.head())
        return data
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None

# Función para entrenar y evaluar los modelos
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    # Regresión Lineal
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_lr = linear_model.predict(X_test)

    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    r2_lr = r2_score(y_test, y_pred_lr)

    # Árbol de Decisión
    tree_model = DecisionTreeRegressor(random_state=42)
    tree_model.fit(X_train, y_train)
    y_pred_tree = tree_model.predict(X_test)

    mae_tree = mean_absolute_error(y_test, y_pred_tree)
    rmse_tree = np.sqrt(mean_squared_error(y_test, y_pred_tree))
    r2_tree = r2_score(y_test, y_pred_tree)

    return {
        "linear": {"mae": mae_lr, "rmse": rmse_lr, "r2": r2_lr},
        "tree": {"mae": mae_tree, "rmse": rmse_tree, "r2": r2_tree}
    }

# Interfaz de Streamlit
st.title("📊 Predicción de Ventas con Modelos de Machine Learning")

# Subir archivo CSV
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])
if uploaded_file is not None:
    # Cargar datos
    data = load_data(uploaded_file)

    if data is not None:
        # Verificar si hay valores nulos
        st.write("### Comprobación de valores nulos:")
        st.write(data.isnull().sum())

        # Si hay valores nulos, podemos eliminar filas con NaN o imputar
        if data.isnull().sum().sum() > 0:
            st.write("### Imputando valores nulos con la media...")
            data.fillna(data.mean(), inplace=True)
            st.write("Valores nulos imputados exitosamente.")

        # Verificar si las columnas son numéricas
        st.write("### Verificación de los tipos de datos:")
        st.write(data.dtypes)

        # Seleccionar columna objetivo
        target_column = st.selectbox("Selecciona la columna objetivo (ventas):", data.columns)

        if target_column:
            # Separar las características y la columna objetivo
            X = data.drop(target_column, axis=1)
            y = data[target_column]

            # Verificar si las columnas son numéricas
            if not np.issubdtype(y.dtype, np.number):
                st.error("La columna objetivo debe contener valores numéricos.")
            else:
                # Verificar que las características sean numéricas
                if not np.all([np.issubdtype(X[col].dtype, np.number) for col in X.columns]):
                    st.error("Todas las columnas de características deben ser numéricas.")
                else:
                    # Normalizar las características si es necesario
                    st.write("### Normalizando los datos...")
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    # Dividir datos en entrenamiento y prueba
                    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

                    # Entrenar y evaluar modelos
                    st.write("Entrenando modelos, por favor espera...")
                    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

                    # Mostrar resultados
                    st.write("## Resultados de Regresión Lineal")
                    st.write(f"MAE: {results['linear']['mae']}")
                    st.write(f"RMSE: {results['linear']['rmse']}")
                    st.write(f"R²: {results['linear']['r2']}")

                    st.write("## Resultados de Árbol de Decisión")
                    st.write(f"MAE: {results['tree']['mae']}")
                    st.write(f"RMSE: {results['tree']['rmse']}")
                    st.write(f"R²: {results['tree']['r2']}")
else:
    st.info("Por favor, sube un archivo CSV para comenzar.")
