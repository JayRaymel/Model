import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# Función para cargar y previsualizar los datos
def load_data(uploaded_file):
    try:
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
        "tree": {"mae": mae_tree, "rmse": rmse_tree, "r2": r2_tree},
        "tree_feature_importances": tree_model.feature_importances_
    }

# Función para procesar los datos
def preprocess_data(data, target_column):
    # Verificar si hay valores nulos y permitir imputar
    if data.isnull().sum().sum() > 0:
        st.write("### Comprobación de valores nulos:")
        st.write(data.isnull().sum())
        imputar = st.radio("¿Cómo deseas manejar los valores nulos?", ('Imputar con la media', 'Eliminar filas con valores nulos'))
        if imputar == 'Imputar con la media':
            data.fillna(data.mean(), inplace=True)
        else:
            data.dropna(inplace=True)
        st.write("Valores nulos procesados.")

    # Verificar y codificar variables categóricas
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        st.write(f"Columna '{col}' convertida a numérica.")

    # Separar características y objetivo
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Normalizar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# Interfaz de Streamlit
st.title("📊 Predicción de Ventas con Modelos de Machine Learning")

# Subir archivo CSV
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])
if uploaded_file is not None:
    # Cargar datos
    data = load_data(uploaded_file)

    if data is not None:
        # Seleccionar columna objetivo
        target_column = st.selectbox("Selecciona la columna objetivo (ventas):", data.columns)

        if target_column:
            # Preprocesar los datos
            X_scaled, y = preprocess_data(data, target_column)

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

            # Mostrar gráfica de comparación de modelos
            st.write("### Comparación de Modelos")
            models = ['Regresión Lineal', 'Árbol de Decisión']
            maes = [results['linear']['mae'], results['tree']['mae']]
            rmse = [results['linear']['rmse'], results['tree']['rmse']]
            r2 = [results['linear']['r2'], results['tree']['r2']]

            fig, ax = plt.subplots(1, 3, figsize=(15, 5))

            ax[0].bar(models, maes, color='skyblue')
            ax[0].set_title("MAE")

            ax[1].bar(models, rmse, color='salmon')
            ax[1].set_title("RMSE")

            ax[2].bar(models, r2, color='lightgreen')
            ax[2].set_title("R²")

            st.pyplot(fig)

            # Importancia de características en el árbol de decisión
            st.write("### Importancia de características (Árbol de Decisión)")
            feature_importances = results['tree_feature_importances']
            features = data.drop(target_column, axis=1).columns
            feature_df = pd.DataFrame({'feature': features, 'importance': feature_importances})
            feature_df = feature_df.sort_values(by='importance', ascending=False)
            st.write(feature_df)

else:
    st.info("Por favor, sube un archivo CSV para comenzar.")

