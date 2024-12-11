import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Streamlit: Cargar archivo CSV
st.title('Predicción de la Demanda de Envíos por Región y Mes')

# Subir archivo CSV
uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"])

if uploaded_file is not None:
    # Cargar datos desde el archivo CSV
    envios = pd.read_csv(uploaded_file)
    
    # Si tienes un archivo CSV con las regiones, eventos, tipos de servicio y rutas, también puedes cargarlo de la misma manera
    # Cargar los datos de regiones, eventos, tipos de servicio, y rutas desde archivos CSV si están disponibles
    uploaded_file_regiones = st.file_uploader("Cargar archivo de Regiones", type=["csv"])
    uploaded_file_eventos = st.file_uploader("Cargar archivo de Eventos", type=["csv"])
    uploaded_file_tipos_servicio = st.file_uploader("Cargar archivo de Tipos de Servicio", type=["csv"])
    uploaded_file_rutas = st.file_uploader("Cargar archivo de Rutas", type=["csv"])

    if uploaded_file_regiones is not None:
        regiones = pd.read_csv(uploaded_file_regiones)
    
    if uploaded_file_eventos is not None:
        eventos = pd.read_csv(uploaded_file_eventos)
    
    if uploaded_file_tipos_servicio is not None:
        tipo_servicio = pd.read_csv(uploaded_file_tipos_servicio)
    
    if uploaded_file_rutas is not None:
        rutas = pd.read_csv(uploaded_file_rutas)

    # Procesar los datos cargados
    envios['fecha_envio'] = pd.to_datetime(envios['fecha_envio'])
    envios['mes'] = envios['fecha_envio'].dt.month

    # Codificar las columnas categóricas
    label_encoder_region = LabelEncoder()
    label_encoder_evento = LabelEncoder()
    label_encoder_tipo_servicio = LabelEncoder()
    label_encoder_ruta = LabelEncoder()

    envios['id_region_encoded'] = label_encoder_region.fit_transform(envios['id_region'])
    envios['id_evento_encoded'] = label_encoder_evento.fit_transform(envios['id_evento'])
    envios['id_tipo_servicio_encoded'] = label_encoder_tipo_servicio.fit_transform(envios['id_tipo_servicio'])
    envios['id_ruta_encoded'] = label_encoder_ruta.fit_transform(envios['id_ruta'])

    # Crear mapeos para las categorías
    region_map = dict(zip(regiones['id_region'], regiones['nombre_region']))
    evento_map = dict(zip(eventos['id_evento'], eventos['nombre_evento']))
    tipo_servicio_map = dict(zip(tipo_servicio['id_tipo_servicio'], tipo_servicio['nombre_servicio']))
    ruta_map = dict(zip(rutas['id_ruta'], rutas['nombre_ruta']))

    # Definir las características y el objetivo
    features = ['id_region_encoded', 'cantidad_envios', 'id_evento_encoded', 'id_tipo_servicio_encoded', 'id_ruta_encoded', 'mes']
    X = envios[features]
    y = envios['cantidad_envios']  # Cambiar a 'cantidad_envios' para predecir la demanda

    # Entrenar el modelo de Random Forest para regresión
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Predicciones
    predicciones = rf.predict(X)

    # Mostrar los resultados en Streamlit
    region_name = st.selectbox('Selecciona la región', list(region_map.values()))
    region_id = list(region_map.keys())[list(region_map.values()).index(region_name)]

    mes = st.selectbox('Mes', ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'])
    mes_num = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'].index(mes) + 1

    # Filtrar los datos
    datos_filtros = envios[(envios['mes'] == mes_num) & (envios['id_region'] == region_id)]

    # Mostrar las predicciones
    if not datos_filtros.empty:
        st.write(f'Predicción de la demanda de envíos para la región {region_name} en el mes {mes}')

        # Mapear las columnas codificadas a sus nombres descriptivos
        datos_filtros['region_nombre'] = datos_filtros['id_region'].map(region_map)
        datos_filtros['evento_nombre'] = datos_filtros['id_evento'].map(evento_map)
        datos_filtros['tipo_servicio_nombre'] = datos_filtros['id_tipo_servicio'].map(tipo_servicio_map)
        datos_filtros['ruta_nombre'] = datos_filtros['id_ruta'].map(ruta_map)

        # Mostrar las columnas con nombres descriptivos y las predicciones
        datos_filtros['prediccion_demanda'] = predicciones[:len(datos_filtros)]
        st.write(datos_filtros[['cantidad_envios', 'tarifa_promedio', 'region_nombre', 'evento_nombre', 'tipo_servicio_nombre', 'ruta_nombre', 'prediccion_demanda']])

        # Crear un gráfico de barras con Plotly
        fig = px.bar(
            datos_filtros,
            x='ruta_nombre',
            y='prediccion_demanda',
            color='tipo_servicio_nombre',
            title='Predicción de la Demanda por Ruta',
            labels={'prediccion_demanda': 'Demanda Predicha', 'ruta_nombre': 'Ruta'}
        )
        st.plotly_chart(fig)
    else:
        st.write(f'No hay datos disponibles para la región {region_name} en el mes {mes}')
