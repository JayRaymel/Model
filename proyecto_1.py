import pandas as pd
from supabase_config import supabase_client
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
#from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st
import plotly.express as px

#Cargar datos de Supabase
response = supabase_client.table('envios').select('*').execute()
envios = pd.DataFrame(response.data)

#Cargar los nombres de las regiones
response_regiones = supabase_client.table('regiones').select('id_region, nombre_region').execute()
regiones = pd.DataFrame(response_regiones.data)

#Cargar los nombres de eventos, tipo_servicio y rutas desde la base de datos o definidos en el código
response_eventos = supabase_client.table('eventos').select('id_evento, nombre_evento').execute()
eventos = pd.DataFrame(response_eventos.data)

response_tipo_servicio = supabase_client.table('tipos_servicio').select('id_tipo_servicio, nombre_servicio').execute()
tipo_servicio = pd.DataFrame(response_tipo_servicio.data)

#Suponiendo que las rutas se definen por la región de origen y destino
response_rutas = supabase_client.table('rutas').select('id_ruta, id_region_origen, id_region_destino').execute()
rutas = pd.DataFrame(response_rutas.data)

#Obtener los nombres de las regiones
regiones_map = dict(zip(regiones['id_region'], regiones['nombre_region']))

#Agregar el nombre de la ruta concatenando los nombres de las regiones
rutas['nombre_ruta'] = rutas['id_region_origen'].map(regiones_map) + " -> " + rutas['id_region_destino'].map(regiones_map)

#Preprocesar los datos
envios['fecha_envio'] = pd.to_datetime(envios['fecha_envio'])
envios['mes'] = envios['fecha_envio'].dt.month

#Codificar las columnas categóricas
label_encoder_region = LabelEncoder()
label_encoder_evento = LabelEncoder()
label_encoder_tipo_servicio = LabelEncoder()
label_encoder_ruta = LabelEncoder()

envios['id_region_encoded'] = label_encoder_region.fit_transform(envios['id_region'])
envios['id_evento_encoded'] = label_encoder_evento.fit_transform(envios['id_evento'])
envios['id_tipo_servicio_encoded'] = label_encoder_tipo_servicio.fit_transform(envios['id_tipo_servicio'])
envios['id_ruta_encoded'] = label_encoder_ruta.fit_transform(envios['id_ruta'])

#Crear mapeos para las categorías
region_map = dict(zip(regiones['id_region'], regiones['nombre_region']))
evento_map = dict(zip(eventos['id_evento'], eventos['nombre_evento']))
tipo_servicio_map = dict(zip(tipo_servicio['id_tipo_servicio'], tipo_servicio['nombre_servicio']))
ruta_map = dict(zip(rutas['id_ruta'], rutas['nombre_ruta']))

#Definir las características y el objetivo
features = ['id_region_encoded', 'cantidad_envios', 'id_evento_encoded', 'id_tipo_servicio_encoded', 'id_ruta_encoded', 'mes']
X = envios[features]
y = envios['cantidad_envios']  #Cambiar a 'cantidad_envios' para predecir la demanda

#Entrenar el modelo de Random Forest para regresión
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

#Evaluar el modelo (opcional)
#y_pred = rf.predict(X)
#print("MAE:", mean_absolute_error(y, y_pred))
#print("RMSE:", mean_squared_error(y, y_pred, squared=False))

#Predicciones
predicciones = rf.predict(X)

#Streamlit para mostrar la aplicación
st.title('Predicción de la Demanda de Envíos por Región y Mes')

#Mostrar un selectbox con los nombres de las regiones
region_name = st.selectbox('Selecciona la región', list(region_map.values()))

#Obtener el id_region correspondiente a la región seleccionada
region_id = list(region_map.keys())[list(region_map.values()).index(region_name)]

#Seleccionar mes
mes = st.selectbox('Mes', ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'])

#Convertir el mes seleccionado en el número correspondiente
mes_num = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'].index(mes) + 1

#Filtrar los datos
datos_filtros = envios[(envios['mes'] == mes_num) & (envios['id_region'] == region_id)]

#Mostrar las predicciones
if not datos_filtros.empty:
    st.write(f'Predicción de la demanda de envíos para la región {region_name} en el mes {mes}')
    
    #Mapear las columnas codificadas a sus nombres descriptivos
    datos_filtros.loc[:, 'region_nombre'] = datos_filtros['id_region'].map(region_map)
    datos_filtros.loc[:, 'evento_nombre'] = datos_filtros['id_evento'].map(evento_map)
    datos_filtros.loc[:, 'tipo_servicio_nombre'] = datos_filtros['id_tipo_servicio'].map(tipo_servicio_map)
    datos_filtros.loc[:, 'ruta_nombre'] = datos_filtros['id_ruta'].map(ruta_map)
    
    #Mostrar las columnas con nombres descriptivos y las predicciones
    datos_filtros['prediccion_demanda'] = predicciones[:len(datos_filtros)]  # Asegurarse de que el tamaño sea correcto
    st.write(datos_filtros[['cantidad_envios', 'tarifa_promedio', 'region_nombre', 'evento_nombre', 'tipo_servicio_nombre', 'ruta_nombre', 'prediccion_demanda']])
    #Crear un gráfico de barras con Plotly
    fig = px.bar(
        datos_filtros,
        x='ruta_nombre',
        y='prediccion_demanda',
        color='tipo_servicio_nombre',
        title='Predicción de la Demanda por Ruta',
        labels={'prediccion_demanda': 'Demanda Predicha', 'ruta_nombre': 'Ruta'}
    )
    #Mostrar el gráfico en Streamlit
    st.plotly_chart(fig)
else:
    st.write(f'No hay datos disponibles para la región {region_name} en el mes {mes}')