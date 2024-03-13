# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 19:47:47 2024

@author: EMANUEL MUÑOZ
"""
#LIBRERIAS
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.decomposition import PCA
import skfuzzy as fuzz
from matplotlib.patches import Ellipse
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

#DATOS DE VIVIENDA, DATOS ESPACIALES
#LEER CARACTERISTICAS DE VIVIENDA
url_casas='https://raw.githubusercontent.com/Emax1/UCO/main/idealista_machine_learning.csv'
df_casas=pd.read_csv(url_casas)

#ELIMINAR COLUMNA DE TÍTULO
df_casas0=df_casas.drop(['titulo'],axis=1)

#LEER DATOS ESPACIALES DE VIVIENDA
url_ubi='https://raw.githubusercontent.com/Emax1/UCO/main/ubi.csv'
df_ubi=pd.read_csv(url_ubi)
df_casas0.info()
df_ubi.info()

#UNIFICAR DATASET
df_vivienda=pd.concat([df_ubi, df_casas0], axis=1)
df_vivienda.info()
df_vivienda1=df_vivienda

#Tranformar variables

categorical_columns=["localizacion", "piso", "condicion"]

#CLUSTER DE VIVIENDAS, PREFERENCIAS DEL USUARIO


# Creamos una instancia de OneHotEncoder
encoder = OneHotEncoder(sparse=False)

# Ajustamos y transformamos las columnas categóricas
encoded_cols = encoder.fit_transform(df_vivienda1[["localizacion", "piso", "condicion"]])

# Creamos un DataFrame con las columnas codificadas
encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names(["localizacion", "piso", "condicion"]))

# Concatenamos el DataFrame codificado con el DataFrame original
df_encoded = pd.concat([df_vivienda1.drop(["localizacion", "piso", "condicion"], axis=1), encoded_df], axis=1)

print(df_encoded)
df_encoded.info()




######## TRANSFORMAR UNA VARIABLE NUMÉRICA

df_vivienda1["precio"]

np.log(df_vivienda1['precio'])


# Creamos un histograma de la columna 'log_variable'
plt.hist(df_vivienda1["precio"], bins=10, color='skyblue', edgecolor='black')

# Agregamos etiquetas y título
plt.xlabel('Valor del precio')
plt.ylabel('Frecuencia')
plt.title('Histograma del Logaritmo de la Variable')

# Mostramos el histograma
plt.show()


# Creamos un histograma de la columna 'log_variable'
plt.hist(np.log(df_vivienda1['precio']), bins=10, color='skyblue', edgecolor='black')

# Agregamos etiquetas y título
plt.xlabel('Valor del Logaritmo')
plt.ylabel('Frecuencia')
plt.title('Histograma del Logaritmo de la Variable')

# Mostramos el histograma
plt.show()

df_vivienda1.info()


######TRANFORMAR METROS CUADRADOS
df_vivienda1["metros_reales"]

np.log(df_vivienda1['metros_reales'])


# Creamos un histograma de la columna 'log_variable'
plt.hist(df_vivienda1["metros_reales"], bins=10, color='skyblue', edgecolor='black')

# Agregamos etiquetas y título
plt.xlabel('Valor del MT2')
plt.ylabel('Frecuencia')
plt.title('Histograma del Logaritmo de la Variable')

# Mostramos el histograma
plt.show()


# Creamos un histograma de la columna 'log_variable'
plt.hist(np.log(df_vivienda1['metros_reales']), bins=10, color='skyblue', edgecolor='black')

# Agregamos etiquetas y título
plt.xlabel('Valor del MT2')
plt.ylabel('Frecuencia')
plt.title('Histograma del Logaritmo de la Variable')

# Mostramos el histograma
plt.show()







#ANÁLISIS


#PROBAR MODELO NO SUPERVISADO, PREDICCIÓN
###CLUSTER KMEAN
######### PREDECIR CLUSTER

kmeans = KMeans(n_clusters=3)
kmeans.fit(df_encoded)
labels = kmeans.predict(df_encoded)

###Viviendas por clase
Counter(labels)

#presentación de las clases
# Ajusta el modelo PCA
pca = PCA()
pca.fit(df_encoded)

# Obtén los loadings
#loadings = pca.components_

componentes_principales = pca.transform(df_encoded)



# Creamos el scatter plot
#ax.scatter(componentes_principales[:,0], componentes_principales[:,1], componentes_principales[:,2], c=labels, label='Componentes principales')
plt.scatter(componentes_principales[:,0], componentes_principales[:,1], c=labels)

# Agregamos etiquetas y título
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.title('Gráfico de Puntos con Color por Clase')

# Mostramos la leyenda
plt.colorbar(label='Tipo')

# Mostramos el gráfico
plt.show()


################
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(componentes_principales[:,0], componentes_principales[:,1], componentes_principales[:,2], c=labels, label='Componentes principales')
# Ajustamos la posición del gráfico
ax.set_position([0.1, 0.1, 0.6, 0.6])  # [left, bottom, width, height]
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.title('PCA en tres dimensiones')
plt.legend()




#######MODELADO FUZZY
# Creamos el modelo FCM
cntr1, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    df_encoded.T.values, 3, 2, error=0.005, maxiter=1000)

# Obtenemos la pertenencia máxima para cada punto
labelsFZ = np.argmax(u, axis=0)

####grado de pertenencia
u.T
u.T[[1]]

Counter(labelsFZ)
Counter(labels)


# Visualizamos el gráfico de dispersión de los datos con colores según la pertenencia
plt.scatter(componentes_principales[:,0], componentes_principales[:,1], c=labelsFZ)

# Agregamos etiquetas y título
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.title('Gráfico de Puntos con Color por Clase')

# Mostramos la leyenda
plt.colorbar(label='Tipo')

# Mostramos el gráfico
plt.show()



# Creamos el modelo FCM para el gráfico con elipses
data = pd.concat([pd.DataFrame(componentes_principales[:,0],columns=["CP1"]), pd.DataFrame(componentes_principales[:,1],columns=["CP2"])], axis=1)


# Creamos el modelo FCM
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    data.T.values, 3, 2, error=0.005, maxiter=1000)

# Obtenemos la pertenencia máxima para cada punto
labelsFZ = np.argmax(u, axis=0)

# Visualizamos el gráfico de dispersión de los datos con colores según la pertenencia
plt.scatter(data['CP1'], data['CP2'], c=labelsFZ)

# Añadimos elipses alrededor de los centroides
for i, centroide in enumerate(cntr):
    cov_matrix = np.cov(data[labelsFZ == i].T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * np.sqrt(eigenvalues)
    ellipse = Ellipse(xy=centroide, width=width, height=height, angle=angle, edgecolor='red', fc='None', lw=2)
    plt.gca().add_patch(ellipse)

plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.title('Gráfico de Cluster con Elipses')
plt.grid(True)
plt.show()

#####ANÁLISIS
labelsFZ
df_encoded['precio']

# Crear gráfico de cajas y bigotes
plt.figure(figsize=(8, 6))
sns.boxplot(x=labelsFZ, y=df_encoded['precio'], data=data, palette='viridis')
plt.xlabel('Clase')
plt.ylabel('Precio')
plt.title('Gráfico de Cajas y Bigotes de Precio por Clase')
plt.grid(True)
plt.show()

deseo=df_encoded.iloc[1]
#PREDECIR CLASES MODELO DISUFO
#deseo del AGENTE
nuevos_datos=df_encoded.iloc[[99]]

#MODELO DE PREDICCIÓN
u_pred, _, _, _,_,_ = fuzz.cluster.cmeans_predict(
    nuevos_datos.T.values, cntr1, 2, error=0.005, maxiter=1000)

#GRADO DE PERTENENCIA AL CLUSTER
grado_deseos=u_pred.T


##DETERMINAR EL NÚMERO DE VIVIENDAS RECOMENDADAS POR CLUSTER
np.round(7*grado_deseos)
n_vecinos_x_cluster=np.round(7*grado_deseos)

#CREAR UN DATAFRAME POR CADA CLUSTER FILTAR
df_encoded_clase=df_encoded.copy()

df_encoded_clase['CLUSTER']=labelsFZ
df_encoded_clase.info()


# Filtrar los datos por la característica de la columna 'CLUSTER'
viviendas_C2= df_encoded_clase[df_encoded_clase['CLUSTER'] == 2]
viviendas_C2.info()

viviendas_C1= df_encoded_clase[df_encoded_clase['CLUSTER'] == 1]
viviendas_C1.info()


viviendas_C0= df_encoded_clase[df_encoded_clase['CLUSTER'] == 0]
viviendas_C0.info()

#Cluster 2:2 ,Cluster 1:5 ,Cluster 0:0
Counter(labelsFZ)


#Vecinos mas cercanos de cada cLuster con respecto a los deseos del 
#agente
#Cluster 2

# Ajustar el modelo k-NN
k = 3  # Número de vecinos
knn2 = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
knn2.fit(viviendas_C2.drop(['CLUSTER'],axis=1), viviendas_C2['CLUSTER'])

# Obtener los índices y distancias de los 7 vecinos más cercanos
distances2, indices2 = knn2.kneighbors(nuevos_datos, n_neighbors=10)

# Ajustar el modelo k-NN
k = 3  # Número de vecinos
knn1 = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
knn1.fit(viviendas_C1.drop(['CLUSTER'],axis=1), viviendas_C1['CLUSTER'])

# Obtener los índices y distancias de los 7 vecinos más cercanos
distances1, indices1 = knn1.kneighbors(nuevos_datos, n_neighbors=10)

# Ajustar el modelo k-NN
k = 3  # Número de vecinos
knn0 = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
knn0.fit(viviendas_C0.drop(['CLUSTER'],axis=1), viviendas_C0['CLUSTER'])

# Obtener los índices y distancias de los 7 vecinos más cercanos
distances0, indices0 = knn0.kneighbors(nuevos_datos, n_neighbors=10)

n_c2=int(n_vecinos_x_cluster[0, 0])
n_c1=int(n_vecinos_x_cluster[0, 1])
n_c0=int(n_vecinos_x_cluster[0, 2])

#Indices de los barrios recomendados
ic2=indices2[0][:n_c2].tolist()
ic1=indices1[0][:n_c1].tolist()
ic0=indices0[0][:n_c0].tolist()

#Distancias de los barrios recomendados
dis2=distances2[0][:n_c2].tolist()
dis1=distances1[0][:n_c1].tolist()
dis0=distances0[0][:n_c0].tolist()

#Unir indices
Indices_totales=np.concatenate((ic2, ic1,ic0))

#Unir distancias
Distancias_totales=np.concatenate((dis2, dis1,dis0))

# Imprimir los índices y distancias de los vecinos más cercanos
print("Índices de vecinos más cercanos:", Indices_totales)
print("Distancias de vecinos más cercanos:", Distancias_totales)
lista_indices = Indices_totales.tolist()

TOP7_HOUSIG=df_encoded.iloc[lista_indices]
nuevos_datos.values
TOP7_HOUSIG.columns
TOP7_HOUSIG.iloc[[0]].values
#FUNCIÓN DE CALIFICACIÓN


#FUNCIÓN PARA CALIFICAR VIVIENDAS RECOMENDADAS

registro=nuevos_datos.iloc[[0]]
len(TOP7_HOUSIG)
lista_indices
id_deseo=0.007
pesos= np.full(28, 100 / 28)
    ####TRABAJAR EN UNA FUNCIÓN DE LOS PESOS PARA MEJORAR LA RELEVANCIA

def CALIFICAR_VIVIENDAS(registro, TOP7_HOUSIG,id_deseo,lista_indices,pesos):
    resultados = []
    deseos_id = []
    for i in range(len(TOP7_HOUSIG)):
        valoracion = (registro.reset_index(drop=True) == TOP7_HOUSIG.iloc[[i]].reset_index(drop=True)).astype(int)
        resultados.append(np.sum(pesos * valoracion.values))
        deseos_id.append(id_deseo)
    # Crear DataFrame
    df = pd.DataFrame({'lista_indices': lista_indices,'deseos_ID':deseos_id,'resultados': resultados})
    df = pd.DataFrame([lista_indices,deseos_id,resultados]).T
    df.columns = ['lista_indices','deseos_ID', 'resultados']
    # Crear una nueva columna que contenga las etiquetas "relevante" o "no relevante" en función de si el valor en la columna "resultados" es mayor que 70
    df['Relevante'] = df['resultados'].apply(lambda x: 'relevante' if x > 70 else 'no relevante')
    return df

CALIFICAR_VIVIENDAS(registro, TOP7_HOUSIG,id_deseo,lista_indices,pesos)
CALIFICAR_VIVIENDAS(df_encoded.iloc[[22]], TOP7_HOUSIG,id_deseo,lista_indices,pesos)

#####FUNCIÓN PARAR SELEECIONAR MUESTRA DE VIVIENDAS
nuevos_datos=df_encoded.iloc[[99]]

def MUESTRA_VIVIENDAS(df_encoded,nuevos_datos,id_deseo,pesos):
    #######MODELADO FUZZY
    # Creamos el modelo FCM
    cntr1, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        df_encoded.T.values, 3, 2, error=0.005, maxiter=1000)

    # Obtenemos la pertenencia máxima para cada punto
    labelsFZ = np.argmax(u, axis=0)

    ####grado de pertenencia
    u.T
    u.T[[1]]

    Counter(labelsFZ)
    Counter(labels)
    
    #MODELO DE PREDICCIÓN
    u_pred, _, _, _,_,_ = fuzz.cluster.cmeans_predict(
        nuevos_datos.T.values, cntr1, 2, error=0.005, maxiter=1000)

    #GRADO DE PERTENENCIA AL CLUSTER
    grado_deseos=u_pred.T


    ##DETERMINAR EL NÚMERO DE VIVIENDAS RECOMENDADAS POR CLUSTER
    np.round(7*grado_deseos)
    n_vecinos_x_cluster=np.round(7*grado_deseos)

    #CREAR UN DATAFRAME POR CADA CLUSTER FILTAR
    df_encoded_clase=df_encoded.copy()

    df_encoded_clase['CLUSTER']=labelsFZ
    df_encoded_clase.info()


    # Filtrar los datos por la característica de la columna 'CLUSTER'
    viviendas_C2= df_encoded_clase[df_encoded_clase['CLUSTER'] == 2]
    viviendas_C2.info()

    viviendas_C1= df_encoded_clase[df_encoded_clase['CLUSTER'] == 1]
    viviendas_C1.info()


    viviendas_C0= df_encoded_clase[df_encoded_clase['CLUSTER'] == 0]
    viviendas_C0.info()

    #Cluster 2:2 ,Cluster 1:5 ,Cluster 0:0
    Counter(labelsFZ)


    #Vecinos mas cercanos de cada cLuster con respecto a los deseos del 
    #agente
    #Cluster 2

    # Ajustar el modelo k-NN
    k = 3  # Número de vecinos
    knn2 = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn2.fit(viviendas_C2.drop(['CLUSTER'],axis=1), viviendas_C2['CLUSTER'])

    # Obtener los índices y distancias de los 7 vecinos más cercanos
    distances2, indices2 = knn2.kneighbors(nuevos_datos, n_neighbors=10)

    # Ajustar el modelo k-NN
    k = 3  # Número de vecinos
    knn1 = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn1.fit(viviendas_C1.drop(['CLUSTER'],axis=1), viviendas_C1['CLUSTER'])

    # Obtener los índices y distancias de los 7 vecinos más cercanos
    distances1, indices1 = knn1.kneighbors(nuevos_datos, n_neighbors=10)

    # Ajustar el modelo k-NN
    k = 3  # Número de vecinos
    knn0 = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn0.fit(viviendas_C0.drop(['CLUSTER'],axis=1), viviendas_C0['CLUSTER'])

    # Obtener los índices y distancias de los 7 vecinos más cercanos
    distances0, indices0 = knn0.kneighbors(nuevos_datos, n_neighbors=10)

    n_c2=int(n_vecinos_x_cluster[0, 0])
    n_c1=int(n_vecinos_x_cluster[0, 1])
    n_c0=int(n_vecinos_x_cluster[0, 2])

    #Indices de los barrios recomendados
    ic2=indices2[0][:n_c2].tolist()
    ic1=indices1[0][:n_c1].tolist()
    ic0=indices0[0][:n_c0].tolist()

    #Distancias de los barrios recomendados
    dis2=distances2[0][:n_c2].tolist()
    dis1=distances1[0][:n_c1].tolist()
    dis0=distances0[0][:n_c0].tolist()

    #Unir indices
    Indices_totales=np.concatenate((ic2, ic1,ic0))

    #Unir distancias
    Distancias_totales=np.concatenate((dis2, dis1,dis0))

    # Imprimir los índices y distancias de los vecinos más cercanos
    print("Índices de vecinos más cercanos:", Indices_totales)
    print("Distancias de vecinos más cercanos:", Distancias_totales)
    lista_indices = Indices_totales.tolist()

    TOP7_HOUSIG=df_encoded.iloc[lista_indices]
    nuevos_datos.values
    TOP7_HOUSIG.columns
    TOP7_HOUSIG.iloc[[0]].values
    relevancia=CALIFICAR_VIVIENDAS(nuevos_datos, TOP7_HOUSIG,id_deseo,lista_indices,pesos)
    return TOP7_HOUSIG,ic2,ic1,ic0,relevancia
    #return TOP7_HOUSIG,relevancia
    return relevancia
    
pesos= np.full(28, 100 / 28)
####TRABAJAR EN UNA FUNCIÓN DE LOS PESOS PARA MEJORAR LA RELEVANCIA  

nuevos_datos=df_encoded.iloc[[99]]

MUESTRA_VIVIENDAS(df_encoded,df_encoded.iloc[[99]],pesos)
MUESTRA_VIVIENDAS(df_encoded,df_encoded.iloc[[78]],pesos)

id_deseo=0.0007
TOP7,ic2,ic1,ic0,relevancia=MUESTRA_VIVIENDAS(df_encoded,df_encoded.iloc[[11]],id_deseo,pesos)
TOP7
relevancia


df1=MUESTRA_VIVIENDAS(df_encoded,df_encoded.iloc[[99]],id_deseo,pesos)


MUESTRA_VIVIENDAS(df_encoded,df_encoded.iloc[[78]],id_deseo,pesos)
MUESTRA_VIVIENDAS(df_encoded,df_encoded.iloc[[77]],id_deseo,pesos)
MUESTRA_VIVIENDAS(df_encoded,df_encoded.iloc[[112]],id_deseo,pesos)
MUESTRA_VIVIENDAS(df_encoded,df_encoded.iloc[[22]],id_deseo,pesos)

df1 = df1.append(df2, ignore_index=True)


TOP7
id_deseo=0.007
CALIFICAR_VIVIENDAS(df_encoded.iloc[[99]], TOP7,id_deseo)
CALIFICAR_VIVIENDAS(df_encoded.iloc[[78]], TOP7,id_deseo)
CALIFICAR_VIVIENDAS(df_encoded.iloc[[77]], TOP7,id_deseo)
CALIFICAR_VIVIENDAS(df_encoded.iloc[[112]], TOP7,id_deseo)

CALIFICAR_VIVIENDAS(nuevos_datos, TOP7,id_deseo,lista_indices)

# Nuevo registro a agregar

df_deseos=df.DATAFRAME()
# Agregar el nuevo registro al DataFrame existente
df_deseos = df.append(nuevo_registro, ignore_index=True)

#####
#en las variables numéricas ubicar un rango de valores o una poderación con la distancias.
#empaquetar funciones
#automatizar la función califciar

######DEFINIR PESOS POR RANDOM FOREST
# Cargar el conjunto de datos de Iris
PESOS=df_encoded_clase.copy()
X = PESOS.drop(['CLUSTER','latitud','longitud'],axis=1)
y = PESOS['CLUSTER']

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un modelo de Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar el modelo
rf_model.fit(X_train, y_train)

# Obtener la importancia de las características
importances = rf_model.feature_importances_

# Crear un DataFrame para mostrar la importancia de las características
feature_importance_df = pd.DataFrame(importances, index=X.columns, columns=['Importance'])
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Mostrar la importancia de las características
print("Importancia de las características:")
print(feature_importance_df)

# Graficar la importancia de las características
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df.index, feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importance')
plt.show()


#####PESOS POR XGBOOST

PESOS=df_encoded_clase.copy()
X = PESOS.drop(['CLUSTER','latitud','longitud'],axis=1)
y = PESOS['CLUSTER']

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un modelo de XGBoost
xgb_model = xgb.XGBClassifier(random_state=42)

# Entrenar el modelo
xgb_model.fit(X_train, y_train)

# Obtener la importancia de las características
importances = xgb_model.feature_importances_

# Crear un DataFrame para mostrar la importancia de las características
feature_importance_df = pd.DataFrame(importances, index=X.columns, columns=['Importance'])
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Mostrar la importancia de las características
print("Importancia de las características:")
print(feature_importance_df)

# Graficar la importancia de las características
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df.index, feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importance')
plt.show()

#####GREAFICO DE LA PRECISIÓN métrica de precisión en la parte superior-k (Top-K Precision)


Distribución de Precisión en la Parte Superior-k, podemos calcular la precisión en la parte superior-k para cada usuario en el conjunto de datos y luego visualizar la distribución de estas precisiones. Aquí tienes un ejemplo de cómo hacerlo:




##REgistrar agente


#MATRIZ DE CONFUSIÓN
