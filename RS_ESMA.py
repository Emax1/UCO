# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:52:02 2024

@author: UTM
"""
#IMPORTAR FUNCIONES
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

def datos_encode ():
    url_casas='https://raw.githubusercontent.com/Emax1/UCO/main/idealista_machine_learning.csv'
    df_casas=pd.read_csv(url_casas)
    
    #ELIMINAR COLUMNA DE TÍTULO
    df_casas0=df_casas.drop(['titulo'],axis=1)
    
    #LEER DATOS ESPACIALES DE VIVIENDA
    url_ubi='https://raw.githubusercontent.com/Emax1/UCO/main/ubi.csv'
    
    df_ubi=pd.read_csv(url_ubi)
    
    #UNIFICAR DATASET
    df_vivienda=pd.concat([df_ubi, df_casas0], axis=1)
    df_vivienda.info()
    df_vivienda1=df_vivienda
    
    categorical_columns=["localizacion", "piso", "condicion"]
    
    #CLUSTER DE VIVIENDAS, PREFERENCIAS DEL USUARIO
    
    
    # Creamos una instancia de OneHotEncoder
    encoder = OneHotEncoder(sparse=False)
    
    # Ajustamos y transformamos las columnas categóricas
    encoded_cols = encoder.fit_transform(df_vivienda1[["localizacion", "piso", "condicion"]])
    
    # Creamos un DataFrame con las columnas codificadas
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(["localizacion", "piso", "condicion"]))
    
    # Concatenamos el DataFrame codificado con el DataFrame original
    df_encoded = pd.concat([df_vivienda1.drop(["localizacion", "piso", "condicion"], axis=1), encoded_df], axis=1)
    return df_encoded


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

    #Counter(labelsFZ)
    #Counter(labels)
    
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
    #df_encoded_clase.info()


    # Filtrar los datos por la característica de la columna 'CLUSTER'
    viviendas_C2= df_encoded_clase[df_encoded_clase['CLUSTER'] == 2]
    #viviendas_C2.info()

    viviendas_C1= df_encoded_clase[df_encoded_clase['CLUSTER'] == 1]
    #viviendas_C1.info()


    viviendas_C0= df_encoded_clase[df_encoded_clase['CLUSTER'] == 0]
    #viviendas_C0.info()

    #Cluster 2:2 ,Cluster 1:5 ,Cluster 0:0
    #Counter(labelsFZ)


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
    #print("Índices de vecinos más cercanos:", Indices_totales)
    #print("Distancias de vecinos más cercanos:", Distancias_totales)
    lista_indices = Indices_totales.tolist()

    TOP7_HOUSIG=df_encoded.iloc[lista_indices]
    #nuevos_datos.values
    #TOP7_HOUSIG.columns
    #TOP7_HOUSIG.iloc[[0]].values
    relevancia=CALIFICAR_VIVIENDAS(nuevos_datos, TOP7_HOUSIG,id_deseo,lista_indices,pesos)
    #return TOP7_HOUSIG,ic2,ic1,ic0,relevancia
    #return TOP7_HOUSIG,relevancia
    return relevancia

def DF_VALORADOS(viviendas,deseos,id_deseo,pesos,n_deseos):
    df_valorado = pd.DataFrame()
    for i in range(len(n_deseos)):
        df0=MUESTRA_VIVIENDAS(viviendas,deseos.iloc[[n_deseos[i]]],id_deseo,pesos)
        df_valorado = df_valorado.append(df0, ignore_index=True)
    relevantes=Counter(df_valorado['Relevante'])['relevante']
    norelevantes=Counter(df_valorado['Relevante'])['no relevante']
    precision=relevantes/norelevantes*100
    #print('RELEVANTES',relevantes,'/NO RELEVANTES',norelevantes,'--->Top-K PRECISION:',precision,"%")
    return df_valorado,relevantes,norelevantes,precision
