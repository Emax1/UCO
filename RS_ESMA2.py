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
        #valoracion = (registro.reset_index(drop=True) == TOP7_HOUSIG.iloc[[i]].reset_index(drop=True)).astype(int)
        #resultados.append(np.sum(pesos * valoracion.values))    
        valoracion = valor_pesos_hetero_(TOP7_HOUSIG.iloc[[i]].reset_index(drop=True),registro.reset_index(drop=True))
        resultados.append(valoracion)
        deseos_id.append(id_deseo)
    # Crear DataFrame
    df = pd.DataFrame({'lista_indices': lista_indices,'deseos_ID':deseos_id,'resultados': resultados})
    df = pd.DataFrame([lista_indices,deseos_id,resultados]).T
    df.columns = ['lista_indices','deseos_ID', 'resultados']
    # Crear una nueva columna que contenga las etiquetas "relevante" o "no relevante" en función de si el valor en la columna "resultados" es mayor que 70
    df['Relevante'] = df['resultados'].apply(lambda x: 'relevante' if x >= 75 else 'no relevante')
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
        #df_valorado = df_valorado.append(df0, ignore_index=True)
        df_valorado = pd.concat([df_valorado, df0], ignore_index=True)
    relevantes=Counter(df_valorado['Relevante'])['relevante']
    norelevantes=Counter(df_valorado['Relevante'])['no relevante']
    precision=relevantes/(norelevantes+relevantes)*100
    print('RELEVANTES',relevantes,'/(NO RELEVANTES+RELEVANTES)',norelevantes+relevantes,'--->Top-K PRECISION:',precision,"%")
    return df_valorado,relevantes,norelevantes,precision

def valor_pesos_hetero_(viviendas,deseos): 
    pesos_heterogenios= np.full(26, 0)
    pesos_heterogenios[0]=peso_numerica(viviendas['precio'],deseos['precio'],39.5194)
    pesos_heterogenios[1]=peso_numerica(viviendas['metros_reales'],deseos['metros_reales'],25.7451)
    pesos_heterogenios[2]=peso_numerica(viviendas['habitaciones'],deseos['habitaciones'],6.2467)
    pesos_heterogenios[3]=peso_numerica(viviendas['baños'],deseos['baños'],5.1951)
    pesos_heterogenios[4]=peso_cualitativa(viviendas['localizacion_Plaza de Cuba - República Argentina'],deseos['localizacion_Plaza de Cuba - República Argentina'],2.4457)
    pesos_heterogenios[5]=peso_cualitativa(viviendas['garaje'],deseos['garaje'],2.2578)
    pesos_heterogenios[6]=peso_cualitativa(viviendas['armarios_empotrados'],deseos['armarios_empotrados'],1.4783)
    pesos_heterogenios[7]=peso_cualitativa(viviendas['trastero'],deseos['trastero'],1.3972)
    pesos_heterogenios[8]=peso_cualitativa(viviendas['terraza'],deseos['terraza'],1.1438)
    pesos_heterogenios[9]=peso_cualitativa(viviendas['condicion_Promoción de obra nueva'],deseos['condicion_Promoción de obra nueva'],1.1376)
    pesos_heterogenios[10]=peso_cualitativa(viviendas['balcon'],deseos['balcon'],0.9944)
    pesos_heterogenios[11]=peso_cualitativa(viviendas['ascensor'],deseos['ascensor'],0.8224)
    pesos_heterogenios[12]=peso_cualitativa(viviendas['aire_acondicionado'],deseos['aire_acondicionado'],0.814)
    pesos_heterogenios[13]=peso_cualitativa(viviendas['calefaccion'],deseos['calefaccion'],0.7822)
    pesos_heterogenios[14]=peso_cualitativa(viviendas['piso_Primeros_pisos'],deseos['piso_Primeros_pisos'],0.741)
    pesos_heterogenios[15]=peso_cualitativa(viviendas['piso_Muchas_plantas'],deseos['piso_Muchas_plantas'],0.6565)
    pesos_heterogenios[16]=peso_cualitativa(viviendas['piso_Ultimos_pisos'],deseos['piso_Ultimos_pisos'],0.6539)
    pesos_heterogenios[17]=peso_cualitativa(viviendas['condicion_Segunda mano/para reformar'],deseos['condicion_Segunda mano/para reformar'],0.6404)
    pesos_heterogenios[18]=peso_cualitativa(viviendas['condicion_Segunda mano/buen estado'],deseos['condicion_Segunda mano/buen estado'],0.6105)
    pesos_heterogenios[19]=peso_cualitativa(viviendas['localizacion_Tablada'],deseos['localizacion_Tablada'],0.5884)
    pesos_heterogenios[20]=peso_cualitativa(viviendas['localizacion_Ramón de Carranza - Madre Rafols'],deseos['localizacion_Ramón de Carranza - Madre Rafols'],0.5675)
    pesos_heterogenios[21]=peso_cualitativa(viviendas['piso_Bajo'],deseos['piso_Bajo'],0.5402)
    pesos_heterogenios[22]=peso_cualitativa(viviendas['localizacion_Parque de los Principes - Calle Niebla'],deseos['localizacion_Parque de los Principes - Calle Niebla'],0.4815)
    pesos_heterogenios[23]=peso_cualitativa(viviendas['localizacion_Los Remedios'],deseos['localizacion_Los Remedios'],0.3726)
    pesos_heterogenios[24]=peso_cualitativa(viviendas['localizacion_Asunción - Adolfo Suárez'],deseos['localizacion_Asunción - Adolfo Suárez'],0.3648)
    pesos_heterogenios[25]=peso_cualitativa(viviendas['localizacion_Blas Infante'],deseos['localizacion_Blas Infante'],0.2531)
    sumapesos2=np.sum(pesos_heterogenios)
    return sumapesos2

def peso_numerica(pa,pb,peso_asignado):
    dif=np.abs(pa-pb)
    suma=pa+pb
    proporcion=dif/suma
    peso_actualizado=peso_asignado-(peso_asignado*proporcion)
    return peso_actualizado

def peso_cualitativa(value1, value2, peso_asignado):
    if (value1.reset_index(drop=True) == value2.reset_index(drop=True)).all():
        peso_actualizado = peso_asignado
    else:
        peso_actualizado = 0
    return peso_actualizado
