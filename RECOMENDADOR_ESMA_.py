# -*- coding: utf-8 -*-
"""
Created on Mon May 20 01:24:52 2024

@author: UTM
"""
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.optimize import minimize
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
import networkx as nx
import json

url='https://raw.githubusercontent.com/Emax1/UCO/main/DATA_CLUSTER.csv'
df = pd.read_csv(url,sep=';')

filtro=df['CLUSTERS'] == 0
df[df['CLUSTERS'] == 0].drop(columns=['CLUSTERS'])
df[df['CLUSTERS'] == 1].drop(columns=['CLUSTERS'])
df[df['CLUSTERS'] == 2].drop(columns=['CLUSTERS'])
df[df['CLUSTERS'] == 3].drop(columns=['CLUSTERS'])


# Ejemplo de uso:
cntr1 = np.array([[1.09596557, 1.3395186 , 0.43391829, 1.05127016, 0.28866462, 0.59347439, 1.15834743, 1.73605758, 1.607543  ],
                       [1.21173477, 1.37694887, 0.57504857, 1.38316483, 0.39871606, 0.87635972, 1.07171046, 1.64678945, 1.64727528],
                       [0.81671054, 1.14504672, 0.31804309, 0.69581386, 0.22226769, 0.61604004, 1.17845305, 1.77336872, 1.61765853],
                       [0.95745016, 1.24587318, 0.3355842 , 0.72059413, 0.23812295, 0.58429637, 1.18133344, 1.76213257, 1.6032819 ]])



cntr1
###preferencias


# Función para calcular la distancia euclidiana entre dos puntos
def calcular_distancia(x, y):
    distancia = np.linalg.norm(x - y)  # Distancia euclidiana
    return distancia

# Función objetivo para minimizar la distancia euclidiana
def objetivo(x, registro):
    distancia = calcular_distancia(x, registro)
    return distancia

# Función para calcular la distancia entre el registro y cada fila del DataFrame utilizando PSO
def calcular_distancias_pso(registro, df,n):
    lb = [min(df[col]) for col in df.columns]  # Límites inferiores para cada dimensión
    ub = [max(df[col]) for col in df.columns]  # Límites superiores para cada dimensión
    bounds = [(low, high) for low, high in zip(lb, ub)]  # Limitaciones de las variables

    def calcular_distancia(x, y):
        return objetivo(x, y)

    res = minimize(calcular_distancia, registro, args=(registro,), method='Powell', bounds=bounds, options={'maxiter': 100})

    # Obtener la distancia mínima y el registro correspondiente
    distancia_minima = res.fun
    registro_minimo = res.x

    # Calcular las distancias entre el registro mínimo y todas las filas del DataFrame
    distancias = [calcular_distancia(registro_minimo, row.values) for _, row in df.iterrows()]

    # Obtener los índices de las 3 distancias mínimas
    indices_top = np.argsort(distancias)[:n]

    # Obtener los valores de las 3 distancias mínimas
    menores_distancias = [distancias[i] for i in indices_top]

    return distancia_minima, registro_minimo, menores_distancias, indices_top



def add_user_data(user_data, user, viviendas, similitud, calificaciones):
    # Si el usuario ya existe en el diccionario, actualizar la información
    if user in user_data:
        user_data[user]['viviendas'].extend(viviendas)
        user_data[user]['similitud'].extend(similitud)
        user_data[user]['calificaciones'].extend(calificaciones)
    # Si el usuario no existe en el diccionario, agregar la información nueva
    else:
        user_data[user] = {
            'viviendas': viviendas,
            'similitud': similitud,
            'calificaciones': calificaciones
        }


# Diccionario inicial de datos de usuarios
user_data = {
}


def MD_FUZZY(cntr,id_preferencia,preferencias,user_data):
    preferecniasLog10=np.log10(pd.DataFrame([preferencias]).iloc[[0]]+1)
    u_pred, _, _, _,_,_ = fuzz.cluster.cmeans_predict(preferecniasLog10.T.values, cntr, 2, error=0.005, maxiter=1000)
    n=np.round(u_pred.T*10)
    url='https://raw.githubusercontent.com/Emax1/UCO/main/DATA_CLUSTER.csv'
    df = pd.read_csv(url,sep=';')
    df[df['CLUSTERS'] == 0].drop(columns=['CLUSTERS'])
    
    distancia_minima0, registro_minimo0, menores_distancias0, indices_c0 = calcular_distancias_pso(preferecniasLog10.values[0],df[df['CLUSTERS'] == 0].drop(columns=['CLUSTERS']),int(n[0,0]))
    distancia_minima1, registro_minimo1, menores_distancias1, indices_c1 = calcular_distancias_pso(preferecniasLog10.values[0],df[df['CLUSTERS'] == 1].drop(columns=['CLUSTERS']),int(n[0,1]))
    distancia_minima2, registro_minimo2, menores_distancias2, indices_c2 = calcular_distancias_pso(preferecniasLog10.values[0],df[df['CLUSTERS'] == 2].drop(columns=['CLUSTERS']),int(n[0,2]))
    distancia_minima3, registro_minimo3, menores_distancias3, indices_c3 = calcular_distancias_pso(preferecniasLog10.values[0],df[df['CLUSTERS'] == 3].drop(columns=['CLUSTERS']),int(n[0,3]))
    
    viviendas_indices=np.concatenate((indices_c0, indices_c1,indices_c2,indices_c3))
    viviendas_distancias=np.concatenate((menores_distancias0, menores_distancias1,menores_distancias2,menores_distancias3))
    TOP_HOUSIG=df.iloc[viviendas_indices].drop(columns=['CLUSTERS'])
    
    # Calcular la similitud del coseno
    #preferecnias son datos_nuevos
    similarity = cosine_similarity(preferecniasLog10, TOP_HOUSIG)
    
    add_user_data(user_data, f'PREFERENCIA_{id_preferencia}', viviendas_indices, viviendas_distancias, similarity)
    
    viviendas_indices
    viviendas_distancias
    similarity
    
    return n, preferecniasLog10, viviendas_indices, viviendas_distancias,similarity
    
id_preferencia=3
n, preferecniasLog10, viviendas_indices, viviendas_distancias,similarity= MD_FUZZY(cntr1,id_preferencia,preferencias1,user_data)

viviendas_indices
user_data


df_concat_rows=pd.DataFrame()
df_concat_rows.shape[0]





id_preferencia=len(df_concat_rows)

preferencias1=df_concat_rows.iloc[len(df_concat_rows)-1].values

n, preferecniasLog10, viviendas_indices, viviendas_distancias,similarity= MD_FUZZY(cntr1,id_preferencia,preferencias1,user_data)

####PROBAR MODELO
# Supongamos que tienes los valores de una fila en una lista llamada `valores`
preferencias1 = [14.57087195,33.47465456,0.678789802,0.371342386,0.927562365,4.973912956,13.91774998,54.46487486,30.51297683]  # Agrega todos los valores de la fila aquí
#preferecnia tiene que almacenarce enun data frame
#y la recomendación con la key en un diccionario

preferencias2 = [12,21,1,12,1,2,15,61,39] 
preferencias3 = [4,11,1,4,1,3,15,62,39] 
preferencias4 = [14,22,3,26,1,7,12,55,44] 
preferencias5 = [11,22,1,3,1,2,14,61,38] 



# Crea el DataFrame utilizando pd.DataFrame()
preferencias1=np.log10(pd.DataFrame([valores1]).iloc[[0]]+1)
preferencias2=np.log10(pd.DataFrame([valores2]).iloc[[0]]+1)
preferencias3=np.log10(pd.DataFrame([valores3]).iloc[[0]]+1)
preferencias4=np.log10(pd.DataFrame([valores4]).iloc[[0]]+1)
preferencias5=np.log10(pd.DataFrame([valores5]).iloc[[0]]+1)
new_record=np.log10(pd.DataFrame([preferencias1]).iloc[[0]]+1)



# Añadir el nuevo registro al DataFrame existente usando append()
df_concat_rows = df_concat_rows.append(new_record, ignore_index=True)


####
#diciionario de datos

#automatizar ingreso de datos

#simular datos


## llamar la función desde github
