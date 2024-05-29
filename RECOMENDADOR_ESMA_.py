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
    
    add_user_data(user_data, f'N_{id_preferencia}', viviendas_indices, viviendas_distancias, similarity)
    
    viviendas_indices
    viviendas_distancias
    similarity
    
    return n, preferecniasLog10, viviendas_indices, viviendas_distancias,similarity
