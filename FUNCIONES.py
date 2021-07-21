# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:23:49 2020

@author: Emanuel G. Muñoz
"""
#Preprocesamiento de datos
#Estadística Desciptiva y detección de outliers

import numpy as np #útil para ralizar cálculos aanzados
import pandas as pd ##contiene funciones para ayudar al análisis de datos
import matplotlib.pyplot as plt #Para gráficos de muy buena calidad
from tabulate import tabulate
#$ pip install tabulate 
import scipy.stats as ss  #Medidas de forma
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn import linear_model

import random
import operator
import math
#import matplotlib.pyplot as plt 
from scipy.stats import multivariate_normal 



from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import confusion_matrix 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.neural_network import MLPClassifier, MLPRegressor
#Random Forest trees
from sklearn.ensemble import RandomForestClassifier
# training a DescisionTreeClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #visualisation
import numpy as np
from sklearn.model_selection import train_test_split 
sns.set(color_codes=True)



#Función en Python
def DETALLES(dataset):
    #Función de detalles de la base de datos
    filas=dataset.shape[0]
    columnas=dataset.shape[1]
   # nombrescol=dataset.columns
    estructura=dataset.dtypes
    percent_missing = dataset.isnull().sum() * 100 / len(dataset)
    resumen = [['Detalles' , 'Valor'],['n Unidades elementales' , filas],['n Variables' , columnas],
              ['Tipo de datos',estructura],['% Datos perdidos',percent_missing]]
    print(tabulate(resumen,
                   headers='firstrow', 
               tablefmt='fancy_grid',
               stralign='center',
               floatfmt='.0f'))    
    return

def TABLA_FREC_CUAL(x):
    lis = x.unique()
    #Tabla de frecuencias absolutas
    #Obtenes las frecuencias absolutas de cada clase
    # dat = pd.DataFrame(lis, columns=['Atributos'])
    datafi = pd.crosstab(index=x, columns = "fi")
    #Creamos una lista con los valores de las frecuencias
    atributo=datafi.index
    dat = pd.DataFrame(atributo)
    li = datafi.values
    #Agregamos la columna al dataframe
    dat["fi"] = li
    #Observamos
    #Tabbla de frecuencia relativa
    datahi = 100 * datafi["fi"] / len(x)
    datahi = datahi.values
    #Agregamos nueva columna de frecuentas relativas
    dat["hi"] = datahi 
    #Obtenemos las frecuencias absolutas acumuladas
    #Sacamos una lista de los valores donde obtendremos la FI
    Fi = dat["fi"].values
    #Recorremos la lista para ir creando una nueva lista con las sumas
    #Obtener la frecuencia absoluta acumulada
    a = []
    b = 0
    for c in Fi:
        b = c + b
        a.append(b)
    #Agregamos la nueva columna Fi al Dataframe dat
    dat["Fi"] = a 
    #Recorremos lista para obtener la frecuencia relativa acumulada
    Hi = dat["hi"].values
    #Obtenemos Hi
    a = []
    b = 0
    for c in Hi:
        b = c + b
        a.append(b)
    #Agregamos la nueva columna Hi al Dataframe
    dat["Hi"] = a 
    # print(dat)
    fig,ax=plt.subplots()
    ax.bar(lis,datahi)
    plt.show()
    dat=dat.append({'fi':li.sum(),'hi':datahi.sum()}, ignore_index=True)
    print(tabulate(dat, 
                    headers=["Atributo","fi", "pi", "Pi", "Hi"], 
                    tablefmt='fancy_grid',
                    stralign='center',
                    floatfmt='.0f'))
    
    return

def TABLA_FREC_DISC(x):
    lis = x.unique()
    #Tabla de frecuencias absolutas
    #Obtenes las frecuencias absolutas de cada clase
    # dat = pd.DataFrame(lis, columns=['Atributos'])
    datafi = pd.crosstab(index=x, columns = "fi")
    #Creamos una lista con los valores de las frecuencias
    atributo=datafi.index
    dat = pd.DataFrame(atributo)
    li = datafi.values
    #Agregamos la columna al dataframe
    dat["fi"] = li
    #Observamos
    #Tabbla de frecuencia relativa
    datahi = 100 * datafi["fi"] / len(x)
    datahi = datahi.values
    #Agregamos nueva columna de frecuentas relativas
    dat["hi"] = datahi 
    #Obtenemos las frecuencias absolutas acumuladas
    #Sacamos una lista de los valores donde obtendremos la FI
    Fi = dat["fi"].values
    #Recorremos la lista para ir creando una nueva lista con las sumas
    #Obtener la frecuencia absoluta acumulada
    a = []
    b = 0
    for c in Fi:
        b = c + b
        a.append(b)
    #Agregamos la nueva columna Fi al Dataframe dat
    dat["Fi"] = a 
    #Recorremos lista para obtener la frecuencia relativa acumulada
    Hi = dat["hi"].values
    #Obtenemos Hi
    a = []
    b = 0
    for c in Hi:
        b = c + b
        a.append(b)
    #Agregamos la nueva columna Hi al Dataframe
    dat["Hi"] = a 
    # print(dat)
    fig,ax=plt.subplots()
    ax.bar(lis,datahi,width=0.1)
    plt.show()
    dat=dat.append({'fi':li.sum(),'hi':datahi.sum()}, ignore_index=True)
    print(tabulate(dat, 
                    headers=["Valores","fi", "pi", "Pi", "Hi"], 
                    tablefmt='fancy_grid',
                    stralign='center',
                    floatfmt='.0f'))
    
    return

def DESCRIBIR(x,nombre):
    #Función de gráficos y mdeidas de resumen
    print('\n Gráficos Estadísticos \n')
    Estadisticos = list()
    plt.subplot(1,2,1)
    plt.title(nombre)
    plt.hist(x, edgecolor='black',linewidth=1)
    plt.subplot(1,2,2)
    plt.title(nombre)
    plt.boxplot(x,vert=False)
    plt.show()
    #Claculamos los estadísticos para el boxplot del objeto 'Rios'
    Valor_minimo=x.min()
    Valor_maximo=x.max()
    rango=Valor_maximo-Valor_minimo
    Q1=x.quantile(0.25)
    Q2=x.quantile(0.5)
    Q3=x.quantile(0.75)
    IQR=Q3-Q1
    Media=x.mean()
    Mediana=x.median()
    varianza=x.std(ddof=1) #ddof=0 Población, ddof=1 Muestra
    sd=x.std(ddof=1) #ddof=0 Población, ddof=1 Muestra
    cv=sd/Media*100
    asimetria = ss.skew(x)
    #Para valores cercanos a 0, la variable es simétrica. Si es positiva tiene cola a la derecha y si es negativa tiene cola a la izquierda.
    #Calculamos los valores de los bigotes inferior y superior
    BI_Calculado=Q1-1.5*IQR
    BS_Calculado=Q3+1.5*IQR
    ubicacion_outliers=(x<BI_Calculado) | (x>BS_Calculado)
    outliers=x[ubicacion_outliers]
    noutliers=len(outliers)
    print('\n Lista de Estadísticos \n')
    Estadisticos = [['Promedio' , Media],['Mediana' , Mediana], ['Valor Minimo' , Valor_minimo], ['Valor Maximo',Valor_maximo],
                    ['Q1' , Q1],['Q2' , Q2], ['Q3' , Q3], 
                     ['Rango' , rango],['IQR' , IQR],['SD' , sd],['Cv' , cv],['Varianza' , varianza],
                     ['Asimetría', asimetria], ['n Outliers', noutliers] ,['Outliers', outliers]]
    print(tabulate(Estadisticos,
                   headers=['Estadístico' , 'Valor'], 
               tablefmt='fancy_grid',
               stralign='center',
               floatfmt='.0f'))
    return


##### funciones del Recomender systems
def MaxMinSt(base_local,Nuevo_registro):
    base_df=base_local.append(Nuevo_registro)
    base_df
    registro_normalizado=(Nuevo_registro-base_df.min())/(base_df.max()-base_df.min())
    return registro_normalizado

def norm_base(base_df):
    normalizado=(base_df-base_df.min())/(base_df.max()-base_df.min())
    return normalizado



def wcss_(base_df):
    base_norm=norm_base(base_df)
    wcss=[]
    for i in range(1,11):
        kmeans=KMeans(n_clusters=i,max_iter=300)
        kmeans.fit(base_norm) #Aplico K-menas a la base de datos
        wcss.append(kmeans.inertia_)
    #Graficar los resultados de wcss para formar codo de Jambú
    plt.plot(range(1,11),wcss)
    plt.title('Codo de Jambú')
    plt.xlabel('Número de Clusters')
    plt.ylabel('WCSS') #WCSS, es un indicador de que tan similares son los individuos denro de los clusters
    plt.show()



def clusterkmean(base_df,k,rw_norm):
#def clusterkmean(base_df,k):
    base_norm=norm_base(base_df)
    np.random.seed(0)
    clustering=KMeans(n_clusters=k,max_iter=300) #Crear el modelo
    clustering.fit(base_norm)#Aplicar el Modelo
    #base_norm['KMeans_Clusters']=clustering.labels_
    #base_df['KMeans_Clusters']=clustering.labels_
    pca=PCA(n_components=2)
    pca_=pca.fit_transform(base_norm)
    pca_rw_norm=pca.transform(rw_norm)
    pca_df=pd.DataFrame(data=pca_,columns=['Componente_1','Componente_2'])
    #pca_nombres_vinos=pd.concat([pca_vinos_df,base_norm[['KMeans_Clusters']]],axis=1)
    #graficar_puntos_contorno(pca_df)
    pca_df['KMeans_Clusters']=clustering.labels_
    #print(pca_df)
    graficar_puntos_contorno(pca_df)
    fig=plt.figure(figsize=(6,6)) #Tamaño de la figura
    ax=fig.add_subplot(1,1,1) #Se crea un gráfico
    ax.set_xlabel('Componente1',fontsize=15)
    ax.set_ylabel('Componente2',fontsize=15)
    ax.set_title('Componentes Principales',fontsize=20)
    #mejorar con los vectores para que sea versatil
    color_theme=np.array(['blue','green','orange'])
    #color_theme=np.array(list(range(0,k)))
    #color_theme=np.array([1,2,3])
    ax.scatter(x=pca_df.Componente_1,y=pca_df.Componente_2,
    c=color_theme[pca_df.KMeans_Clusters],s=50) #c color de los puntos, s=tamaño de los puntos
    ax.text(pca_rw_norm[0,0],pca_rw_norm[0,1],s='x',fontsize=50,c='red')
    plt.show()
    #3d
    pca=PCA(n_components=3)
    pca_=pca.fit_transform(base_norm)
    pca_rw_norm=pca.transform(rw_norm)
    pca_df=pd.DataFrame(data=pca_,columns=['Componente_1','Componente_2','Componente_3'])
    #pca_nombres_vinos=pd.concat([pca_vinos_df,base_norm[['KMeans_Clusters']]],axis=1)
    pca_df['KMeans_Clusters']=clustering.labels_
    fig = plt.figure()
    ax = Axes3D(fig)
    #color_theme=np.array(list(range(0,k)))
    #color_theme=np.array([1,2,3])
    ax.scatter(pca_df.Componente_1, pca_df.Componente_2, pca_df.Componente_3, marker='o',
               c=color_theme[pca_df.KMeans_Clusters],s=50)
    plt.show()
    centroides=clustering.cluster_centers_
    grupos=clustering.labels_
    #print(pca_vinos_df.head(100))
    return (centroides,grupos)






def membresia(Datos_df,k,Nuevo_dato):
    #new_dato=Nuevo_dato
    base_norm=MaxMinSt(Datos_df,Nuevo_dato)
    #[0] para recuperar los cntroides y [1] para los grupos
    kmean=clusterkmean(Datos_df,k,base_norm)
    centros_df=pd.DataFrame(kmean[0])
    grupos=kmean[1]
    #base_norm=norm_base(Datos_df)
    #Se calcula el grado de pertenencia
    #crea data frame de ceros
    U_df=pd.DataFrame(np.zeros( (base_norm.shape[0], centros_df.shape[0]) ))
    #crea un vector con el número de elementos de cluster
    c=list(range(0,centros_df.shape[0]))
    h=list(range(0,Datos_df.shape[1]))
    #crea un vector con el número de elementos de fila
    registros=list(range(0,base_norm.shape[0]))
    cluster=[]
    for l in registros:
        suma=[]
        for j in c:
            valor=0
            for i in h:
                valor=valor+(base_norm.iloc[l,i]-centros_df.iloc[j,i])**2
                #valor=valor+(Datos_df.iloc[l,i]-centros_df.iloc[j,i])**2
            suma.append(valor)
        for j in c:
            total=0
            for i in c:
                total=(total+suma[j]/suma[i])
                memb=np.divide(1,total)
            U_df.iloc[l,j]=memb
            cluster.append(memb)
    mx=np.max(cluster)
    nc=cluster.index(mx)
    print('Cluster: ',nc)
    print('\n Membresia: \n',U_df)
    return (grupos,nc)






def Distancias(data_filter,gustos):
    registros=list(range(0,data_filter.shape[0]))
    columnas=list(range(0,data_filter.shape[1]))
    distancias_euclidiana=[]
    for j in registros:
        valor=0
        for i in columnas:
            valor=valor+(data_filter.iloc[j,i]-gustos.iloc[0,i])**2
        total=np.sqrt(valor)
        distancias_euclidiana.append(total)
    return(distancias_euclidiana)




def filtrado(datos_df,k,nuevo_dato):
    miembro=membresia(datos_df,k,nuevo_dato)
    grupos=miembro[0]
    cluster=miembro[1]
    variables_cluster=datos_df.copy()
    variables_cluster['KMeans_Clusters']=grupos
    data_filter = variables_cluster[variables_cluster['KMeans_Clusters'] ==cluster]
    datos_filtros=data_filter.drop(['KMeans_Clusters'],axis=1)
    distancias=Distancias(datos_filtros,nuevo_dato)
    datos_filtros['Distancias']=distancias
    top7=datos_filtros.sort_values('Distancias').head(7)
    indices_7=top7.index
    #print('\n Top 7 Barrios Recomendados\n',top7)
    nomenclatura=pd.DataFrame({
    'Nombre Variables':['% Red de Agua Potable y alcantarillado','% Red de electricidad','% Iluminación pública','% Cobertura Recolección Basura','% Cobertura del transporte público','% Cobertura Internet (conectividad)','Distancia a  unidad educativa más cercana(km)','Distancia a centro de cuidado infantil más cercana','Distancia a unidad de salud / hospitalaia más cercana','Distancia a unidad de abastecimiento  PPN más cercana','Distancia a unidad policial más cercana','Distancia a unidad administrativa más cercana','Distancia a parada transporte público mas cercano','Distancia a parque barrial más cercano','Num actividades de acción vecinal barrio / últimos 6 meses','Num Reuniones barrio / últimos 6 meses','Num incidencias reportadas / últimos 6 meses',
                        'Num prácticas tradicionales y festivas del barrio','Num espacios para desarrollo cultural del barrio'],
    'Código':['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19']})
    print('\n Top 7 Barrios Recomendados \n')
    print(top7)
    print(nomenclatura)
    return indices_7

def individuo(base_df,n_index):
    nuevo_dat=base_df.iloc[n_index]
    nomenclatura=pd.DataFrame({
    'Nombre Variables':['% Red de Agua Potable y alcantarillado','% Red de electricidad','% Iluminación pública','% Cobertura Recolección Basura','% Cobertura del transporte público','% Cobertura Internet (conectividad)','Distancia a  unidad educativa más cercana(km)','Distancia a centro de cuidado infantil más cercana','Distancia a unidad de salud / hospitalaia más cercana','Distancia a unidad de abastecimiento  PPN más cercana','Distancia a unidad policial más cercana','Distancia a unidad administrativa más cercana','Distancia a parada transporte público mas cercano','Distancia a parque barrial más cercano','Num actividades de acción vecinal barrio / últimos 6 meses','Num Reuniones barrio / últimos 6 meses','Num incidencias reportadas / últimos 6 meses',
                        'Num prácticas tradicionales y festivas del barrio','Num espacios para desarrollo cultural del barrio'],
    'Código':['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19'],
    'Valor':nuevo_dat
    })
    print(nomenclatura)



def test(base_df,k,gusto,n_index):
    nuevo_dato=pd.DataFrame(gusto.iloc[n_index]).T
    Vector_busqueda_guardado=filtrado(base_df,k,nuevo_dato)
    vector_completo=np.concatenate((n_index,Vector_busqueda_guardado),axis=None)
    guardarBusqueda(vector_completo)
    
def Transformar(df_):
    df_T=df_.copy()
    df_T.iloc[:,	0	]	=	df_T.iloc[:,	0	]	*100/4
    df_T.iloc[:,	1	]	=	df_T.iloc[:,	1	]	*100/4
    df_T.iloc[:,	2	]	=	df_T.iloc[:,	2	]	*80/4
    df_T.iloc[:,	3	]	=	df_T.iloc[:,	3	]	*90/4
    df_T.iloc[:,	4	]	=	df_T.iloc[:,	4	]	*80/4
    df_T.iloc[:,	5	]	=	df_T.iloc[:,	5	]	*100/4
    df_T.iloc[:,	6	]	=	4*0.8/	df_T.iloc[:,	6	]
    df_T.iloc[:,	7	]	=	4*0.3/	df_T.iloc[:,	7	]
    df_T.iloc[:,	8	]	=	4*1/	df_T.iloc[:,	8	]
    df_T.iloc[:,	9	]	=	4*0.3/	df_T.iloc[:,	9	]
    df_T.iloc[:,	10	]	=	4*1/	df_T.iloc[:,	10	]
    df_T.iloc[:,	11	]	=	4*0.3/	df_T.iloc[:,	11	]
    df_T.iloc[:,	12	]	=	4*0.3/	df_T.iloc[:,	12	]
    df_T.iloc[:,	13	]	=	4*0.8/	df_T.iloc[:,	13	]
    df_T.iloc[:,	14	]	=	df_T.iloc[:,	14	]	*50/4
    df_T.iloc[:,	15	]	=	df_T.iloc[:,	15	]	*30/4
    df_T.iloc[:,	16	]	=	df_T.iloc[:,	16	]	*20/4
    df_T.iloc[:,	17	]	=	df_T.iloc[:,	17	]	*50/4
    df_T.iloc[:,	18	]	=	df_T.iloc[:,	18	]	*50/4
    return(df_T)

def guardarBusqueda(Vector_busqueda):
    df=pd.read_csv('D:/1.-UCO-DOCTORADO/python/PROCESAMIENTO DE DATOS CON PYTHON/busquedas.csv',engine='python',sep=';')
    fila=df.shape[0]
    df.loc[fila]=Vector_busqueda
    df.to_csv('D:/1.-UCO-DOCTORADO/python/PROCESAMIENTO DE DATOS CON PYTHON/busquedas.csv',index=False,sep=';')
    return

def soloBusqueda(index_user,df_user,barrios):
    nuevo_dato=pd.DataFrame(df_user.iloc[index_user]).T
    df_user2=df_user.copy()
    distancias=Distancias(df_user2,nuevo_dato)
    df_user2['Distancias']=distancias
    top7=df_user2.sort_values('Distancias').head(7)
    print('Usuarios parecidos')
    print(top7)
    top7.index
    Vector_busqueda=top7.index
    #casa=pd.read_csv('D:/1.-UCO-DOCTORADO/python/PROCESAMIENTO DE DATOS CON PYTHON/datos_simulados_casas.csv',
                  #engine='python',sep=',')
    #si los usuarios semesjates tienene busqueda
    df=pd.read_csv('D:/1.-UCO-DOCTORADO/python/PROCESAMIENTO DE DATOS CON PYTHON/busquedas.csv',engine='python',sep=';')
    n=df[df.Usuario.isin(Vector_busqueda)].shape[0]
    busqueda=df[df.Usuario.isin(Vector_busqueda)]
    n_base=[n,busqueda]
    n_base[0]
    #Condicioón principal de la busqueda
    if n>0:
       print('Barrios de usuarios')
       print(busqueda)
    else:
        print('Buscar transformar y aplicar algoritmo')
        #individuo(gusto,6)
        gusto=Transformar(df_user)
        test(barrios,3,gusto,index_user)
    return



def estimador_nan (df_nan,df,v_resp):
    X=df.drop([v_resp],axis=1)
    Y=df[v_resp].round(0)
    X_train, X_test,Y_train, Y_test = train_test_split(X,Y,random_state=0,test_size = 0.70)
    #X_test,Y_test=X_train,Y_train
    #rdf
    print("Random forest")
    clf_4 = RandomForestClassifier(n_estimators=10)
    clf_4 = clf_4.fit(X_train,Y_train)
    print(clf_4.score(X_test,Y_test)*100)
    #svm
    print("SVM")
    clf=svm.SVC(kernel='linear',C=1).fit(X_train,Y_train)
    classifier_preddictions=clf.predict(X_test)
    print(accuracy_score(Y_test,classifier_preddictions)*100)
    #nnet
    print("NNET")
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5,2), random_state=1)
    clf.fit(X_train,Y_train)
    print(clf.score(X_test,Y_test)*100)
    #tree
    print("Tree")
    dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, Y_train) 
    print(dtree_model.score(X_test,Y_test)*100)
    #nvb
    print("NaviBayes")
    gnb = GaussianNB().fit(X_train, Y_train)
    # accuracy on X_test 
    accuracy = gnb.score(X_test, Y_test)
    print (accuracy*100 )
    
    #Estimaciones de los valores perdidos
    print("Estimaciones de los valores perdidos")
    X_=df.drop([v_resp],axis=1)
    v_estimados=clf_4.predict(X_)
    print("Valores estimados")
    print(v_estimados)
    bol=df_nan[v_resp].isnull()
    dt3=df[v_resp] 
    #Los datos a predecir
    indices=dt3[bol].index
    v_a_imputar=v_estimados[indices]
    return v_a_imputar

def dt_nan(df_,i):
    #Analizar los valores perdidos
    plt.figure(figsize=(20,10))
    sns.heatmap(df_,cmap="RdYlGn_r",annot=False)
    plt.show()
    print("Valores perdidos por columna: ")
    print(df_.isnull().sum())
    missing_values_count = df_.isnull().sum()
    total_cells = np.product(df_.shape)
    total_missing = missing_values_count.sum()
    print("Total de Valores perdidos: ")
    print(total_missing)
    print("% de Valores perdidos: ")
    print(total_missing/total_cells*100)
    if (i==1):
        print("Elimino los Valores perdidos: ")
        df_imputada=df_.dropna()
    elif (i==2):
        print("Imputo con los Valores de la fila anterior: ")
        df_imputada=df_.fillna(method = 'bfill', axis=0).fillna(0)
    else:
        print("Imputo con la media de la variable: ")
        df_imputada=df_.copy()
        df_imputada.fillna(df_.mean(), inplace=True)
    plt.figure(figsize=(20,10))
    sns.heatmap(df_imputada,cmap="RdYlGn_r",annot=False)
    plt.show()
    return df_imputada

def mt_corr_psk (df):
    plt.figure(figsize=(20,10))
    print("Matriz de correlación pearson")
    c= df.corr()
    print(c)
    print("Matriz de correlación spearman")
    c=df.corr(method='spearman')
    print(c)
    print("Matriz de correlación kendall")
    c=df.corr(method='kendall')
    print(c)
    sns.heatmap(c,cmap="BrBG",annot=True)
    #sns.heatmap(c,cmap="RdYlGn_r",annot=True)
    


#Función para el calculo de membresia
def membresia_(Datos_df,centros_df):
    #Se calcula el grado de pertenencia
    #crea data frame de ceros
    U_df=pd.DataFrame(np.zeros( (Datos_df.shape[0], centros_df.shape[0]) ))
    #crea un vector con el número de elementos de cluster
    c=list(range(0,centros_df.shape[0]))
    h=list(range(0,Datos_df.shape[1]))
    #crea un vector con el número de elementos de fila
    registros=list(range(0,Datos_df.shape[0]))
    cluster=[]
    for l in registros:
        suma=[]
        for j in c:
            valor=0
            for i in h:
                valor=valor+(Datos_df.iloc[l,i]-centros_df.iloc[j,i])**2
            suma.append(valor)
        for j in c:
            total=0
            for i in c:
                total=(total+suma[j]/suma[i])
                memb=np.divide(1,total)
            U_df.iloc[l,j]=memb
            cluster.append(memb)
    mx=np.max(cluster)
    nc=cluster.index(mx)
    #print('Cluster: ',nc+1)
    #print('\n Membresia: \n',U_df)
    return U_df

#Función para el calculo de membresia
def membresiaCLU(Datos_df,centros_df):
    #Se calcula el grado de pertenencia
    #crea data frame de ceros
    U_df=pd.DataFrame(np.zeros( (Datos_df.shape[0], centros_df.shape[0]) ))
    #crea un vector con el número de elementos de cluster
    c=list(range(0,centros_df.shape[0]))
    h=list(range(0,Datos_df.shape[1]))
    #crea un vector con el número de elementos de fila
    registros=list(range(0,Datos_df.shape[0]))
    cluster=[]
    for l in registros:
        suma=[]
        for j in c:
            valor=0
            for i in h:
                valor=valor+(Datos_df.iloc[l,i]-centros_df.iloc[j,i])**2
            suma.append(valor)
        for j in c:
            total=0
            for i in c:
                total=(total+suma[j]/suma[i])
                memb=np.divide(1,total)
            U_df.iloc[l,j]=memb
            cluster.append(memb)
    mx=np.max(cluster)
    nc=cluster.index(mx)
    #print('Cluster: ',nc+1)
    #print('\n Membresia: \n',U_df)
    return nc


#Funcipon para normalizar la base cambiar escala
def norm_base(base_df):
    normalizado=(base_df-base_df.min())/(base_df.max()-base_df.min())
    return normalizado
#Funcipon para desnormalizar la base cambiar escala
def desnormalizar(base_df,estimaciones):
    desnormalizar=estimaciones*(base_df.max()-base_df.min())+base_df.min()
    return desnormalizar

#Predictor fuzzy cmean nnet
def pridictor_fuzzy_cmean_nnet(df_full,x,filas,umbral):
    if (x =='v1'or x =='v2'or x =='v3'or x =='v4'or x =='v5'or x =='v6'):
        df=df_full.iloc[:,0:6]
        print('Disponibilidad de servicios básicos')
        categoria='Disponibilidad de servicios básicos'
    elif ((x =='v7'or x =='v8'or x =='v9'or x =='v10'or x =='v11'or x =='v12'or x =='v13'or x =='v14')):
        print('Ubicación')
        categoria='Ubicación'
        df=df_full.iloc[:,6:14]
    elif ((x =='v15'or x =='v16'or x =='v17')):    
        df=df_full.iloc[:,14:17]
        print('cohesión social')
        categoria='cohesión social'
    elif ((x =='v18'or x =='v19')): 
        df=df_full.iloc[:,17:19]    
        print('Pertinencia cultural')
        categoria='Pertinencia cultural'
    #df=seleccionar_base_var(df_full,'v2')    
    clustering=KMeans(n_clusters=3,max_iter=300)
    clustering.fit(df)#Aplicar el Modelo
    #print('\n')
    #print('Categoria')
    #print(categoria)
    df['KMeans_Clusters']=clustering.labels_
    #print('Base_clase')
    #print(df)
    #clase=df.iloc[fila,-1]
    #CLase 0
    datos_filt=df[df['KMeans_Clusters'] ==0]
    #datos_filt
    #print('Base_clase_filtro')
    #print(datos_filt)
    datos_filt1=datos_filt.drop(['KMeans_Clusters'],axis=1)
    #print('Base_clase_filtro prueba')
    #print(datos_filt1)
    centroides=pd.DataFrame(clustering.cluster_centers_)
    #print('Centroides_prueba')
    #print(centroides)
    registros=list(range(0,datos_filt1.shape[0]))
    #print('registros prueba')
    #print(registros)
    mbr=[]
    for i in registros:
        menrbe=membresia_(pd.DataFrame(datos_filt1.iloc[i]).T,centroides).iloc[0,0]
        mbr.append(menrbe)
    datos_filt1['Membreship']=mbr
    #print('Base_clase_filtro_Membresia')
    #print(datos_filt1)
    datos_con_umbral=datos_filt1[datos_filt1['Membreship'] >umbral]
    #Prredicción
    #print('Base_clase_filtro_umbral')
    #print(datos_con_umbral)
    base_norm=norm_base(datos_con_umbral)
    X=base_norm.drop([x,'Membreship'],axis=1)
    Y=base_norm[x]
    #print('Base_normalizada')
    #print(base_norm)
    #print('X')
    #print(X)
    #print('Y')
    #print(Y)
        #X_train, X_test,Y_train, Y_test = train_test_split(X,Y,random_state=0,test_size = 0.70)
    clf0 = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(16), random_state=1)
    clf0.fit(X,Y)
    print("Poder de predicción de la red neuronal clase 0:")
    print(clf0.score(X,Y)*100)
    v_estimados=clf0.predict(X)
    #plt.plot(v_estimados,Y, 'ro')
    #plt.show()
    #CLase 1
    datos_filt=df[df['KMeans_Clusters'] ==1]
    #datos_filt
    #print('Base_clase_filtro')
    #print(datos_filt)
    datos_filt1=datos_filt.drop(['KMeans_Clusters'],axis=1)
    #print('Base_clase_filtro_prueba_clse1')
    #print(datos_filt1)
    centroides=pd.DataFrame(clustering.cluster_centers_)
    registros=list(range(0,datos_filt1.shape[0]))
    mbr=[]
    for i in registros:
        menrbe=membresia_(pd.DataFrame(datos_filt1.iloc[i]).T,centroides).iloc[0,1]
        mbr.append(menrbe)
    datos_filt1['Membreship']=mbr
    #print('Base_clase_filtro_Membresia')
    #print(datos_filt1)
    datos_con_umbral=datos_filt1[datos_filt1['Membreship'] >umbral]
    #Prredicción
    #print('Base_clase_filtro_umbral')
    #print(datos_con_umbral)
    base_norm=norm_base(datos_con_umbral)
    X=base_norm.drop([x,'Membreship'],axis=1)
    Y=base_norm[x]
    #print('Base_normalizada')
    #print(base_norm)
    #print('X')
    #print(X)
    #print('Y')
    #print(Y)
        #X_train, X_test,Y_train, Y_test = train_test_split(X,Y,random_state=0,test_size = 0.70)
    clf1 = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(16), random_state=1)
    clf1.fit(X,Y)
    print("Poder de predicción de la red neuronal  clase 1:")
    print(clf1.score(X,Y)*100)
    v_estimados=clf1.predict(X)
    #plt.plot(v_estimados,Y, 'ro')
    #plt.show()
    #CLase 2  
    datos_filt=df[df['KMeans_Clusters'] ==2]
    #datos_filt
    #print('Base_clase_filtro')
    #print(datos_filt)
    datos_filt1=datos_filt.drop(['KMeans_Clusters'],axis=1)
    centroides=pd.DataFrame(clustering.cluster_centers_)
    registros=list(range(0,datos_filt1.shape[0]))
    mbr=[]
    for i in registros:
        menrbe=membresia_(pd.DataFrame(datos_filt1.iloc[i]).T,centroides).iloc[0,2]
        mbr.append(menrbe)
    datos_filt1['Membreship']=mbr
    #print('Base_clase_filtro_Membresia')
    #print(datos_filt1)
    datos_con_umbral=datos_filt1[datos_filt1['Membreship'] >umbral]
    #Prredicción
    #print('Base_clase_filtro_umbral')
    #print(datos_con_umbral)
    base_norm=norm_base(datos_con_umbral)
    X=base_norm.drop([x,'Membreship'],axis=1)
    Y=base_norm[x]
    #print('Base_normalizada')
    #print(base_norm)
    #print('X')
    #print(X)
    #print('Y')
    #print(Y)
        #X_train, X_test,Y_train, Y_test = train_test_split(X,Y,random_state=0,test_size = 0.70)
    clf2 = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(16), random_state=1)
    clf2.fit(X,Y)
    print("Poder de predicción de la red neuronal  clase 2:")
    print(clf2.score(X,Y)*100)
    v_estimados=clf2.predict(X)
    #plt.plot(v_estimados,Y, 'ro')
    #plt.show()
    #Regresión lineal
    regr = linear_model.LinearRegression()
    # Entrenamos nuestro modelo
    regr.fit(X, Y)
    #print("Poder de predicción una regresión:")
    #print(regr.score(X, Y)*100)
    # Hacemos las predicciones que en definitiva una línea (en este caso, al ser 2D)
    #y_pred = regr.predict(X)
    estimadores=desnormalizar(df_full[x],v_estimados)
    
    #print('matriz de menbresia')
    #print(df)
    df_=df.drop(['KMeans_Clusters'],axis=1)
    #print('matriz de pura prueba')
    #print(df_)
    valores_a_imputar=[]
    for i in range(len(filas)):
        #print(filas[i])
        menrbe=membresiaCLU(pd.DataFrame(df_.iloc[filas[i]]).T,centroides)
        #print("MEMBRESIA")
        #print(menrbe)
        
        base_norm1=norm_base(df_)
        X1=pd.DataFrame(base_norm1.iloc[filas[i]]).T
        #print('X1')
        #print(X1)
        X2=X1.drop([x],axis=1)
        #0,1,2
        #print("matriz X2")
        #print(X2)
        #membresiaCLU()
        if (menrbe == 0):
            #cluster 0
            #print('modelo 1')
            valor_a_imputar=clf0.predict(X2)
        elif (menrbe == 1):
            #cluster 1
            #print('modelo 2')
            valor_a_imputar=clf1.predict(X2)
        elif (menrbe == 2):
            #print('modelo 3')
            #cluster 2
            valor_a_imputar=clf2.predict(X2)
        valor_a_imputar2=desnormalizar(df_full[x],valor_a_imputar)
        #print('Valor estimado') 
        #print(valor_a_imputar2)
        valores_a_imputar.append(valor_a_imputar2)
    return valores_a_imputar#clf0,clf1,clf2,centroides




def imputar_valores(dt_p,df_full_imputada):
    #X.isnull().any()
    XX=dt_p.copy()
    Columnas_na=XX.columns[XX.isnull().any()]
    h=list(range(0,len(Columnas_na)))
    for i in h:
        #print(Columnas_na[i])
        bol=XX[Columnas_na[i]].isnull()
        indices=XX.v4[bol].index
        #print('indices')
        #print(indices)
        valores_estiamdo=pridictor_fuzzy_cmean_nnet(df_full_imputada,x=Columnas_na[i],filas=indices,umbral=0.64)
        #print('Estimaciones')
        k=list(range(0,len(valores_estiamdo)))
        for j in k:
            XX[Columnas_na[i]][indices[j]]=valores_estiamdo[j]
    return XX

def imputar_valores_fachada(dt_p):
    #X.isnull().any()
    df_full_imputada=dt_nan(dt_p,3)
    XX=dt_p.copy()
    Columnas_na=XX.columns[XX.isnull().any()]
    h=list(range(0,len(Columnas_na)))
    for i in h:
        #print(Columnas_na[i])
        bol=XX[Columnas_na[i]].isnull()
        indices=XX.v4[bol].index
        #print('indices')
        #print(indices)
        valores_estiamdo=pridictor_fuzzy_cmean_nnet(df_full_imputada,x=Columnas_na[i],filas=indices,umbral=0.64)
        #print('Estimaciones')
        k=list(range(0,len(valores_estiamdo)))
        for j in k:
            XX[Columnas_na[i]][indices[j]]=valores_estiamdo[j]
    return XX


def graficar_puntos_contorno(xy_df):
    #xy_df = np.array(xy_df)
    dt_0=xy_df[xy_df['KMeans_Clusters'] ==0]
    #print(dt_0)
    dt_0=dt_0.drop(['KMeans_Clusters'],axis=1)
    
    #print(dt_0)
    dt_1=xy_df[xy_df['KMeans_Clusters'] ==0]
    dt_1=dt_1.drop(['KMeans_Clusters'],axis=1)
    #print(dt_1)
    dt_2=xy_df[xy_df['KMeans_Clusters'] ==0]
    dt_2=dt_2.drop(['KMeans_Clusters'],axis=1)
    #print(dt_2)
    
    
    xy_df = np.array(xy_df)
    xy0_df=np.array(dt_0)
    xy1_df=np.array(dt_1)
    xy2_df=np.array(dt_2)
    
    
    pc1=xy_df[:,0]
    pc2=xy_df[:,1]
    m1 = random.choice(xy0_df)
    m2 = random.choice(xy1_df)
    m3 = random.choice(xy2_df)
    cov1 = np.cov(np.transpose(xy0_df))
    cov2 = np.cov(np.transpose(xy1_df))
    cov3 = np.cov(np.transpose(xy2_df))
    x1 = np.linspace(pc1.min(),pc1.max(),pc1.shape[0])  
    x2 = np.linspace(pc1.min(),pc1.max(),pc1.shape[0])
    X, Y = np.meshgrid(x1,x2) 
    Z1 = multivariate_normal(m1, cov1)  
    Z2 = multivariate_normal(m2, cov2)
    Z3 = multivariate_normal(m3, cov3)
    pos = np.empty(X.shape + (2,))                # a new array of given shape and type, without initializing entries
    pos[:, :, 0] = X; pos[:, :, 1] = Y   
    plt.figure(figsize=(10,10))                                                          # creating the figure and assigning the size
    plt.scatter(pc1, pc2, marker='o')     
    plt.contour(X, Y, Z1.pdf(pos), colors="r" ,alpha = 0.7) 
    plt.contour(X, Y, Z2.pdf(pos), colors="b" ,alpha = 0.7) 
    plt.contour(X, Y, Z3.pdf(pos), colors="g" ,alpha = 0.7) 
    plt.axis('equal')                                                                  # making both the axis equal
    plt.xlabel('Componente 1', fontsize=16)                                                  # X-Axis
    plt.ylabel('Componente 2', fontsize=16)                                                  # Y-Axis
    plt.title('Initial Random Clusters', fontsize=22)                                            # Title of the plot
    plt.grid()                                                                         # displaying gridlines
    plt.show()


def filtroJSon(datos_df,k,nuevo_dato):
    miembro=membresiaJson(datos_df,k,nuevo_dato)
    grupos=miembro[0]
    cluster=miembro[1]
    variables_cluster=datos_df.copy()
    variables_cluster['KMeans_Clusters']=grupos
    data_filter = variables_cluster[variables_cluster['KMeans_Clusters'] ==cluster]
    datos_filtros=data_filter.drop(['KMeans_Clusters'],axis=1)
    distancias=Distancias(datos_filtros,nuevo_dato)
    datos_filtros['Distancias']=distancias
    top7=datos_filtros.sort_values('Distancias').head(7)
    indices_7=top7.index
    #print('\n Top 7 Barrios Recomendados\n',top7)
    nomenclatura=pd.DataFrame({
    'Nombre Variables':['% Red de Agua Potable y alcantarillado','% Red de electricidad','% Iluminación pública','% Cobertura Recolección Basura','% Cobertura del transporte público','% Cobertura Internet (conectividad)','Distancia a  unidad educativa más cercana(km)','Distancia a centro de cuidado infantil más cercana','Distancia a unidad de salud / hospitalaia más cercana','Distancia a unidad de abastecimiento  PPN más cercana','Distancia a unidad policial más cercana','Distancia a unidad administrativa más cercana','Distancia a parada transporte público mas cercano','Distancia a parque barrial más cercano','Num actividades de acción vecinal barrio / últimos 6 meses','Num Reuniones barrio / últimos 6 meses','Num incidencias reportadas / últimos 6 meses',
                        'Num prácticas tradicionales y festivas del barrio','Num espacios para desarrollo cultural del barrio'],
    'Código':['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19']})
    print('\n Top 7 Barrios Recomendados \n')
    print(top7)
    print(nomenclatura)
    return top7

def testJson(base_df,k,gusto,n_index):
    nuevo_dato=pd.DataFrame(gusto.iloc[n_index]).T
    Top7=filtroJSon(base_df,k,nuevo_dato)
    return Top7
    #vector_completo=np.concatenate((n_index,Vector_busqueda_guardado),axis=None)
    #guardarBusqueda(vector_completo)

def individuoJson(base_df,n_index):
    nuevo_dat=base_df.iloc[n_index]
    nomenclatura=pd.DataFrame({
    'Nombre Variables':['% Red de Agua Potable y alcantarillado','% Red de electricidad','% Iluminación pública','% Cobertura Recolección Basura','% Cobertura del transporte público','% Cobertura Internet (conectividad)','Distancia a  unidad educativa más cercana(km)','Distancia a centro de cuidado infantil más cercana','Distancia a unidad de salud / hospitalaia más cercana','Distancia a unidad de abastecimiento  PPN más cercana','Distancia a unidad policial más cercana','Distancia a unidad administrativa más cercana','Distancia a parada transporte público mas cercano','Distancia a parque barrial más cercano','Num actividades de acción vecinal barrio / últimos 6 meses','Num Reuniones barrio / últimos 6 meses','Num incidencias reportadas / últimos 6 meses',
                        'Num prácticas tradicionales y festivas del barrio','Num espacios para desarrollo cultural del barrio'],
    'Código':['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19'],
    'Valor':nuevo_dat
    })
    return nomenclatura

def membresiaJson(Datos_df,k,Nuevo_dato):
    #new_dato=Nuevo_dato
    base_norm=MaxMinSt(Datos_df,Nuevo_dato)
    #[0] para recuperar los cntroides y [1] para los grupos
    kmean=clusterkmeanJSon(Datos_df,k,base_norm)
    centros_df=pd.DataFrame(kmean[0])
    grupos=kmean[1]
    #base_norm=norm_base(Datos_df)
    #Se calcula el grado de pertenencia
    #crea data frame de ceros
    U_df=pd.DataFrame(np.zeros( (base_norm.shape[0], centros_df.shape[0]) ))
    #crea un vector con el número de elementos de cluster
    c=list(range(0,centros_df.shape[0]))
    h=list(range(0,Datos_df.shape[1]))
    #crea un vector con el número de elementos de fila
    registros=list(range(0,base_norm.shape[0]))
    cluster=[]
    for l in registros:
        suma=[]
        for j in c:
            valor=0
            for i in h:
                valor=valor+(base_norm.iloc[l,i]-centros_df.iloc[j,i])**2
                #valor=valor+(Datos_df.iloc[l,i]-centros_df.iloc[j,i])**2
            suma.append(valor)
        for j in c:
            total=0
            for i in c:
                total=(total+suma[j]/suma[i])
                memb=np.divide(1,total)
            U_df.iloc[l,j]=memb
            cluster.append(memb)
    mx=np.max(cluster)
    nc=cluster.index(mx)
    print('Cluster: ',nc)
    print('\n Membresia: \n',U_df)
    return (grupos,nc)

def clusterkmeanJSon(base_df,k,rw_norm):
#def clusterkmean(base_df,k):
    base_norm=norm_base(base_df)
    np.random.seed(0)
    clustering=KMeans(n_clusters=k,max_iter=300) #Crear el modelo
    clustering.fit(base_norm)#Aplicar el Modelo
    #base_norm['KMeans_Clusters']=clustering.labels_
    #base_df['KMeans_Clusters']=clustering.labels_
    pca=PCA(n_components=2)
    pca_=pca.fit_transform(base_norm)
    pca_rw_norm=pca.transform(rw_norm)
    pca_df=pd.DataFrame(data=pca_,columns=['Componente_1','Componente_2'])
    #pca_nombres_vinos=pd.concat([pca_vinos_df,base_norm[['KMeans_Clusters']]],axis=1)
    #graficar_puntos_contorno(pca_df)
    pca_df['KMeans_Clusters']=clustering.labels_
    #print(pca_df)
    #graficar_puntos_contorno(pca_df)
    #fig=plt.figure(figsize=(6,6)) #Tamaño de la figura
    #ax=fig.add_subplot(1,1,1) #Se crea un gráfico
    #ax.set_xlabel('Componente1',fontsize=15)
    #ax.set_ylabel('Componente2',fontsize=15)
    #ax.set_title('Componentes Principales',fontsize=20)
    #mejorar con los vectores para que sea versatil
    #color_theme=np.array(['blue','green','orange'])
    #color_theme=np.array(list(range(0,k)))
    #color_theme=np.array([1,2,3])
    #ax.scatter(x=pca_df.Componente_1,y=pca_df.Componente_2,
    #c=color_theme[pca_df.KMeans_Clusters],s=50) #c color de los puntos, s=tamaño de los puntos
    #ax.text(pca_rw_norm[0,0],pca_rw_norm[0,1],s='x',fontsize=50,c='red')
    #plt.show()
    #3d
    #pca=PCA(n_components=3)
   # pca_=pca.fit_transform(base_norm)
    #pca_rw_norm=pca.transform(rw_norm)
    #pca_df=pd.DataFrame(data=pca_,columns=['Componente_1','Componente_2','Componente_3'])
    #pca_nombres_vinos=pd.concat([pca_vinos_df,base_norm[['KMeans_Clusters']]],axis=1)
    pca_df['KMeans_Clusters']=clustering.labels_
    #fig = plt.figure()
    #ax = Axes3D(fig)
    #color_theme=np.array(list(range(0,k)))
    #color_theme=np.array([1,2,3])
    ##ax.scatter(pca_df.Componente_1, pca_df.Componente_2, pca_df.Componente_3, marker='o',
   #            c=color_theme[pca_df.KMeans_Clusters],s=50)
    #plt.show()
    centroides=clustering.cluster_centers_
    grupos=clustering.labels_
    #print(pca_vinos_df.head(100))
    return (centroides,grupos)

