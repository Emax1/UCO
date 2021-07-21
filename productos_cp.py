import pandas as pd
import json
from FUNCIONES import test,Transformar,individuo,testJson,individuoJson
#Barrios=pd.read_csv('D:/1.-UCO-DOCTORADO/python/PROCESAMIENTO DE DATOS CON PYTHON/datos_simulados_2.csv',engine='python',sep=';')
#df=pd.read_csv('D:/1.-UCO-DOCTORADO/python/PROCESAMIENTO DE DATOS CON PYTHON/datos_simulados_gustos.csv',engine='python',sep=',')

ubi_Barrios='https://raw.githubusercontent.com/Emax1/UCO/main/datos_simulados_2.csv'
ubi_gusto='https://raw.githubusercontent.com/Emax1/UCO/main/datos_simulados_gustos.csv'

Barrios=pd.read_csv(ubi_Barrios,engine='python',sep=',')
df=pd.read_csv(ubi_gusto,engine='python',sep=',')


gusto=Transformar(df)

datos_user=Barrios.iloc[78]
#data_filter = Barrios[Barrios['v1'] > 99]
#data_filter

def UNO_SOLO(dt):
    datos_user=Barrios.iloc[dt]
    datos_user=datos_user.to_json(orient="split")
    datos_user=json.loads(datos_user)
    return datos_user

Barrios2=Barrios.to_json()
datos_user2=datos_user.to_json()
#data_filter2=data_filter.to_json(orient="split")
#data_filter2=json.loads(data_filter2)


productos={
      'col1':[10,20,30],
      'col2':['A','B','C']
      }

productos2=[
    {'name':'laptop','price':800,'quantity':4},
    {'name':'laptop','price':800,'quantity':4},
    {'name':'laptop','price':800,'quantity':4},
    {'name':'laptop','price':800,'quantity':4}
]
d = {"sitio": "Recursos Python", "url": "recursospython.com"}

#individuo(Barrios,78)
#individuo(gusto,78)

#dt1=test(Barrios,3,gusto,78)

#test(Barrios,3,gusto,20)

def recomenderJSon(dt):
    datos_barrios=testJson(Barrios,3,gusto,dt)
    datos_barrios=datos_barrios.to_json(orient="split")
    datos_barrios=json.loads(datos_barrios)
    return datos_barrios

def verJSon(dt):
    datos_barrios=individuoJson(gusto,dt)
    datos_barrios=datos_barrios.to_json(orient="split")
    datos_barrios=json.loads(datos_barrios)
    return datos_barrios

def verBarrioJSon(dt):
    datos_barrios=individuoJson(Barrios,dt)
    datos_barrios=datos_barrios.to_json(orient="split")
    datos_barrios=json.loads(datos_barrios)
    return datos_barrios

#individuoJson(Barrios,444)
#gusto.info()
#dfg=recomenderJSon(78)
#dfg=testJson(Barrios,3,gusto,23)
