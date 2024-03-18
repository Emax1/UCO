# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 20:56:06 2024

@author: UTM
"""
import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.common.action_chains import ActionChains
from pywinauto.application import Application
import time
import random
from collections import Counter

# Genera un número aleatorio en el rango de 1 a 184



def ACCION_LOCALIZACION(localizacion,driver):
    if localizacion == 'Ramón de Carranza - Madre Rafols':
        numero = 1
    elif localizacion == 'Los Remedios':
        numero = 2
    elif localizacion == 'Plaza de Cuba - República Argentina':
        numero = 3
    elif localizacion == 'Asunción - Adolfo Suárez':
        numero = 4
    elif localizacion == 'Parque de los Principes - Calle Niebla':
        numero = 5
    elif localizacion == 'Tablada':
        numero = 6
    elif localizacion == 'Blas Infante':
        numero = 7    
    id_elemento="//*[@id='localización']/option["+str(numero)+"]"
    #return id_elemento
    driver.find_element(By.XPATH,id_elemento).click()
    

def ACCION_ASCENSOR(var1,driver):
    if var1 == 1:
        numero = 2
    elif var1 == 0:
        numero = 1
    id_elemento="//*[@id='ascensor']/option["+str(numero)+"]"
    #return id_elemento
    driver.find_element(By.XPATH,id_elemento).click()
    


def ACCION_BANOS(var1,driver):
    #return id_elemento
    numero=var1-1
    slideFrom = driver.find_element(By.XPATH,"/html/body/bslib-layout-columns/div[1]/div/div/bslib-layout-columns/div[1]/div/div/div[3]/span/span[3]")#.click()
    # Mover el cursor del mouse a las coordenadas calculadas y soltar el clic
    actions = ActionChains(driver)
    actions.move_to_element(slideFrom).click_and_hold()
    actions.move_by_offset(numero*100, 0).release().perform()


def ACCION_TRASTERO(var1,driver):
    if var1 == 1:
        numero = 2
    elif var1 == 0:
        numero = 1
    id_elemento="//*[@id='trastero']/option["+str(numero)+"]"
    #return id_elemento
    driver.find_element(By.XPATH,id_elemento).click()
    


def ACCION_PISO(var1,driver):
    if var1 == 'Bajo':
        numero = 1
    elif var1 == 'Muchas_plantas':
        numero = 2
    elif var1 == 'Primeros_pisos':
        numero = 3
    elif var1 == 'Ultimos_pisos':
        numero = 4
 
    id_elemento="//*[@id='piso']/option["+str(numero)+"]"
    #return id_elemento
    driver.find_element(By.XPATH,id_elemento).click()
    


def ACCION_HABITACIONES(var1,driver):
    #return id_elemento
    numero=var1-1
    slideFrom = driver.find_element(By.XPATH,"/html/body/bslib-layout-columns/div[1]/div/div/bslib-layout-columns/div[1]/div/div/div[6]/span/span[1]/span[6]")#.click()
    # Mover el cursor del mouse a las coordenadas calculadas y soltar el clic
    actions = ActionChains(driver)
    actions.move_to_element(slideFrom).click_and_hold()
    actions.move_by_offset(numero*100/2, 0).release().perform()
    # para obtener la posisicon a cuanto le toca cada valor 300/(n-1), en este caso 300/(7-1)=50
    #0 50 100 150 200 250 300
    #0*100/2
    #1*100/2
    #2*100/2
    #3*100/2
    #4*100/2
    #5*100/2
    #6*100/2


def ACCION_METROS2(var1,driver):
    ##### LLENAR UN CUADRO DE TEXTO
    cuadro_texto =driver.find_element(By.ID,"metros_cuadrados")#.click()

    # Limpiar el cuadro de texto (opcional)
    cuadro_texto.clear()

    # Escribir texto en el cuadro de texto
    texto_a_ingresar = int(var1)
    cuadro_texto.send_keys(texto_a_ingresar)



def ACCION_CONDICION(var1,driver):
    if var1 == 'Segunda mano/buen estado':
        numero = 1
    elif var1 == 'Segunda mano/para reformar':
        numero = 2
    elif var1 == 'Promoción de obra nueva':
        numero = 3
    id_elemento="//*[@id='condicion']/option["+str(numero)+"]"
    #return id_elemento
    driver.find_element(By.XPATH,id_elemento).click()
    


def ACCION_ARMARIOS(var1,driver):
    if var1 == 1:
        numero = 2
    elif var1 == 0:
        numero = 1
    id_elemento="//*[@id='armarios_empotrados']/option["+str(numero)+"]"
    #return id_elemento
    driver.find_element(By.XPATH,id_elemento).click()


def ACCION_TERRAZA(var1,driver):
    if var1 == 1:
        numero = 2
    elif var1 == 0:
        numero = 1
    id_elemento="//*[@id='terraza']/option["+str(numero)+"]"
    #return id_elemento
    driver.find_element(By.XPATH,id_elemento).click()


def ACCION_BALCON(var1,driver):
    if var1 == 1:
        numero = 2
    elif var1 == 0:
        numero = 1
    id_elemento="//*[@id='balcon']/option["+str(numero)+"]"
    #return id_elemento
    driver.find_element(By.XPATH,id_elemento).click()



def ACCION_GARAJE(var1,driver):
    if var1 == 1:
        numero = 2
    elif var1 == 0:
        numero = 1
    id_elemento="//*[@id='garaje']/option["+str(numero)+"]"
    #return id_elemento
    driver.find_element(By.XPATH,id_elemento).click()


def ACCION_CALEFACCION(var1,driver):
    if var1 == 1:
        numero = 2
    elif var1 == 0:
        numero = 1
    id_elemento="//*[@id='calefaccion']/option["+str(numero)+"]"
    #return id_elemento
    driver.find_element(By.XPATH,id_elemento).click()



def ACCION_AIRE_ACONDICIONADO(var1,driver):
    if var1 == 1:
        numero = 2
    elif var1 == 0:
        numero = 1
    id_elemento="//*[@id='aire']/option["+str(numero)+"]"
    #return id_elemento
    driver.find_element(By.XPATH,id_elemento).click()


def SELECCION_CARACTERISTICA(df_casas,driver):
    numero_aleatorio = random.randint(1, len(df_casas))
    localizacion=df_casas.iloc[numero_aleatorio]['localizacion']
    precio=df_casas.iloc[numero_aleatorio]['precio']
    ascensor=df_casas.iloc[numero_aleatorio]['ascensor']
    banos=df_casas.iloc[numero_aleatorio]['baños']
    trastero=df_casas.iloc[numero_aleatorio]['trastero']
    piso=df_casas.iloc[numero_aleatorio]['piso']
    habitaciones=df_casas.iloc[numero_aleatorio]['habitaciones']
    metros_reales=df_casas.iloc[numero_aleatorio]['metros_reales']
    condicion=df_casas.iloc[numero_aleatorio]['condicion']
    armarios_empotrados=df_casas.iloc[numero_aleatorio]['armarios_empotrados']
    terraza=df_casas.iloc[numero_aleatorio]['terraza']
    balcon=df_casas.iloc[numero_aleatorio]['balcon']
    garaje=df_casas.iloc[numero_aleatorio]['garaje']
    calefaccion=df_casas.iloc[numero_aleatorio]['calefaccion']
    aire_acondicionado=df_casas.iloc[numero_aleatorio]['aire_acondicionado']
    #numero=[]
    ACCION_LOCALIZACION(localizacion,driver)
    ACCION_ASCENSOR(ascensor,driver)
    ACCION_BANOS(banos,driver)
    ACCION_TRASTERO(trastero,driver)
    ACCION_PISO(piso,driver)
    ACCION_HABITACIONES(habitaciones,driver)
    ACCION_METROS2(metros_reales,driver)
    ACCION_CONDICION(condicion,driver)
    ACCION_ARMARIOS(armarios_empotrados,driver)
    ACCION_TERRAZA(terraza,driver)
    ACCION_BALCON(balcon,driver)
    ACCION_GARAJE(garaje,driver)
    ACCION_CALEFACCION(garaje,driver)
    ACCION_AIRE_ACONDICIONADO(aire_acondicionado,driver)
    
def CALIFICAR_CHECKBOX(numero,relevancia,driver):
    if 70 <= relevancia <= 80:
        driver.find_element(By.XPATH,'//*[@id="time'+numero+'"]/div/label[1]/input').click()
    elif 81 <= relevancia <= 90:
        driver.find_element(By.XPATH,'//*[@id="time'+numero+'"]/div/label[1]/input').click()
        driver.find_element(By.XPATH,'//*[@id="time'+numero+'"]/div/label[2]/input').click()
    elif 91 <= relevancia <= 100:
        driver.find_element(By.XPATH,'//*[@id="time'+numero+'"]/div/label[1]/input').click()
        driver.find_element(By.XPATH,'//*[@id="time'+numero+'"]/div/label[2]/input').click()
        driver.find_element(By.XPATH,'//*[@id="time'+numero+'"]/div/label[3]/input').click()
        

def GENERADOR_RELEVANCIA():
    rlv=random.randint(50, 100)
    rlv1=random.randint(50, 100)
    rlv2=random.randint(50, 100)
    rlv3=random.randint(50, 100)
    rlv4=random.randint(50, 100)
    rlv5=random.randint(50, 100)
    rlv6=random.randint(50, 100)
    return rlv,rlv1,rlv2,rlv3,rlv4,rlv5,rlv6

def ACTIVAR_DESACTIVAR_CHEKBOX(rlv,rlv1,rlv2,rlv3,rlv4,rlv5,rlv6,driver):
    #rlv relevancia obtenido del data frame
    CALIFICAR_CHECKBOX("",rlv,driver)
    CALIFICAR_CHECKBOX(str(1),rlv1,driver)
    CALIFICAR_CHECKBOX(str(2),rlv2,driver)
    CALIFICAR_CHECKBOX(str(3),rlv3,driver)
    CALIFICAR_CHECKBOX(str(4),rlv4,driver)
    CALIFICAR_CHECKBOX(str(5),rlv5,driver)
    CALIFICAR_CHECKBOX(str(6),rlv6,driver)        

    
# Bucle while para actualizar la página cada cierto intervalo de tiempo
def SIMULAR_AGENTE(df_casas,driver):
    while True:
        SELECCION_CARACTERISTICA(df_casas,driver)
        time.sleep(7)
        driver.find_element(By.ID,'btnMemb').click()
        # Espera unos segundos para que aparezca la ventana emergente
        time.sleep(3)
        # Inicia una nueva aplicación para la ventana emergente
        app = Application(backend="uia").connect(title="Figure 1")
        # Encuentra y cierra la ventana emergente
        app.window(title="Figure 1").close()
        time.sleep(3)
        driver.find_element(By.ID,'btn').click()
        time.sleep(3)
        rlv,rlv1,rlv2,rlv3,rlv4,rlv5,rlv6=GENERADOR_RELEVANCIA()
        ACTIVAR_DESACTIVAR_CHEKBOX(rlv,rlv1,rlv2,rlv3,rlv4,rlv5,rlv6,driver)
        
        time.sleep(2)
        
        driver.find_element(By.ID,'btnCal').click()
        time.sleep(2)
        
        driver.find_element(By.XPATH,'//*[@id="shiny-modal"]/div/div/div[3]/button').click()
        
        ACTIVAR_DESACTIVAR_CHEKBOX(rlv,rlv1,rlv2,rlv3,rlv4,rlv5,rlv6,driver)
        #btnMemb
        time.sleep(2)
        driver.refresh()
