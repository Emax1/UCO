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



def ACCION_LOCALIZACION(localizacion):
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
    

def ACCION_ASCENSOR(var1):
    if var1 == 1:
        numero = 2
    elif var1 == 0:
        numero = 1
    id_elemento="//*[@id='ascensor']/option["+str(numero)+"]"
    #return id_elemento
    driver.find_element(By.XPATH,id_elemento).click()
    


def ACCION_BANOS(var1):
    #return id_elemento
    numero=var1-1
    slideFrom = driver.find_element(By.XPATH,"/html/body/bslib-layout-columns/div[1]/div/div/bslib-layout-columns/div[1]/div/div/div[3]/span/span[3]")#.click()
    # Mover el cursor del mouse a las coordenadas calculadas y soltar el clic
    actions = ActionChains(driver)
    actions.move_to_element(slideFrom).click_and_hold()
    actions.move_by_offset(numero*100, 0).release().perform()


def ACCION_TRASTERO(var1):
    if var1 == 1:
        numero = 2
    elif var1 == 0:
        numero = 1
    id_elemento="//*[@id='trastero']/option["+str(numero)+"]"
    #return id_elemento
    driver.find_element(By.XPATH,id_elemento).click()
    


def ACCION_PISO(var1):
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
    


def ACCION_HABITACIONES(var1):
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


def ACCION_METROS2(var1):
    ##### LLENAR UN CUADRO DE TEXTO
    cuadro_texto =driver.find_element(By.ID,"metros_cuadrados")#.click()

    # Limpiar el cuadro de texto (opcional)
    cuadro_texto.clear()

    # Escribir texto en el cuadro de texto
    texto_a_ingresar = int(var1)
    cuadro_texto.send_keys(texto_a_ingresar)



def ACCION_CONDICION(var1):
    if var1 == 'Segunda mano/buen estado':
        numero = 1
    elif var1 == 'Segunda mano/para reformar':
        numero = 2
    elif var1 == 'Promoción de obra nueva':
        numero = 3
    id_elemento="//*[@id='condicion']/option["+str(numero)+"]"
    #return id_elemento
    driver.find_element(By.XPATH,id_elemento).click()
    


def ACCION_ARMARIOS(var1):
    if var1 == 1:
        numero = 2
    elif var1 == 0:
        numero = 1
    id_elemento="//*[@id='armarios_empotrados']/option["+str(numero)+"]"
    #return id_elemento
    driver.find_element(By.XPATH,id_elemento).click()


def ACCION_TERRAZA(var1):
    if var1 == 1:
        numero = 2
    elif var1 == 0:
        numero = 1
    id_elemento="//*[@id='terraza']/option["+str(numero)+"]"
    #return id_elemento
    driver.find_element(By.XPATH,id_elemento).click()


def ACCION_BALCON(var1):
    if var1 == 1:
        numero = 2
    elif var1 == 0:
        numero = 1
    id_elemento="//*[@id='balcon']/option["+str(numero)+"]"
    #return id_elemento
    driver.find_element(By.XPATH,id_elemento).click()



def ACCION_GARAJE(var1):
    if var1 == 1:
        numero = 2
    elif var1 == 0:
        numero = 1
    id_elemento="//*[@id='garaje']/option["+str(numero)+"]"
    #return id_elemento
    driver.find_element(By.XPATH,id_elemento).click()


def ACCION_CALEFACCION(var1):
    if var1 == 1:
        numero = 2
    elif var1 == 0:
        numero = 1
    id_elemento="//*[@id='calefaccion']/option["+str(numero)+"]"
    #return id_elemento
    driver.find_element(By.XPATH,id_elemento).click()



def ACCION_AIRE_ACONDICIONADO(var1):
    if var1 == 1:
        numero = 2
    elif var1 == 0:
        numero = 1
    id_elemento="//*[@id='aire']/option["+str(numero)+"]"
    #return id_elemento
    driver.find_element(By.XPATH,id_elemento).click()


def SELECCION_CARACTERISTICA(df_casas):
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
    ACCION_LOCALIZACION(localizacion)
    ACCION_ASCENSOR(ascensor)
    ACCION_BANOS(banos)
    ACCION_TRASTERO(trastero)
    ACCION_PISO(piso)
    ACCION_HABITACIONES(habitaciones)
    ACCION_METROS2(metros_reales)
    ACCION_CONDICION(condicion)
    ACCION_ARMARIOS(armarios_empotrados)
    ACCION_TERRAZA(terraza)
    ACCION_BALCON(balcon)
    ACCION_GARAJE(garaje)
    ACCION_CALEFACCION(garaje)
    ACCION_AIRE_ACONDICIONADO(aire_acondicionado)