# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 01:37:35 2024

@author: UTM
"""

import time
import random

def editar_numero_archivo(nombre_archivo, nuevo_numero):
    try:
        # Abrir el archivo en modo escritura
        with open(nombre_archivo, 'w') as archivo:
            # Escribir el nuevo número en el archivo
            archivo.write(str(nuevo_numero))
        print("Número editado exitosamente en el archivo.")
    except Exception as e:
        print("Error al editar el número en el archivo:", e)


def AGENTE_REGISTRO_VIVIENDA(n_usuarios,n_viviendas):
    while True:
        time.sleep(20)
        editar_numero_archivo(n_usuarios, random.randint(100, 120))
        editar_numero_archivo(n_viviendas, random.randint(100, 120))
