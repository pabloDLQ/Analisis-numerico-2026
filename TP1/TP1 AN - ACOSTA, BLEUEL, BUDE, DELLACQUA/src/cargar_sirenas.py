import os
import numpy as np
from scipy.io import wavfile

def cargar_audio(ruta_archivo):
    """
    Carga un archivo WAV y devuelve la frecuencia de muestreo y los datos.
    Si el archivo tiene multiples canales, devuelve el primer canal (mono).
    """
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"El archivo {ruta_archivo} no existe.")
    fs, data = wavfile.read(ruta_archivo) #fs es la frecuencia de muestreo, data es un array con los datos de audio

    if data.ndim > 1:       # Si el audio tiene multiples canales, solo toma el primero (mono)
        data = data[:, 0]
    return fs, data         # Retorna la frecuencia y los datos

def cargar_sirenas(carpeta='data'):
    """
    Carga los archivos sirena_1.wav y sirena_2.wav desde la carpeta especificada.
    Devuelve un diccionario con la informacion.
    """
    rutas = {           #diccionario con las rutas de los archivos
        'sirena1': os.path.join(carpeta, 'sirena_1.wav'),
        'sirena2': os.path.join(carpeta, 'sirena_2.wav')
    }
    resultados = {}     #diccionario vacio para almacenar los resultados
    for nombre, ruta in rutas.items():
        try:
            fs, data = cargar_audio(ruta)
            resultados[nombre] = {'fs': fs, 'data': data, 'ruta': ruta}
            #print(f"Archivo {nombre} cargado: fs={fs} Hz, duracion={len(data)/fs:.2f} s, muestras={len(data)}")
        except Exception as e:
            print(f"Error al cargar {nombre}: {e}")
            resultados[nombre] = None
    return resultados

