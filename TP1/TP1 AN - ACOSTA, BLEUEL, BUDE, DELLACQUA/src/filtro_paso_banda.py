import numpy as np
from scipy.signal import butter, filtfilt

def disenar_filtro_paso_banda(fs, freq_min, freq_max, orden=5):
    """
    Diseña un filtro paso banda Butterworth.
    
    Parametros:
    -----------
    fs : float
        Frecuencia de muestreo en Hz
    freq_min : float
        Frecuencia minima del filtro en Hz (banda de paso commence aqui)
    freq_max : float
        Frecuencia maxima del filtro en Hz (banda de paso termina aqui)
    orden : int, opcional
        Orden del filtro (por defecto 5)
    
    Retorna:
    --------
    b, a : arrays
        Coeficientes del filtro (numerador y denominador)
    """
    if freq_min <= 0 or freq_max <= 0:
        raise ValueError("Las frecuencias deben ser positivas")
    if freq_min >= freq_max:
        raise ValueError("freq_min debe ser menor que freq_max")
    if freq_max >= fs / 2:
        raise ValueError(f"freq_max debe ser menor que la frecuencia de Nyquist ({fs/2} Hz)")
    
    # Normalizar las frecuencias respecto a la frecuencia de Nyquist
    nyquist = fs / 2
    freq_min_norm = freq_min / nyquist
    freq_max_norm = freq_max / nyquist
    
    # Diseñar el filtro paso banda
    b, a = butter(orden, [freq_min_norm, freq_max_norm], btype='band')
    return b, a

def aplicar_filtro_paso_banda(data, fs, freq_min, freq_max, orden=5):
    """
    Aplica un filtro paso banda a una senal de audio.
    
    Parametros:
    -----------
    data : array-like
        Array con los datos de la senal (dominio del tiempo)
    fs : float
        Frecuencia de muestreo en Hz
    freq_min : float
        Frecuencia minima del filtro en Hz
    freq_max : float
        Frecuencia maxima del filtro en Hz
    orden : int, opcional
        Orden del filtro (por defecto 5)
    
    Retorna:
    --------
    senal_filtrada : ndarray
        Array con la senal filtrada
    """
    # Diseñar el filtro
    b, a = disenar_filtro_paso_banda(fs, freq_min, freq_max, orden)
    
    # Aplicar el filtro (filtfilt aplica el filtro dos veces para mantener la fase)
    senal_filtrada = filtfilt(b, a, data)
    
    return senal_filtrada

def filtrar_dictado(sirenas_dict, freq_min, freq_max, orden=5):
    """
    Aplica un filtro paso banda a un diccionario de sirenas o senales.
    
    Parametros:
    -----------
    sirenas_dict : dict
        Diccionario con la estructura: {'sirena_X': {'fs': fs, 'data': data, 'ruta': ruta}, ...}
    freq_min : float
        Frecuencia minima del filtro en Hz
    freq_max : float
        Frecuencia maxima del filtro en Hz
    orden : int, opcional
        Orden del filtro (por defecto 5)
    
    Retorna:
    --------
    sirenas_filtradas : dict
        Diccionario con la misma estructura pero con los datos filtrados
    """
    sirenas_filtradas = {}
    
    for nombre, sirena in sirenas_dict.items():
        if sirena is None:
            sirenas_filtradas[nombre] = None
            continue
        
        fs = sirena['fs']
        data = sirena['data']
        ruta = sirena['ruta']
        
        # Aplicar el filtro
        data_filtrada = aplicar_filtro_paso_banda(data, fs, freq_min, freq_max, orden)
        
        # Guardar en la estructura
        sirenas_filtradas[nombre] = {
            'fs': fs,
            'data': data_filtrada,
            'ruta': ruta
        }
    
    return sirenas_filtradas
