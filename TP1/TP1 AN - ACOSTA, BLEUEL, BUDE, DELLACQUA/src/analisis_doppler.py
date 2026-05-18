import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import os

def analizar_fft_ventanas_temporales(fs, data, titulo="FFT en Ventanas Temporales", 
                                     tamaño_ventana_s=0.5, color='b', archivo_salida=None):
    """
    Analiza FFT en ventanas temporales pequeñas para observar cambios de frecuencia (efecto Doppler).
    
    Parámetros:
    - fs: frecuencia de muestreo (Hz)
    - data: array con las muestras
    - titulo: título principal
    - tamaño_ventana_s: tamaño de ventana en segundos (default 0.5s)
    - color: color de las líneas
    - archivo_salida: ruta para guardar imagen
    
    Retorna:
    - diccionario con información de frecuencias por ventana
    """
    
    # Calcular tamaño de ventana en muestras
    tamaño_ventana_muestras = int(tamaño_ventana_s * fs)
    num_ventanas = len(data) // tamaño_ventana_muestras
    
    if num_ventanas < 1:
        print(f"Error: Señal muy corta para ventanas de {tamaño_ventana_s}s")
        return None
    
    # Crear figura con subplots
    ventanas_por_fila = 3
    num_filas = (num_ventanas + ventanas_por_fila - 1) // ventanas_por_fila
    
    fig, axes = plt.subplots(num_filas, ventanas_por_fila, figsize=(15, 4*num_filas))
    
    # Asegurar que axes sea un array 2D
    if num_filas == 1 and ventanas_por_fila == 1:
        axes = np.array([[axes]])
    elif num_filas == 1:
        axes = axes.reshape(1, -1)
    elif ventanas_por_fila == 1:
        axes = axes.reshape(-1, 1)
    
    info_ventanas = []
    
    # Procesar cada ventana
    for i in range(num_ventanas):
        inicio = i * tamaño_ventana_muestras
        fin = inicio + tamaño_ventana_muestras
        
        # Extraer ventana
        ventana_data = data[inicio:fin]
        
        # Aplicar ventana de Hann
        ventana_hann = signal.windows.hann(len(ventana_data))
        datos_ventaneados = ventana_data * ventana_hann
        
        # Calcular FFT
        fft = np.fft.fft(datos_ventaneados)
        frecuencias = np.fft.fftfreq(len(ventana_data), 1/fs)
        magnitud = np.abs(fft)
        
        # Solo frecuencias positivas
        idx_positivas = frecuencias >= 0
        frecuencias_pos = frecuencias[idx_positivas]
        magnitud_pos = magnitud[idx_positivas]
        magnitud_db = 20 * np.log10(magnitud_pos + 1e-10)
        
        # Encontrar pico (excluyendo DC y frecuencias bajas)
        min_freq = 50  # Hz
        idx_min_freq = np.argmax(frecuencias_pos >= min_freq)
        
        if idx_min_freq < len(magnitud_pos):
            mag_filtrada = magnitud_pos[idx_min_freq:]
            idx_pico_local = np.argmax(mag_filtrada)
            idx_pico = idx_min_freq + idx_pico_local
        else:
            idx_pico = 1 + np.argmax(magnitud_pos[1:])
        
        freq_pico = frecuencias_pos[idx_pico]
        pot_pico = magnitud_db[idx_pico]
        
        info_ventanas.append({
            'ventana': i,
            'tiempo_inicio': inicio / fs,
            'tiempo_fin': fin / fs,
            'freq_pico': freq_pico,
            'magnitud_pico': pot_pico
        })
        
        # Graficar
        fila = i // ventanas_por_fila
        col = i % ventanas_por_fila
        ax = axes[fila, col]
        
        ax.plot(frecuencias_pos, magnitud_db, color=color, linewidth=0.8)
        ax.plot(freq_pico, pot_pico, 'r*', markersize=12)
        
        tiempo_inicio = inicio / fs
        tiempo_fin = fin / fs
        ax.set_title(f"Ventana {i+1} ({tiempo_inicio:.2f}-{tiempo_fin:.2f}s)\nPico: {freq_pico:.1f} Hz", 
                    fontsize=9)
        ax.set_xlabel("Frecuencia (Hz)", fontsize=8)
        ax.set_ylabel("Magnitud (dB)", fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim([0, 2000])
    
    # Desactivar subplots vacíos
    total_subplots = num_filas * ventanas_por_fila
    for i in range(num_ventanas, total_subplots):
        fila = i // ventanas_por_fila
        col = i % ventanas_por_fila
        axes[fila, col].set_visible(False)
    
    fig.suptitle(f"{titulo} (ventanas de {tamaño_ventana_s}s)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Guardar si se especifica
    if archivo_salida:
        carpeta = os.path.dirname(archivo_salida)
        if carpeta and not os.path.exists(carpeta):
            os.makedirs(carpeta)
        fig.savefig(archivo_salida, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado: {archivo_salida}")
        plt.close(fig)
    
    return info_ventanas


def calcular_velocidad_ambulancia(freq_min, freq_max, freq_promedio=None):
    """
    Calcula la velocidad de la ambulancia usando efecto Doppler.
    
    Calcula la velocidad en el período de frecuencia máxima y en el período
    de frecuencia mínima, luego promedia ambos resultados.
    
    Parámetros:
    - freq_min: frecuencia mínima detectada (Hz)
    - freq_max: frecuencia máxima detectada (Hz)
    - freq_promedio: frecuencia de reposo estimada (si no se proporciona, usa promedio)
    
    Retorna:
    - diccionario con velocidades calculadas
    """
    
    v_sonido = 343  # m/s a 20°C
    
    if freq_promedio is None:
        freq_promedio = (freq_min + freq_max) / 2
    
    # Fórmula Doppler (observador estacionario):
    # Asumimos que en el período de freq_max, la ambulancia se acerca
    # y en el período de freq_min, se aleja
    
    # Para acercamiento (usando freq_max):
    # f_max = f0 * v_sonido / (v_sonido - v_ambulancia)
    # Despejando: v_ambulancia = v_sonido * (1 - f0 / f_max)
    v_acerca = abs(v_sonido * (1 - freq_promedio / freq_max))
    
    # Para alejamiento (usando freq_min):
    # f_min = f0 * v_sonido / (v_sonido + v_ambulancia)
    # Despejando: v_ambulancia = v_sonido * (f0 / f_min - 1)
    v_aleja = abs(v_sonido * (freq_promedio / freq_min - 1))
    
    # Promedio entre acercamiento y alejamiento
    v_promedio = (v_acerca + v_aleja) / 2
    
    return {
        'v_acerca_ms': v_acerca,
        'v_acerca_kmh': v_acerca * 3.6,
        'v_aleja_ms': v_aleja,
        'v_aleja_kmh': v_aleja * 3.6,
        'v_promedio_ms': v_promedio,
        'v_promedio_kmh': v_promedio * 3.6,
        'v_sonido': v_sonido
    }


def calcular_velocidades_por_ventana(frecuencias_pico, freq_promedio=None):
    """
    Calcula velocidades de acercamiento y alejamiento para cada ventana.
    
    Parámetros:
    - frecuencias_pico: lista de frecuencias pico detectadas en cada ventana
    - freq_promedio: frecuencia de reposo estimada (si no se proporciona, usa promedio)
    
    Retorna:
    - tupla: (velocidades_acerca, velocidades_aleja) en m/s
    """
    
    v_sonido = 343  # m/s a 20°C
    
    if freq_promedio is None:
        freq_promedio = np.mean(frecuencias_pico)
    
    velocidades_acerca = []
    velocidades_aleja = []
    
    # Comparar cada frecuencia con el promedio
    for freq_pico in frecuencias_pico:
        if freq_pico > freq_promedio:
            # Acercamiento: frecuencia aumenta (mayor que promedio)
            v_acerca = abs(v_sonido * (1 - freq_promedio / freq_pico))
            velocidades_acerca.append(v_acerca)
            velocidades_aleja.append(None)
        else:
            # Alejamiento: frecuencia disminuye (menor que promedio)
            v_aleja = abs(v_sonido * (freq_promedio / freq_pico - 1))
            velocidades_acerca.append(None)
            velocidades_aleja.append(v_aleja)
    
    return velocidades_acerca, velocidades_aleja


def calcular_velocidades_metodo2(frecuencias_pico, info_ventanas, freq_real, instante_paso,
                                  t_freq=None, frecuencias_instantaneas=None):
    """
    MÉTODO 2: Calcula velocidades usando la frecuencia real del espectrograma e instante de paso.
    
    Clasifica cada ventana como acercamiento o alejamiento basándose en:
    - Si tiempo_fin < instante_paso: ACERCAMIENTO
    - Si tiempo_fin > instante_paso: ALEJAMIENTO
    
    Si se proporciona t_freq y frecuencias_instantaneas, usa la función fundamental
    para obtener la frecuencia real correspondiente a cada instante.
    
    Parámetros:
    -----------
    frecuencias_pico : list
        Lista de frecuencias pico detectadas en cada ventana
    info_ventanas : list
        Lista de dicts con información de cada ventana (tiempo_inicio, tiempo_fin, etc.)
    freq_real : float
        Frecuencia real emitida (f₀) obtenida del espectrograma
    instante_paso : float
        Instante de paso más cercano del micrófono (en segundos)
    t_freq : ndarray, optional
        Array de tiempo de las ventanas STFT (para función fundamental)
    frecuencias_instantaneas : ndarray, optional
        Array de frecuencias instantáneas detectadas (para función fundamental)
    
    Retorna:
    --------
    tuple : (velocidades_acerca, velocidades_aleja, clasificaciones)
        - velocidades_acerca: lista de velocidades en acercamiento (None si no aplica)
        - velocidades_aleja: lista de velocidades en alejamiento (None si no aplica)
        - clasificaciones: lista de strings indicando la clasificación de cada ventana
    """
    
    v_sonido = 343  # m/s a 20°C
    
    # Si se proporciona datos para la función fundamental, calcularla
    if t_freq is not None and frecuencias_instantaneas is not None:
        from src.analizar_espectrograma import frecuencia_fundamental
        freq_fundamental_array = frecuencia_fundamental(t_freq, frecuencias_instantaneas, instante_paso)
    else:
        freq_fundamental_array = None
    
    velocidades_acerca = []
    velocidades_aleja = []
    clasificaciones = []
    
    # Procesar cada ventana
    for i, (freq_pico, info) in enumerate(zip(frecuencias_pico, info_ventanas)):
        tiempo_fin = info['tiempo_fin']
        
        # Determinar la frecuencia real a usar para esta ventana
        # Si tenemos función fundamental, usar el valor correspondiente al índice más cercano
        if freq_fundamental_array is not None:
            # Encontrar el índice más cercano en t_freq al tiempo_fin de esta ventana
            idx_cercano = np.argmin(np.abs(t_freq - tiempo_fin))
            freq_real_ventana = freq_fundamental_array[idx_cercano]
        else:
            freq_real_ventana = freq_real
        
        # Clasificar basándose en el tiempo relativo al instante de paso
        if tiempo_fin < instante_paso:
            # ACERCAMIENTO: ventana termina antes del instante de paso
            v_acerca = abs(v_sonido * (1 - freq_real_ventana / freq_pico))
            velocidades_acerca.append(v_acerca)
            velocidades_aleja.append(None)
            clasificaciones.append("ACERCAMIENTO")
        else:
            # ALEJAMIENTO: ventana termina después del instante de paso
            v_aleja = abs(v_sonido * (freq_real_ventana / freq_pico - 1))
            velocidades_acerca.append(None)
            velocidades_aleja.append(v_aleja)
            clasificaciones.append("ALEJAMIENTO")
    
    return velocidades_acerca, velocidades_aleja, clasificaciones


def obtener_datos_espectrograma(fs, data, numero_sirena=1, tamaño_ventana_s=0.5):
    """
    Obtiene los datos del espectrograma sin generar gráficos.
    
    Parámetros:
    -----------
    fs : float
        Frecuencia de muestreo
    data : ndarray
        Array con las muestras de audio
    numero_sirena : int
        Número de sirena (1 o 2)
    tamaño_ventana_s : float
        Tamaño de ventana STFT
    
    Retorna:
    --------
    dict : Diccionario con:
        - frecuencia_real_hz: Frecuencia real emitida
        - instante_paso_s: Instante de paso del micrófono
        - info_estimacion: Detalles adicionales
        - t_freq: Array de tiempo de las ventanas STFT
        - frecuencias_instantaneas: Array de frecuencias instantáneas detectadas
    """
    
    from src.analizar_espectrograma import _estimar_frecuencia_real_e_instante_paso
    
    # Calcular parámetros STFT
    nperseg = int(tamaño_ventana_s * fs)
    noverlap = int(nperseg * 0.75)
    
    # Estimar frecuencia real e instante de paso
    # Ahora retorna 5 valores: frecuencia_real, instante_paso, info_estimacion, t, frecuencias_instantaneas
    frecuencia_real, instante_paso, info_estimacion, t_freq, frecuencias_instantaneas = _estimar_frecuencia_real_e_instante_paso(
        fs, data, nperseg, noverlap
    )
    
    return {
        'frecuencia_real_hz': frecuencia_real,
        'instante_paso_s': instante_paso,
        'info_estimacion': info_estimacion,
        't_freq': t_freq,
        'frecuencias_instantaneas': frecuencias_instantaneas
    }

