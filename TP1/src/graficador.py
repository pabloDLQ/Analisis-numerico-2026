import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from matplotlib.ticker import MultipleLocator
import os

def graficar_tiempo_frecuencia(fs, data, titulo="Analisis Tiempo-Frecuencia", color='b', metricas=None, archivo_salida=None):
    """
    Grafica una senal en tiempo y frecuencia (FFT completa y acotada [0-5000 Hz]).
    
    Parametros:
    - fs: frecuencia de muestreo (Hz)
    - data: array con las muestras
    - titulo: titulo principal de la grafica
    - color: color de la linea
    - metricas: diccionario con metricas de calidad (opcional)
    """
    # Crear figura con 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # --- SUBPLOT 1: Dominio del tiempo ---
    tiempo = np.arange(len(data)) / fs
    ax1.plot(tiempo, data, color=color, linewidth=0.8)
    ax1.set_title(f"{titulo} - Dominio del Tiempo")
    ax1.set_xlabel("Tiempo (s)")
    ax1.set_ylabel("Amplitud")
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Normalizar las marcas del eje Y cada 10000 unidades
    ax1.yaxis.set_major_locator(MultipleLocator(10000))
    
    # Agregar metricas al grafico de tiempo
    if metricas:
        rms = metricas.get('rms', 0)
        pico = metricas.get('pico', 0)
        duracion = metricas.get('duracion_s', 0)
        info_text = f"RMS: {rms:.2e} | Pico: {pico:.2e} | Duracion: {duracion:.2f}s"
        ax1.text(0.02, 0.95, info_text, transform=ax1.transAxes, 
                fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Aplicar ventana de Hann una sola vez
    ventana = signal.windows.hann(len(data))
    datos_ventaneados = data * ventana
    
    # FFT
    fft = np.fft.fft(datos_ventaneados)
    frecuencias = np.fft.fftfreq(len(data), 1/fs)
    magnitud = np.abs(fft)
    
    # Solo frecuencias positivas
    idx_positivas = frecuencias >= 0
    frecuencias_pos = frecuencias[idx_positivas]
    magnitud_pos = magnitud[idx_positivas]
    
    # Convertir a dB
    magnitud_db = 20 * np.log10(magnitud_pos + 1e-10)
    
    # --- SUBPLOT 2: Dominio de la frecuencia (FFT completa) ---
    ax2.plot(frecuencias_pos, magnitud_db, color=color, linewidth=0.8)
    ax2.set_title(f"{titulo} - FFT Completa (0-{fs/2:.0f} Hz)")
    ax2.set_xlabel("Frecuencia (Hz)")
    ax2.set_ylabel("Magnitud (dB)")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_xlim([0, fs/2])  # Mostrar hasta Nyquist
    
    # Marcar pico principal - excluyendo DC y frecuencias muy bajas
    min_freq = 20  # Hz - excluye DC y ruido de baja frecuencia
    idx_min_freq = np.argmax(frecuencias_pos >= min_freq)
    
    # Si hay frequencies > min_freq, buscar pico en ese rango
    if frecuencias_pos[idx_min_freq] >= min_freq:
        mag_filtrada = magnitud_pos[idx_min_freq:]
        idx_pico_local = np.argmax(mag_filtrada)
        idx_pico = idx_min_freq + idx_pico_local
    else:
        # Fallback: buscar desde indice 1 (excluir indice 0 que es DC)
        idx_pico = 1 + np.argmax(magnitud_pos[1:])
    
    freq_pico = frecuencias_pos[idx_pico]
    pot_pico = magnitud_db[idx_pico]
    ax2.plot(freq_pico, pot_pico, 'r*', markersize=15, label=f"Pico: {freq_pico:.1f} Hz")
    ax2.legend()
    
    # --- SUBPLOT 3: FFT Acotada [0-5000 Hz] ---
    freq_max = 5000
    idx_max = np.argmax(frecuencias_pos > freq_max)
    if idx_max == 0:
        idx_max = len(frecuencias_pos)
    
    frecuencias_acotadas = frecuencias_pos[:idx_max]
    magnitud_acotada = magnitud_db[:idx_max]
    
    ax3.plot(frecuencias_acotadas, magnitud_acotada, color=color, linewidth=0.8)
    ax3.set_title(f"{titulo} - FFT Acotada (0-5000 Hz)")
    ax3.set_xlabel("Frecuencia (Hz)")
    ax3.set_ylabel("Magnitud (dB)")
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.set_xlim([0, freq_max])
    
    # Marcar pico en el rango acotado
    idx_min_freq_acotado = np.argmax(frecuencias_acotadas >= min_freq)
    if len(magnitud_acotada) > idx_min_freq_acotado:
        mag_filtrada_acotada = magnitud_acotada[idx_min_freq_acotado:]
        if len(mag_filtrada_acotada) > 0:
            idx_pico_local = np.argmax(mag_filtrada_acotada)
            idx_pico_acotado = idx_min_freq_acotado + idx_pico_local
            freq_pico_acotado = frecuencias_acotadas[idx_pico_acotado]
            pot_pico_acotado = magnitud_acotada[idx_pico_acotado]
            ax3.plot(freq_pico_acotado, pot_pico_acotado, 'r*', markersize=15, label=f"Pico: {freq_pico_acotado:.1f} Hz")
            ax3.legend()
    
    plt.subplots_adjust(hspace=0.35, wspace=0.3)
    
    # Guardar el grafico si se especifica un archivo de salida
    if archivo_salida:
        carpeta = os.path.dirname(archivo_salida)
        if carpeta and not os.path.exists(carpeta):
            os.makedirs(carpeta)
        fig.savefig(archivo_salida, dpi=300, bbox_inches='tight')
        print(f"Grafico guardado: {archivo_salida}")
        plt.close(fig)  # Cerrar la figura para liberar memoria


def graficar_comparacion_seniales(fs1, data1, fs2, data2, titulo1="Sirena 1", titulo2="Sirena 2", metricas1=None, metricas2=None, archivo_salida=None):
    """
    Grafica dos senales lado a lado para comparacion.
    
    Parametros:
    - fs1, fs2: frecuencias de muestreo
    - data1, data2: datos de las senales
    - titulo1, titulo2: titulos de cada senal
    - metricas1, metricas2: diccionarios con metricas de calidad
    """
    fig = plt.figure(figsize=(14, 10))
    
    # 4 subplots: tiempo y frecuencia para cada senal
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    colores = ['blue', 'red']
    datas = [data1, data2]
    fss = [fs1, fs2]
    titulos = [titulo1, titulo2]
    metricas_lista = [metricas1, metricas2]
    
    for idx, (data, fs, titulo, color, metricas) in enumerate(zip(datas, fss, titulos, colores, metricas_lista)):
        # --- Tiempo ---
        ax_time = fig.add_subplot(gs[0, idx])
        tiempo = np.arange(len(data)) / fs
        ax_time.plot(tiempo, data, color=color, linewidth=0.8)
        ax_time.set_title(f"{titulo} - Tiempo")
        ax_time.set_xlabel("Tiempo (s)")
        ax_time.set_ylabel("Amplitud")
        ax_time.grid(True, linestyle='--', alpha=0.6)
        
        # Agregar info
        if metricas:
            snr = metricas.get('snr_db', 0)
            rms = metricas['metricas'].get('rms', 0) if 'metricas' in metricas else 0
            info_text = f"SNR: {snr:.2f} dB\nRMS: {rms:.2e}"
            ax_time.text(0.98, 0.97, info_text, transform=ax_time.transAxes, 
                        fontsize=9, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # --- Frecuencia ---
        ax_freq = fig.add_subplot(gs[1, idx])
        
        # FFT
        ventana = signal.windows.hann(len(data))
        datos_ventaneados = data * ventana
        fft = np.fft.fft(datos_ventaneados)
        frecuencias = np.fft.fftfreq(len(data), 1/fs)
        magnitud = np.abs(fft)
        
        idx_positivas = frecuencias >= 0
        frecuencias_pos = frecuencias[idx_positivas]
        magnitud_pos = magnitud[idx_positivas]
        magnitud_db = 20 * np.log10(magnitud_pos + 1e-10)
        
        ax_freq.plot(frecuencias_pos, magnitud_db, color=color, linewidth=0.8)
        ax_freq.set_title(f"{titulo} - Frecuencia")
        ax_freq.set_xlabel("Frecuencia (Hz)")
        ax_freq.set_ylabel("Magnitud (dB)")
        ax_freq.grid(True, linestyle='--', alpha=0.6)
        ax_freq.set_xlim([0, fs/2])
        
        # Marcar pico
        if metricas and 'freq_pico' in metricas:
            freq_pico = metricas['freq_pico']
            pot_pico = metricas['pot_pico']
            ax_freq.plot(freq_pico, pot_pico, 'r*', markersize=15, label=f"Pico: {freq_pico:.1f} Hz")
            ax_freq.legend()
    
    
    fig.suptitle("Comparacion Detallada de Senales de Sirenas", fontsize=14, fontweight='bold')
    plt.subplots_adjust(hspace=0.35, wspace=0.3, top=0.93)
    
    # Guardar el grafico si se especifica un archivo de salida
    if archivo_salida:
        carpeta = os.path.dirname(archivo_salida)
        if carpeta and not os.path.exists(carpeta):
            os.makedirs(carpeta)
        fig.savefig(archivo_salida, dpi=300, bbox_inches='tight')
        print(f"Grafico guardado: {archivo_salida}")
        plt.close(fig)  # Cerrar la figura para liberar memoria