"""
Generador de gráficos FFT en ventanas temporales de 0.5 segundos.

Este módulo genera 16 gráficos FFT para Sirena 1 (8 segundos de duración)
usando ventanas de 0.5 segundos cada una, mostrando la evolución de frecuencias.

Uso desde línea de comandos:
    python -m src.generar_FFT_temporales
        Genera 16 gráficos de FFT en ventanas de 0.5s para Sirena 1

Uso como módulo:
    from src.generar_FFT_temporales import generar_fft_ventanas
    generar_fft_ventanas(fs, data, tamaño_ventana_s=0.5)
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import argparse
import sys
import os

# Importar funciones de cargar_sirenas
from src.cargar_sirenas import cargar_sirenas


def generar_fft_ventana(fs, data, t_inicio, t_fin, num_ventana, 
                       titulo_base="FFT Sirena 1", color='b', archivo_salida=None):
    """
    Genera un gráfico FFT acotado (0-5000 Hz) para una ventana temporal específica.
    
    Parámetros:
    - fs: frecuencia de muestreo (Hz)
    - data: array con las muestras
    - t_inicio: tiempo de inicio en segundos
    - t_fin: tiempo de fin en segundos
    - num_ventana: número de ventana (para identificación)
    - titulo_base: base del título del gráfico
    - color: color de las líneas
    - archivo_salida: ruta completa del archivo PNG a guardar (opcional)
    
    Retorna:
    - diccionario con información de la ventana
    """
    
    # Convertir tiempos a índices de muestras
    idx_inicio = int(t_inicio * fs)
    idx_fin = int(t_fin * fs)
    
    # Extraer segmento
    segmento = data[idx_inicio:idx_fin]
    
    # Crear figura con 1 subplot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Información de la ventana
    pico = np.max(np.abs(segmento))
    rms = np.sqrt(np.mean(segmento**2))
    
    # Aplicar ventana de Hann
    ventana = signal.windows.hann(len(segmento))
    segmento_ventaneado = segmento * ventana
    
    # Calcular FFT
    fft_resultado = np.fft.fft(segmento_ventaneado)
    frecuencias = np.fft.fftfreq(len(segmento), 1/fs)
    magnitud = np.abs(fft_resultado)
    
    # Solo frecuencias positivas
    idx_positivas = frecuencias >= 0
    frecuencias_pos = frecuencias[idx_positivas]
    magnitud_pos = magnitud[idx_positivas]
    
    # Convertir a dB (escala logarítmica)
    magnitud_db = 20 * np.log10(magnitud_pos + 1e-10)
    
    # --- Gráfico FFT (0-5000 Hz) ---
    idx_max_freq = np.where(frecuencias_pos <= 5000)[0]
    ax.plot(frecuencias_pos[idx_max_freq], magnitud_db[idx_max_freq], color=color, linewidth=0.8)
    ax.set_title(f"{titulo_base} - Ventana {num_ventana} [{t_inicio:.1f}s - {t_fin:.1f}s] - Gráfico FFT")
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Magnitud (dB)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(0, 5000)
    
    # Información de la ventana en el gráfico
    info_text = f"Duración: 0.50s | RMS: {rms:.2e} | Pico: {pico:.2e}"
    ax.text(0.02, 0.95, info_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Encontrar y marcar picos en rango acotado
    picos_acotados_idx, _ = signal.find_peaks(magnitud_db[idx_max_freq], height=-40, distance=20)
    if len(picos_acotados_idx) > 0:
        top_picos_acotados = picos_acotados_idx[np.argsort(magnitud_db[idx_max_freq][picos_acotados_idx])[-3:]]
        top_picos_acotados = np.sort(top_picos_acotados)
        for idx in top_picos_acotados:
            ax.plot(frecuencias_pos[idx_max_freq][idx], magnitud_db[idx_max_freq][idx], 'ro', markersize=6)
    
    plt.tight_layout()
    
    # Guardar imagen
    if archivo_salida:
        os.makedirs(os.path.dirname(archivo_salida), exist_ok=True)
        plt.savefig(archivo_salida, dpi=150, bbox_inches='tight')
        print(f"[{num_ventana:02d}] Guardado: {os.path.basename(archivo_salida)}")
    
    plt.close()
    
    # Encontrar frecuencia dominante
    idx_max = np.argmax(magnitud_pos)
    freq_dominante = frecuencias_pos[idx_max]
    
    # Información de la ventana
    info_ventana = {
        'num_ventana': num_ventana,
        'intervalo': f'{t_inicio:.1f}s - {t_fin:.1f}s',
        'frecuencia_dominante_Hz': float(freq_dominante),
        'magnitud_maxima_dB': float(magnitud_db[idx_max]),
        'rms': float(rms),
        'pico': float(pico)
    }
    
    return info_ventana


def generar_fft_ventanas_combinadas(fs, data, t_inicios, titulo_base="FFT Sirena 1", 
                                   color='b', archivo_salida=None):
    """
    Genera un gráfico combinado con 3 ventanas FFT consecutivas (una debajo de otra).
    
    Parámetros:
    - fs: frecuencia de muestreo (Hz)
    - data: array con las muestras
    - t_inicios: lista con 3 tiempos de inicio en segundos
    - titulo_base: base del título del gráfico
    - color: color de las líneas
    - archivo_salida: ruta para guardar la imagen (opcional)
    
    Retorna:
    - diccionario con información de las ventanas
    """
    
    duracion_total = len(data) / fs
    tamaño_ventana_s = t_inicios[1] - t_inicios[0]  # Asumir espaciado uniforme
    
    # Crear figura con 3 subplots (3 ventanas × 1 gráfico cada una)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    info_ventanas = []
    
    for fila, t_inicio in enumerate(t_inicios):
        t_fin = min(t_inicio + tamaño_ventana_s, duracion_total)
        
        # Convertir tiempos a índices
        idx_inicio = int(t_inicio * fs)
        idx_fin = int(t_fin * fs)
        segmento = data[idx_inicio:idx_fin]
        
        # Información de la ventana
        pico = np.max(np.abs(segmento))
        rms = np.sqrt(np.mean(segmento**2))
        
        # Aplicar ventana de Hann y calcular FFT
        ventana = signal.windows.hann(len(segmento))
        segmento_ventaneado = segmento * ventana
        fft_resultado = np.fft.fft(segmento_ventaneado)
        frecuencias = np.fft.fftfreq(len(segmento), 1/fs)
        magnitud = np.abs(fft_resultado)
        
        # Solo frecuencias positivas
        idx_positivas = frecuencias >= 0
        frecuencias_pos = frecuencias[idx_positivas]
        magnitud_pos = magnitud[idx_positivas]
        magnitud_db = 20 * np.log10(magnitud_pos + 1e-10)
        
        # --- Gráfico FFT (0-5000 Hz) ---
        idx_max_freq = np.where(frecuencias_pos <= 5000)[0]
        axes[fila].plot(frecuencias_pos[idx_max_freq], magnitud_db[idx_max_freq], 
                          color=color, linewidth=0.8)
        axes[fila].set_title(f"Gráfico FFT [{t_inicio:.1f}s - {t_fin:.1f}s]")
        axes[fila].set_xlabel("Frecuencia (Hz)")
        axes[fila].set_ylabel("Magnitud (dB)")
        axes[fila].grid(True, linestyle='--', alpha=0.6)
        axes[fila].set_xlim(0, 5000)
        
        # Información de la ventana en el gráfico
        info_text = f"RMS: {rms:.2e} | Pico: {pico:.2e}"
        axes[fila].text(0.02, 0.95, info_text, transform=axes[fila].transAxes, 
                fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Marcar picos en rango acotado
        picos_acotados_idx, _ = signal.find_peaks(magnitud_db[idx_max_freq], height=-40, distance=20)
        if len(picos_acotados_idx) > 0:
            top_picos_acotados = picos_acotados_idx[np.argsort(magnitud_db[idx_max_freq][picos_acotados_idx])[-3:]]
            top_picos_acotados = np.sort(top_picos_acotados)
            for idx in top_picos_acotados:
                axes[fila].plot(frecuencias_pos[idx_max_freq][idx], magnitud_db[idx_max_freq][idx], 
                                  'ro', markersize=6)
        
        # Frecuencia dominante
        idx_max = np.argmax(magnitud_pos)
        freq_dominante = frecuencias_pos[idx_max]
        
        # Información de la ventana
        info_ventana = {
            'intervalo': f'{t_inicio:.1f}s - {t_fin:.1f}s',
            'frecuencia_dominante_Hz': float(freq_dominante),
            'magnitud_maxima_dB': float(magnitud_db[idx_max]),
            'rms': float(rms),
            'pico': float(pico)
        }
        info_ventanas.append(info_ventana)
    
    plt.tight_layout()
    
    # Guardar imagen
    if archivo_salida:
        os.makedirs(os.path.dirname(archivo_salida), exist_ok=True)
        plt.savefig(archivo_salida, dpi=150, bbox_inches='tight')
        print(f"    Combinado: {os.path.basename(archivo_salida)}")
    
    plt.close()
    
    return info_ventanas


def generar_fft_ventanas(fs, data, tamaño_ventana_s=0.5, nombre_sirena="Sirena1", 
                        color='b', carpeta_salida=None):
    """
    Genera múltiples gráficos FFT en ventanas temporales.
    
    Parámetros:
    - fs: frecuencia de muestreo (Hz)
    - data: array con las muestras
    - tamaño_ventana_s: tamaño de cada ventana en segundos (default: 0.5s)
    - nombre_sirena: nombre de la sirena (para archivos)
    - color: color de las líneas
    - carpeta_salida: carpeta padre donde guardar los gráficos
    
    Retorna:
    - lista de diccionarios con información de cada ventana
    """
    
    # Calcular número de ventanas
    duracion_total = len(data) / fs
    num_ventanas = int(np.ceil(duracion_total / tamaño_ventana_s))
    
    print(f"\n{'='*80}")
    print(f"Generando {num_ventanas} gráficos FFT en ventanas de {tamaño_ventana_s}s")
    print(f"Duración total de la señal: {duracion_total:.2f}s")
    print(f"{'='*80}\n")
    
    resultados = []
    
    # Crear carpeta con el nombre de la sirena y el tamaño de ventana
    if carpeta_salida:
        # Formatear el nombre de la carpeta (ej: "Sirena1_0.5", "Sirena2_1.0")
        nombre_carpeta_tamaño = f"{tamaño_ventana_s:.1f}".rstrip('0').rstrip('.')
        if '.' in f"{tamaño_ventana_s:.1f}":
            nombre_carpeta_tamaño = f"{tamaño_ventana_s:.1f}"
        
        nombre_carpeta_final = f"{nombre_sirena}_{nombre_carpeta_tamaño}"
        carpeta_tamaño = os.path.join(carpeta_salida, nombre_carpeta_final)
        print(f"Guardando gráficos en: {carpeta_tamaño}/\n")
    else:
        carpeta_tamaño = None
    
    # Generar gráficos para cada ventana
    for i in range(num_ventanas):
        t_inicio = i * tamaño_ventana_s
        t_fin = min((i + 1) * tamaño_ventana_s, duracion_total)
        
        try:
            # Crear nombre de archivo con el intervalo temporal
            nombre_intervalo = f"[{t_inicio:.1f}-{t_fin:.1f}]"
            
            if carpeta_tamaño:
                archivo_salida = os.path.join(carpeta_tamaño, f"{nombre_intervalo}.png")
            else:
                archivo_salida = None
            
            # Generar gráfico
            info = generar_fft_ventana(fs, data, t_inicio, t_fin, i + 1,
                                      titulo_base=f"FFT {nombre_sirena}",
                                      color=color, archivo_salida=archivo_salida)
            resultados.append(info)
            
            # Imprimir información
            print(f"    Frecuencia dominante: {info['frecuencia_dominante_Hz']:7.1f} Hz | "
                  f"Mag: {info['magnitud_maxima_dB']:7.1f} dB | RMS: {info['rms']:.2e}")
            
        except Exception as e:
            print(f"[ERROR] Ventana {i+1}: {e}")
            resultados.append(None)
    
    # Generar gráficos combinados (3 ventanas por imagen)
    print(f"\nGenerando gráficos combinados (3 ventanas por imagen)...")
    num_combinados = (num_ventanas + 2) // 3  # Redondear hacia arriba
    
    for combo_idx in range(num_combinados):
        # Calcular índices de ventanas para este combinado
        idx_inicio = combo_idx * 3
        indices = [idx_inicio + j for j in range(3) if idx_inicio + j < num_ventanas]
        
        if len(indices) == 0:
            break
        
        # Tiempos de inicio para estas ventanas
        t_inicios = [indices[j] * tamaño_ventana_s for j in range(len(indices))]
        
        try:
            if carpeta_tamaño:
                # Crear nombre descriptivo
                t_inicio_combo = t_inicios[0]
                t_fin_combo = min((indices[-1] + 1) * tamaño_ventana_s, duracion_total)
                nombre_combo = f"COMBO_{combo_idx+1}_[{t_inicio_combo:.1f}-{t_fin_combo:.1f}]"
                archivo_combo = os.path.join(carpeta_tamaño, f"{nombre_combo}.png")
            else:
                archivo_combo = None
            
            # Generar gráfico combinado
            generar_fft_ventanas_combinadas(fs, data, t_inicios,
                                           titulo_base=f"FFT {nombre_sirena} - Combinado {combo_idx+1}",
                                           color=color, archivo_salida=archivo_combo)
            
        except Exception as e:
            print(f"[ERROR] Combinado {combo_idx+1}: {e}")
    
    print(f"\n{'='*80}")
    print(f"Proceso completado. {len([r for r in resultados if r])} gráficos individuales + {num_combinados} gráficos combinados.")
    print(f"{'='*80}\n")
    
    return resultados


def main():
    """Función principal para ejecución desde línea de comandos."""
    
    parser = argparse.ArgumentParser(
        description='Genera 16 gráficos FFT en ventanas de 0.5s para Sirena 1.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
    python -m src.generar_FFT_temporales
        Genera 16 gráficos de FFT en ventanas de 0.5s para Sirena 1
    
    python -m src.generar_FFT_temporales --tamaño 1.0
        Genera 8 gráficos de FFT en ventanas de 1.0s
    
    python -m src.generar_FFT_temporales --sin-guardar
        Genera los gráficos pero no los guarda (solo visualiza)
        """
    )
    
    parser.add_argument('--tamaño', type=float, default=0.5,
                       help='Tamaño de la ventana en segundos (default: 0.5)')
    parser.add_argument('--color', type=str, default='b',
                       help='Color de las líneas (default: b). Ej: r, g, b, m, c, y, k')
    parser.add_argument('--sin-guardar', action='store_true',
                       help='No guardar los gráficos en archivos PNG')
    
    args = parser.parse_args()
    
    # Cargar Sirena 1
    print("\nCargando Sirena 1...")
    sirenas = cargar_sirenas('../data')
    
    if sirenas['sirena1'] is None:
        print("Error: No se pudo cargar Sirena 1")
        sys.exit(1)
    
    fs = sirenas['sirena1']['fs']
    data = sirenas['sirena1']['data']
    
    print(f"Sirena 1 cargada: fs={fs} Hz, duración={len(data)/fs:.2f}s\n")
    
    # Determinar carpeta de salida
    if args.sin_guardar:
        carpeta_salida = None
    else:
        carpeta_salida = '../graficos-creados'
    
    # Generar gráficos
    resultados = generar_fft_ventanas(
        fs, data, 
        tamaño_ventana_s=args.tamaño,
        nombre_sirena='Sirena1',
        color=args.color,
        carpeta_salida=carpeta_salida
    )
    
    # Imprimir resumen
    if carpeta_salida:
        print(f"Los gráficos se encuentran en: {os.path.abspath(carpeta_salida)}/")


if __name__ == '__main__':
    main()
