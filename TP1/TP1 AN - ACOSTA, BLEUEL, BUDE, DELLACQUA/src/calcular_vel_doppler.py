"""
Script para generar gráficos FFT en ventanas temporales y analizar efecto Doppler.
Ejecutar: python -m src.analizar_item2
"""

from src.cargar_sirenas import cargar_sirenas
from src.analisis_doppler import (
    analizar_fft_ventanas_temporales, 
    calcular_velocidad_ambulancia, 
    calcular_velocidades_por_ventana,
    calcular_velocidades_metodo2,
    obtener_datos_espectrograma
)
import numpy as np
import os


def menu_seleccionar_metodo():
    """
    Muestra menú interactivo para seleccionar método de cálculo.
    
    Retorna:
    --------
    int : 1 para Método 1, 2 para Método 2
    """
    print("\n" + "="*70)
    print("SELECCIONAR MÉTODO DE CÁLCULO DE VELOCIDAD DOPPLER")
    print("="*70)
    print("""
MÉTODO 1 (Original):
  - Usa frecuencia mínima y máxima del conjunto completo de ventanas
  - Asume acercamiento en freq_max y alejamiento en freq_min
  - Menos preciso pero simple
  
MÉTODO 2 (Espectrograma):
  - Usa frecuencia real (f₀) calculada por el espectrograma
  - Clasifica cada ventana por comparación con f₀
  - Usa instante de paso del espectrograma
  - Más preciso pero requiere cálculo previo del espectrograma
""")
    
    while True:
        try:
            opcion = input("\nSelecciona el método (1 o 2): ").strip()
            if opcion in ['1', '2']:
                return int(opcion)
            else:
                print("Opción inválida. Intenta de nuevo.")
        except KeyboardInterrupt:
            print("\n\nOperación cancelada.")
            return None


def metodo_1_doppler(fs, data, nombre_sirena, numero_sirena, carpeta_graficos, tamaño_ventana):
    """
    MÉTODO 1: Implementación original usando min/max de frecuencias.
    
    Parámetros:
    -----------
    fs : float
        Frecuencia de muestreo
    data : ndarray
        Array con las muestras de audio
    nombre_sirena : str
        Nombre de la sirena
    numero_sirena : int
        Número de sirena
    carpeta_graficos : str
        Ruta de carpeta de gráficos
    tamaño_ventana : float
        Tamaño de ventana en segundos
    """
    
    archivo_ventanas = os.path.join(carpeta_graficos, f"{nombre_sirena.replace(' ', '')}_FFT_Ventanas_{tamaño_ventana}s.png")
    
    print(f"\nGenerando gráfico FFT en ventanas de {tamaño_ventana} segundos...")
    info_ventanas = analizar_fft_ventanas_temporales(
        fs, data,
        titulo=f"{nombre_sirena} - FFT en Ventanas Temporales",
        tamaño_ventana_s=tamaño_ventana,
        color='blue',
        archivo_salida=archivo_ventanas
    )
    
    # Análisis de variación de frecuencia
    print("\n" + "-"*70)
    print("ANÁLISIS DE CAMBIO DE FRECUENCIA EN EL TIEMPO")
    print("-"*70)
    print(f"\n{'Ventana':<10} {'Tiempo (s)':<20} {'Frecuencia Pico (Hz)':<25}")
    print("-" * 55)
    
    frecuencias_pico = []
    for info in info_ventanas:
        print(f"{info['ventana']+1:<10} {info['tiempo_inicio']:.2f}-{info['tiempo_fin']:.2f}s{'':<8} {info['freq_pico']:.2f}")
        frecuencias_pico.append(info['freq_pico'])
    
    # Estadísticas
    freq_min = np.min(frecuencias_pico)
    freq_max = np.max(frecuencias_pico)
    freq_promedio = np.mean(frecuencias_pico)
    variacion = freq_max - freq_min
    
    print("\n" + "-"*70)
    print(f"Frecuencia mínima: {freq_min:.2f} Hz")
    print(f"Frecuencia máxima: {freq_max:.2f} Hz")
    print(f"Frecuencia promedio: {freq_promedio:.2f} Hz")
    print(f"Variación total: {variacion:.2f} Hz ({(variacion/freq_promedio)*100:.1f}%)")
    
    # Explicar efecto Doppler
    print("\n" + "="*70)
    print("EFECTO FÍSICO OBSERVADO: EFECTO DOPPLER")
    print("="*70)
    print(f"""
La variación de frecuencia de {variacion:.2f} Hz indica el efecto Doppler.

¿Qué está ocurriendo?
- Cuando la ambulancia SE ACERCA: la frecuencia AUMENTA a {freq_max:.2f} Hz
- Cuando la ambulancia SE ALEJA: la frecuencia DISMINUYE a {freq_min:.2f} Hz

Esto ocurre porque:
1. Las ondas sonoras se comprimen cuando la fuente se acerca
2. Las ondas sonoras se expanden cuando la fuente se aleja
3. El observador percibe frecuencias diferentes según el movimiento
""")
    
    # Calcular velocidades por ventana
    print("="*70)
    print("CÁLCULO DE VELOCIDAD DE LA AMBULANCIA - MÉTODO 1")
    print("="*70)
    
    freq_promedio = np.mean(frecuencias_pico)
    v_sonido = 343
    velocidades_acerca, velocidades_aleja = calcular_velocidades_por_ventana(frecuencias_pico, freq_promedio)
    
    print(f"""
Fórmula de Doppler utilizada:
- Acercamiento: f_pico = f₀ × v_sonido / (v_sonido - v_ambulancia)
- Alejamiento: f_pico = f₀ × v_sonido / (v_sonido + v_ambulancia)

Parámetros:
- Velocidad del sonido: {v_sonido} m/s (a 20°C)
- Frecuencia de reposo (f₀): {freq_promedio:.2f} Hz (PROMEDIO)
""")
    
    print("\n" + "-"*70)
    print("VELOCIDADES DE ACERCAMIENTO (por ventana)")
    print("-"*70)
    print(f"{'Ventana':<10} {'Tiempo':<20} {'Vel. Acerca (m/s)':<20} {'Vel. Acerca (km/h)':<20}")
    print("-" * 70)
    
    velocidades_acerca_validas = []
    for i, (v_a, info) in enumerate(zip(velocidades_acerca, info_ventanas)):
        if v_a is not None:
            v_a_kmh = v_a * 3.6
            print(f"{i+1:<10} {info['tiempo_inicio']:.2f}-{info['tiempo_fin']:.2f}s{'':<8} {v_a:>18.2f} {v_a_kmh:>18.2f}")
            velocidades_acerca_validas.append(v_a)
    
    if velocidades_acerca_validas:
        v_acerca_promedio_ms = np.mean(velocidades_acerca_validas)
        v_acerca_promedio_kmh = v_acerca_promedio_ms * 3.6
        print("-" * 70)
        print(f"{'PROMEDIO ACERCAMIENTO':<40} {v_acerca_promedio_ms:>17.2f} {v_acerca_promedio_kmh:>18.2f}")
    else:
        v_acerca_promedio_ms = 0
        v_acerca_promedio_kmh = 0
    
    print("\n" + "-"*70)
    print("VELOCIDADES DE ALEJAMIENTO (por ventana)")
    print("-"*70)
    print(f"{'Ventana':<10} {'Tiempo':<20} {'Vel. Aleja (m/s)':<20} {'Vel. Aleja (km/h)':<20}")
    print("-" * 70)
    
    velocidades_aleja_validas = []
    for i, (v_al, info) in enumerate(zip(velocidades_aleja, info_ventanas)):
        if v_al is not None:
            v_al_kmh = v_al * 3.6
            print(f"{i+1:<10} {info['tiempo_inicio']:.2f}-{info['tiempo_fin']:.2f}s{'':<8} {v_al:>18.2f} {v_al_kmh:>18.2f}")
            velocidades_aleja_validas.append(v_al)
    
    if velocidades_aleja_validas:
        v_aleja_promedio_ms = np.mean(velocidades_aleja_validas)
        v_aleja_promedio_kmh = v_aleja_promedio_ms * 3.6
        print("-" * 70)
        print(f"{'PROMEDIO ALEJAMIENTO':<40} {v_aleja_promedio_ms:>17.2f} {v_aleja_promedio_kmh:>18.2f}")
    else:
        v_aleja_promedio_ms = 0
        v_aleja_promedio_kmh = 0
    
    # Promedios finales
    print("\n" + "="*70)
    print("RESUMEN DE VELOCIDADES PROMEDIO - MÉTODO 1")
    print("="*70)
    print(f"Velocidad promedio ACERCAMIENTO: {v_acerca_promedio_ms:.2f} m/s ({v_acerca_promedio_kmh:.2f} km/h)")
    print(f"Velocidad promedio ALEJAMIENTO:  {v_aleja_promedio_ms:.2f} m/s ({v_aleja_promedio_kmh:.2f} km/h)")
    
    v_global_promedio_ms = (v_acerca_promedio_ms + v_aleja_promedio_ms) / 2
    v_global_promedio_kmh = v_global_promedio_ms * 3.6
    
    # Resultado final destacado
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + f"  MÉTODO 1 - VELOCIDAD DE LA AMBULANCIA: {v_global_promedio_kmh:.2f} km/h ({v_global_promedio_ms:.2f} m/s)".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)


def metodo_2_doppler(fs, data, nombre_sirena, numero_sirena, carpeta_graficos, tamaño_ventana):
    """
    MÉTODO 2: Usa frecuencia real del espectrograma e instante de paso.
    
    Parámetros:
    -----------
    fs : float
        Frecuencia de muestreo
    data : ndarray
        Array con las muestras de audio
    nombre_sirena : str
        Nombre de la sirena
    numero_sirena : int
        Número de sirena
    carpeta_graficos : str
        Ruta de carpeta de gráficos
    tamaño_ventana : float
        Tamaño de ventana en segundos
    """
    
    print("\n" + "="*70)
    print("CÁLCULO DE VELOCIDAD DE LA AMBULANCIA - MÉTODO 2")
    print("="*70)
    
    # Obtener datos del espectrograma
    print(f"\nObteniendo datos del espectrograma...")
    datos_espectrograma = obtener_datos_espectrograma(fs, data, numero_sirena, tamaño_ventana)
    
    freq_real = datos_espectrograma['frecuencia_real_hz']
    instante_paso = datos_espectrograma['instante_paso_s']
    t_freq = datos_espectrograma['t_freq']
    frecuencias_instantaneas = datos_espectrograma['frecuencias_instantaneas']
    
    print(f"[OK] Frecuencia real estimada: {freq_real:.2f} Hz")
    print(f"[OK] Instante de paso: {instante_paso:.4f} s")
    
    # Generar análisis FFT en ventanas
    archivo_ventanas = os.path.join(carpeta_graficos, f"{nombre_sirena.replace(' ', '')}_FFT_Ventanas_M2_{tamaño_ventana}s.png")
    
    print(f"\nGenerando gráfico FFT en ventanas de {tamaño_ventana} segundos...")
    info_ventanas = analizar_fft_ventanas_temporales(
        fs, data,
        titulo=f"{nombre_sirena} - FFT en Ventanas Temporales (Método 2)",
        tamaño_ventana_s=tamaño_ventana,
        color='blue',
        archivo_salida=archivo_ventanas
    )
    
    # Análisis de variación de frecuencia
    print("\n" + "-"*70)
    print("ANÁLISIS DE CAMBIO DE FRECUENCIA EN EL TIEMPO")
    print("-"*70)
    print(f"\n{'Ventana':<10} {'Tiempo (s)':<20} {'Frecuencia Pico (Hz)':<25}")
    print("-" * 55)
    
    frecuencias_pico = []
    for info in info_ventanas:
        print(f"{info['ventana']+1:<10} {info['tiempo_inicio']:.2f}-{info['tiempo_fin']:.2f}s{'':<8} {info['freq_pico']:.2f}")
        frecuencias_pico.append(info['freq_pico'])
    
    # Estadísticas
    freq_min = np.min(frecuencias_pico)
    freq_max = np.max(frecuencias_pico)
    freq_promedio = np.mean(frecuencias_pico)
    variacion = freq_max - freq_min
    
    print("\n" + "-"*70)
    print(f"Frecuencia mínima detectada: {freq_min:.2f} Hz")
    print(f"Frecuencia máxima detectada: {freq_max:.2f} Hz")
    print(f"Frecuencia promedio (todas):  {freq_promedio:.2f} Hz")
    print(f"Frecuencia real (f₀):         {freq_real:.2f} Hz (del espectrograma)")
    print(f"Variación total: {variacion:.2f} Hz ({(variacion/freq_promedio)*100:.1f}%)")
    
    # Calcular velocidades con Método 2
    v_sonido = 343
    velocidades_acerca, velocidades_aleja, clasificaciones = calcular_velocidades_metodo2(
        frecuencias_pico, info_ventanas, freq_real, instante_paso,
        t_freq=t_freq, frecuencias_instantaneas=frecuencias_instantaneas
    )
    
    print(f"""
Fórmula de Doppler utilizada:
- Acercamiento: f_pico = f₀ × v_sonido / (v_sonido - v_ambulancia)
- Alejamiento: f_pico = f₀ × v_sonido / (v_sonido + v_ambulancia)

Parámetros:
- Velocidad del sonido: {v_sonido} m/s (a 20°C)
- Frecuencia de reposo (f₀): {freq_real:.2f} Hz (del espectrograma)
- Instante de paso más cercano: {instante_paso:.4f} s
""")
    
    print("\n" + "-"*70)
    print("VELOCIDADES DE ACERCAMIENTO (por ventana)")
    print("-"*70)
    print(f"{'Ventana':<10} {'Tiempo':<20} {'Clasificación':<25} {'Vel. Acerca (m/s)':<20} {'Vel. (km/h)':<15}")
    print("-" * 90)
    
    velocidades_acerca_validas = []
    for i, (v_a, info, clasificacion) in enumerate(zip(velocidades_acerca, info_ventanas, clasificaciones)):
        if v_a is not None:
            v_a_kmh = v_a * 3.6
            print(f"{i+1:<10} {info['tiempo_inicio']:.2f}-{info['tiempo_fin']:.2f}s{'':<8} {clasificacion:<25} {v_a:>18.2f} {v_a_kmh:>14.2f}")
            velocidades_acerca_validas.append(v_a)
    
    if velocidades_acerca_validas:
        v_acerca_promedio_ms = np.mean(velocidades_acerca_validas)
        v_acerca_promedio_kmh = v_acerca_promedio_ms * 3.6
        print("-" * 90)
        print(f"{'PROMEDIO ACERCAMIENTO':<40} {'':<25} {v_acerca_promedio_ms:>18.2f} {v_acerca_promedio_kmh:>14.2f}")
    else:
        v_acerca_promedio_ms = 0
        v_acerca_promedio_kmh = 0
    
    print("\n" + "-"*70)
    print("VELOCIDADES DE ALEJAMIENTO (por ventana)")
    print("-"*70)
    print(f"{'Ventana':<10} {'Tiempo':<20} {'Clasificación':<25} {'Vel. Aleja (m/s)':<20} {'Vel. (km/h)':<15}")
    print("-" * 90)
    
    velocidades_aleja_validas = []
    for i, (v_al, info, clasificacion) in enumerate(zip(velocidades_aleja, info_ventanas, clasificaciones)):
        if v_al is not None:
            v_al_kmh = v_al * 3.6
            print(f"{i+1:<10} {info['tiempo_inicio']:.2f}-{info['tiempo_fin']:.2f}s{'':<8} {clasificacion:<25} {v_al:>18.2f} {v_al_kmh:>14.2f}")
            velocidades_aleja_validas.append(v_al)
    
    if velocidades_aleja_validas:
        v_aleja_promedio_ms = np.mean(velocidades_aleja_validas)
        v_aleja_promedio_kmh = v_aleja_promedio_ms * 3.6
        print("-" * 90)
        print(f"{'PROMEDIO ALEJAMIENTO':<40} {'':<25} {v_aleja_promedio_ms:>18.2f} {v_aleja_promedio_kmh:>14.2f}")
    else:
        v_aleja_promedio_ms = 0
        v_aleja_promedio_kmh = 0
    
    # Promedios finales
    print("\n" + "="*70)
    print("RESUMEN DE VELOCIDADES PROMEDIO - MÉTODO 2")
    print("="*70)
    print(f"Velocidad promedio ACERCAMIENTO: {v_acerca_promedio_ms:.2f} m/s ({v_acerca_promedio_kmh:.2f} km/h)")
    print(f"Velocidad promedio ALEJAMIENTO:  {v_aleja_promedio_ms:.2f} m/s ({v_aleja_promedio_kmh:.2f} km/h)")
    
    v_global_promedio_ms = (v_acerca_promedio_ms + v_aleja_promedio_ms) / 2
    v_global_promedio_kmh = v_global_promedio_ms * 3.6
    
    # Resultado final destacado
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + f"  MÉTODO 2 - VELOCIDAD DE LA AMBULANCIA: {v_global_promedio_kmh:.2f} km/h ({v_global_promedio_ms:.2f} m/s)".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)


def main(numero_sirena=1, tamaño_ventana=0.5, metodo=None):
    """
    Calcula la velocidad Doppler de una sirena.
    
    Parámetros:
    -----------
    numero_sirena : int, default 1
        Número de sirena a analizar (1 o 2)
    tamaño_ventana : float, default 0.5
        Tamaño de la ventana temporal en segundos
    metodo : int, optional
        Método a usar (1 o 2). Si es None, muestra menú interactivo.
    """
    # Validar entrada
    if numero_sirena not in [1, 2]:
        print("Error: numero_sirena debe ser 1 o 2")
        return
    
    if tamaño_ventana <= 0:
        print("Error: tamaño_ventana debe ser positivo")
        return
    
    # Crear carpeta de gráficos
    carpeta_graficos = os.path.join(os.path.dirname(__file__), "..", "graficos-creados")
    if not os.path.exists(carpeta_graficos):
        os.makedirs(carpeta_graficos)
    
    # Cargar sirenas
    sirenas = cargar_sirenas()
    
    # Seleccionar sirena según parámetro
    if numero_sirena == 1:
        if sirenas['sirena1'] is None:
            print("Error: No se pudo cargar Sirena 1")
            return
        fs = sirenas['sirena1']['fs']
        data = sirenas['sirena1']['data']
        nombre_sirena = "Sirena 1"
    else:  # numero_sirena == 2
        if sirenas['sirena2'] is None:
            print("Error: No se pudo cargar Sirena 2")
            return
        fs = sirenas['sirena2']['fs']
        data = sirenas['sirena2']['data']
        nombre_sirena = "Sirena 2"
    
    print("="*70)
    print("ITEM 2: ANÁLISIS DE FFT EN VENTANAS TEMPORALES (EFECTO DOPPLER)")
    print("="*70)
    print(f"\n{nombre_sirena}: fs = {fs} Hz, duración = {len(data)/fs:.2f} s\n")
    
    # Mostrar menú si no se especificó método
    if metodo is None:
        metodo = menu_seleccionar_metodo()
        if metodo is None:
            return
    
    # Ejecutar método seleccionado
    if metodo == 1:
        metodo_1_doppler(fs, data, nombre_sirena, numero_sirena, carpeta_graficos, tamaño_ventana)
    elif metodo == 2:
        metodo_2_doppler(fs, data, nombre_sirena, numero_sirena, carpeta_graficos, tamaño_ventana)
    else:
        print(f"Error: método debe ser 1 o 2, recibió: {metodo}")
        return

if __name__ == "__main__":
    import sys
    
    numero_sirena = 1
    tamaño_ventana = 0.5
    metodo = None
    
    # Procesar argumentos de línea de comandos
    if len(sys.argv) > 1:
        try:
            numero_sirena = int(sys.argv[1])
        except ValueError:
            print(f"Error: primer argumento debe ser un número (1 o 2), recibió: {sys.argv[1]}")
            sys.exit(1)
    
    if len(sys.argv) > 2:
        try:
            tamaño_ventana = float(sys.argv[2])
        except ValueError:
            print(f"Error: segundo argumento debe ser un número, recibió: {sys.argv[2]}")
            sys.exit(1)
    
    if len(sys.argv) > 3:
        try:
            metodo = int(sys.argv[3])
        except ValueError:
            print(f"Error: tercer argumento debe ser un número (1 o 2), recibió: {sys.argv[3]}")
            sys.exit(1)
    
    main(numero_sirena=numero_sirena, tamaño_ventana=tamaño_ventana, metodo=metodo)
