from src.cargar_sirenas import cargar_sirenas
from src.graficador import graficar_comparacion_seniales, graficar_tiempo_frecuencia
from src.filtro_paso_banda import aplicar_filtro_paso_banda
from src.generar_FFT_temporales import generar_fft_ventanas
from src.calcular_vel_doppler import main as calcular_vel_doppler_main
from src.analizar_espectrograma import main as analizar_espectrograma_main
import os

def menu_principal():
    #comment
    """Muestra el menú principal y retorna la opción seleccionada."""
    print("\n" + "="*70)
    print("ANALISIS DE SENIALES DE SIRENAS - MENU PRINCIPAL")
    print("="*70)
    print("\n1. Analizar Sirenas (Filtrado y gráficos básicos)")
    print("2. Generar FFT en Ventanas Temporales")
    print("3. Análisis Doppler (Calcular velocidad ambulancia)")
    print("4. Generar Espectrogramas")
    print("5. Salir")
    print("\n" + "-"*70)
    
    while True:
        try:
            opcion = input("\nSelecciona una opción (1-5): ").strip()
            if opcion in ['1', '2', '3', '4', '5']:
                return opcion
            else:
                print("Opción inválida. Intenta de nuevo.")
        except KeyboardInterrupt:
            print("\n\nPrograma cancelado.")
            return '5'


def analizar_fft_ventanas_menu(sirenas):
    """Menú interactivo para análisis FFT en ventanas temporales."""
    print("\n" + "="*70)
    print("GENERAR FFT EN VENTANAS TEMPORALES")
    print("="*70)
    
    # Seleccionar sirena
    print("\nSelecciona la sirena a analizar:")
    print("  1. Sirena 1")
    print("  2. Sirena 2")
    
    while True:
        try:
            opcion_sirena = input("\nOpción (1 o 2): ").strip()
            if opcion_sirena == '1':
                if sirenas['sirena1'] is None:
                    print("Error: Sirena 1 no disponible.")
                    return
                fs = sirenas['sirena1']['fs']
                data = sirenas['sirena1']['data']
                nombre_sirena = 'Sirena1'
                break
            elif opcion_sirena == '2':
                if sirenas['sirena2'] is None:
                    print("Error: Sirena 2 no disponible.")
                    return
                fs = sirenas['sirena2']['fs']
                data = sirenas['sirena2']['data']
                nombre_sirena = 'Sirena2'
                break
            else:
                print("Opción inválida. Intenta de nuevo.")
        except KeyboardInterrupt:
            print("\n\nOperación cancelada.")
            return
    
    # Seleccionar tamaño de ventana
    print("\nSelecciona el tamaño de ventana temporal (en segundos):")
    print("  1. 0.25 segundos (32 ventanas)")
    print("  2. 0.5 segundos (16 ventanas)")
    print("  3. 1.0 segundos (8 ventanas)")
    print("  4. Personalizado")
    
    while True:
        try:
            opcion_ventana = input("\nOpción (1-4): ").strip()
            if opcion_ventana == '1':
                tamaño_ventana_s = 0.25
                break
            elif opcion_ventana == '2':
                tamaño_ventana_s = 0.5
                break
            elif opcion_ventana == '3':
                tamaño_ventana_s = 1.0
                break
            elif opcion_ventana == '4':
                try:
                    tamaño_str = input("Ingresa el tamaño en segundos: ").strip()
                    tamaño_ventana_s = float(tamaño_str)
                    if tamaño_ventana_s <= 0 or tamaño_ventana_s > len(data) / fs:
                        print(f"Error: El tamaño debe estar entre 0 y {len(data) / fs:.2f} segundos.")
                        continue
                    break
                except ValueError:
                    print("Entrada inválida. Intenta de nuevo.")
                    continue
            else:
                print("Opción inválida. Intenta de nuevo.")
        except KeyboardInterrupt:
            print("\n\nOperación cancelada.")
            return
    
    # Generar los gráficos
    print(f"\n[INFO] Generando FFT para {nombre_sirena} con ventanas de {tamaño_ventana_s}s...")
    
    carpeta_graficos = os.path.join(os.path.dirname(__file__), "graficos-creados")
    
    # Seleccionar color según la sirena
    color = 'r' if opcion_sirena == '2' else 'b'
    
    generar_fft_ventanas(
        fs, data,
        tamaño_ventana_s=tamaño_ventana_s,
        nombre_sirena=nombre_sirena,
        color=color,
        carpeta_salida=carpeta_graficos
    )
    
    print(f"\n[OK] FFT generado exitosamente.")
    print(f"Los gráficos se encuentran en: {os.path.abspath(carpeta_graficos)}/")


def analizar_doppler_menu(sirenas):
    """Menú interactivo para análisis Doppler (cálculo de velocidad)."""
    print("\n" + "="*70)
    print("ANÁLISIS DOPPLER - CALCULAR VELOCIDAD DE AMBULANCIA")
    print("="*70)
    
    # Seleccionar sirena
    print("\nSelecciona la sirena a analizar:")
    print("  1. Sirena 1")
    print("  2. Sirena 2")
    
    while True:
        try:
            opcion_sirena = input("\nOpción (1 o 2): ").strip()
            if opcion_sirena == '1':
                if sirenas['sirena1'] is None:
                    print("Error: Sirena 1 no disponible.")
                    return
                numero_sirena = 1
                break
            elif opcion_sirena == '2':
                if sirenas['sirena2'] is None:
                    print("Error: Sirena 2 no disponible.")
                    return
                numero_sirena = 2
                break
            else:
                print("Opción inválida. Intenta de nuevo.")
        except KeyboardInterrupt:
            print("\n\nOperación cancelada.")
            return
    
    # Seleccionar tamaño de ventana
    print("\nSelecciona el tamaño de ventana temporal (en segundos):")
    print("  1. 0.25 segundos")
    print("  2. 0.5 segundos")
    print("  3. 1.0 segundos")
    print("  4. Personalizado")
    
    while True:
        try:
            opcion_ventana = input("\nOpción (1-4): ").strip()
            if opcion_ventana == '1':
                tamaño_ventana = 0.25
                break
            elif opcion_ventana == '2':
                tamaño_ventana = 0.5
                break
            elif opcion_ventana == '3':
                tamaño_ventana = 1.0
                break
            elif opcion_ventana == '4':
                try:
                    tamaño_str = input("Ingresa el tamaño en segundos: ").strip()
                    tamaño_ventana = float(tamaño_str)
                    if tamaño_ventana <= 0:
                        print("Error: El tamaño debe ser positivo.")
                        continue
                    break
                except ValueError:
                    print("Entrada inválida. Intenta de nuevo.")
                    continue
            else:
                print("Opción inválida. Intenta de nuevo.")
        except KeyboardInterrupt:
            print("\n\nOperación cancelada.")
            return
    
    # Ejecutar análisis Doppler con los parámetros seleccionados
    print(f"\n[INFO] Calculando velocidad Doppler para Sirena {numero_sirena}")
    calcular_vel_doppler_main(numero_sirena=numero_sirena, tamaño_ventana=tamaño_ventana)


def analizar_espectrograma_menu(sirenas):
    """Menú interactivo para generación de espectrogramas."""
    print("\n" + "="*70)
    print("GENERAR ESPECTROGRAMAS")
    print("="*70)
    
    # Seleccionar sirena
    print("\nSelecciona la sirena a analizar:")
    print("  1. Sirena 1")
    print("  2. Sirena 2")
    
    while True:
        try:
            opcion_sirena = input("\nOpción (1 o 2): ").strip()
            if opcion_sirena == '1':
                if sirenas['sirena1'] is None:
                    print("Error: Sirena 1 no disponible.")
                    return
                numero_sirena = 1
                break
            elif opcion_sirena == '2':
                if sirenas['sirena2'] is None:
                    print("Error: Sirena 2 no disponible.")
                    return
                numero_sirena = 2
                break
            else:
                print("Opción inválida. Intenta de nuevo.")
        except KeyboardInterrupt:
            print("\n\nOperación cancelada.")
            return
    
    # Seleccionar tamaño de ventana STFT
    print("\nSelecciona el tamaño de ventana STFT (en segundos):")
    print("  1. 0.1 segundos")
    print("  2. 0.25 segundos")
    print("  3. 0.5 segundos")
    print("  4. 1.0 segundos")
    print("  5. Personalizado")
    
    while True:
        try:
            opcion_ventana = input("\nOpción (1-5): ").strip()
            if opcion_ventana == '1':
                tamaño_ventana = 0.1
                break
            elif opcion_ventana == '2':
                tamaño_ventana = 0.25
                break
            elif opcion_ventana == '3':
                tamaño_ventana = 0.5
                break
            elif opcion_ventana == '4':
                tamaño_ventana = 1.0
                break
            elif opcion_ventana == '5':
                try:
                    tamaño_str = input("Ingresa el tamaño en segundos: ").strip()
                    tamaño_ventana = float(tamaño_str)
                    if tamaño_ventana <= 0:
                        print("Error: El tamaño debe ser positivo.")
                        continue
                    break
                except ValueError:
                    print("Entrada inválida. Intenta de nuevo.")
                    continue
            else:
                print("Opción inválida. Intenta de nuevo.")
        except KeyboardInterrupt:
            print("\n\nOperación cancelada.")
            return
    
    # Ejecutar análisis de espectrograma con los parámetros seleccionados
    print(f"\n[INFO] Generando espectrograma para Sirena {numero_sirena} con ventana de {tamaño_ventana}s...")
    analizar_espectrograma_main(numero_sirena=numero_sirena, tamaño_ventana_s=tamaño_ventana)


def main():
    # Crear la carpeta de graficos si no existe
    carpeta_graficos = os.path.join(os.path.dirname(__file__), "graficos-creados")
    if not os.path.exists(carpeta_graficos):
        os.makedirs(carpeta_graficos)
    
    # Cargar las sirenas
    sirenas = cargar_sirenas()

    # Verificar que ambas se cargaron correctamente
    if sirenas['sirena1'] is None or sirenas['sirena2'] is None:
        print("Error: no se pudieron cargar los archivos.")
        return

    # Extraer datos
    fs1, data1 = sirenas['sirena1']['fs'], sirenas['sirena1']['data']
    fs2, data2 = sirenas['sirena2']['fs'], sirenas['sirena2']['data']

    # Mostrar informacion basica
    print("\n--- Informacion de las senales ---")
    print(f"Sirena 1: fs = {fs1} Hz, duracion = {len(data1)/fs1:.2f} s")
    print(f"Sirena 2: fs = {fs2} Hz, duracion = {len(data2)/fs2:.2f} s")
    print("\n" + "="*70 + "\n")
    
    # Menú principal
    while True:
        opcion = menu_principal()
        
        if opcion == '1':
            # ============================================================
            # FILTRADO PASO BANDA - AUTOMÁTICO
            # ============================================================
            print("CONFIGURACION DEL FILTRO PASO BANDA")
            print("="*70)
            
            # Filtros predeterminados
            freq_min1 = 940.0
            freq_max1 = 1050.0
            freq_min2 = 1200.0
            freq_max2 = 1500.0
            
            print(f"\n[OK] Filtro Sirena 1: {freq_min1:.1f} Hz - {freq_max1:.1f} Hz (automatico)")
            print(f"[OK] Filtro Sirena 2: {freq_min2:.1f} Hz - {freq_max2:.1f} Hz (automatico)")
            
            # Aplicar filtros independientes a cada sirena
            print(f"\nAplicando filtros...")
            data1_filtrada = aplicar_filtro_paso_banda(data1, fs1, freq_min1, freq_max1)
            data2_filtrada = aplicar_filtro_paso_banda(data2, fs2, freq_min2, freq_max2)
            print(f"[OK] Filtros aplicados")
            
            print("\n" + "="*70 + "\n")
            
            # ============================================================
            # GRAFICOS
            # ============================================================
            print("\n" + "="*70)
            print("GENERANDO GRAFICOS")
            print("="*70)
            
            # Graficar seniales originales
            archivo_sirena1_original = os.path.join(carpeta_graficos, "Sirena1_Original.png")
            archivo_sirena2_original = os.path.join(carpeta_graficos, "Sirena2_Original.png")
            
            graficar_tiempo_frecuencia(fs1, data1, 
                                       titulo="Sirena 1 - Original",
                                       color='blue', 
                                       archivo_salida=archivo_sirena1_original)
            
            graficar_tiempo_frecuencia(fs2, data2, 
                                       titulo="Sirena 2 - Original",
                                       color='red', 
                                       archivo_salida=archivo_sirena2_original)
            
            # Graficar seniales filtradas
            archivo_sirena1_filtrada = os.path.join(carpeta_graficos, "Sirena1_Filtrada.png")
            archivo_sirena2_filtrada = os.path.join(carpeta_graficos, "Sirena2_Filtrada.png")
            
            graficar_tiempo_frecuencia(fs1, data1_filtrada, 
                                       titulo="Sirena 1 - Filtro Paso Banda",
                                       color='blue', 
                                       archivo_salida=archivo_sirena1_filtrada)
            
            graficar_tiempo_frecuencia(fs2, data2_filtrada, 
                                       titulo="Sirena 2 - Filtro Paso Banda",
                                       color='red', 
                                       archivo_salida=archivo_sirena2_filtrada)
            
            print(f"\n[OK] Los graficos han sido guardados en la carpeta '{carpeta_graficos}'")
            print(f"\nCada archivo contiene:")
            print(f"  - Grafico en Dominio del Tiempo")
            print(f"  - FFT Completa (0 - Frecuencia Nyquist)")
            print(f"  - FFT Acotada [0 - 5000 Hz]")
            print(f"\nArchivos generados:")
            print(f"  - {archivo_sirena1_original}")
            print(f"  - {archivo_sirena2_original}")
            print(f"  - {archivo_sirena1_filtrada}")
            print(f"  - {archivo_sirena2_filtrada}")
        
        elif opcion == '2':
            analizar_fft_ventanas_menu(sirenas)
        
        elif opcion == '3':
            analizar_doppler_menu(sirenas)
        
        elif opcion == '4':
            analizar_espectrograma_menu(sirenas)
        
        elif opcion == '5':
            print("\n[OK] Programa finalizado.")
            break


if __name__ == "__main__":
    main()