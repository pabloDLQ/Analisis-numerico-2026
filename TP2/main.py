from src.inciso1 import calcular_vector_traslacion_fourier, calcular_vector_traslacion_pixeles
from src.inciso2 import calcular_angulo_rotacion
from src.inciso3 import calcular_factor_escala
from src.crear_imagenes_polar import rotar_y_guardar_imagen
from src.crear_imagenes_cart import crear_imagen_cartesiana
from src.crear_imagenes_log_polares import crear_imagen_log_polar
import matplotlib.pyplot as plt


def mostrar_resultado(resultado):
    """Muestra el resultado del cálculo de traslación de forma formateada"""
    print()
    print("-" * 60)
    print("RESULTADO - Vector de Traslación (Imagen 1 → Imagen 2)")
    print("-" * 60)
    print(f"  Desplazamiento en X (dx): {resultado['dx']:>8.2f} píxeles")
    print(f"  Desplazamiento en Y (dy): {resultado['dy']:>8.2f} píxeles")
    print(f"  Vector de traslación:     {resultado['vector']}")
    print(f"  Pico de correlación:      {resultado['pico_correlacion']:>8.4f}")
    print(f"  Confianza:                {resultado['confianza']:>8.2%}")
    print(f"  Método:                   {resultado['metodo']}")
    print("-" * 60)
    print()


def mostrar_resultado_inciso2(resultado):
    """Muestra el resultado del cálculo de rotación de forma formateada"""
    print()
    print("-" * 60)
    print("RESULTADO - Ángulo de Rotación Detectado (Imagen 3 → Imagen 4)")
    print("-" * 60)
    print(f"  Ángulo de rotación:       {resultado['angulo_rotacion']:>8.2f}°")
    print(f"  Desplazamiento θ:         {resultado['desplazamiento_theta']:>8.2f} píxeles")
    print(f"  Pico de correlación:      {resultado['pico_correlacion']:>8.4f}")
    print(f"  Confianza:                {resultado['confianza']:>8.2%}")
    print(f"  Método:                   {resultado['metodo']}")
    print("-" * 60)
    print()


def mostrar_resultado_inciso3(resultado):
    """Muestra el resultado del cálculo de escala de forma formateada"""
    print()
    print("-" * 60)
    print("RESULTADO - Factor de Escala Detectado (Imagen 1 → Imagen 5)")
    print("-" * 60)
    print(f"  Factor de escala (espacial): {resultado.get('factor_escala', float('nan')):>8.4f}")
    print(f"  Factor de escala (espectral): {resultado.get('factor_escala_spectral', float('nan')):>8.4f}")
    print(f"  Desplazamiento log-radio:     {resultado.get('desplazamiento_log_radio', float('nan')):>8.4f}")
    print(f"  ln(k) estimado:              {resultado.get('ln_k', float('nan')):>8.4f}")
    print(f"  Pico de correlación:         {resultado.get('pico_correlacion', float('nan')):>8.4f}")
    print(f"  Confianza:                   {resultado.get('confianza', 0.0):>8.2%}")
    print(f"  Método:                      {resultado.get('metodo', '')}")
    print("-" * 60)
    print()


def menu_inciso1():
    """Menú para seleccionar el método de cálculo en el inciso 1"""
    print("\n" + "="*60)
    print("INCISO 1 - Cálculo de Vector de Traslación")
    print("="*60)
    print("\nSelecciona el método de cálculo:")
    print("  1. Transformada de Fourier (Espectro Cruzado Normalizado)")
    print("  2. Comparación directa de píxeles")
    print()
    
    opcion = input("Ingresa tu opción (1 o 2): ").strip()
    
    if opcion == "1":
        print("\nCalculando vector de traslación entre imagen 1 y 2...")
        print("Método: Espectro Cruzado Normalizado (FFT)")
        resultado = calcular_vector_traslacion_fourier(1, 2)
        mostrar_resultado(resultado)
        
    elif opcion == "2":
        print("\nCalculando vector de traslación entre imagen 1 y 2...")
        print("Método: Comparación directa de píxeles")
        print("(Esto puede tomar un momento...)")
        resultado = calcular_vector_traslacion_pixeles(1, 2)
        mostrar_resultado(resultado)
        
    else:
        print("\nOpción inválida. Por favor, ingresa 1 o 2.")
        menu_inciso1()


def menu_inciso2():
    """Menú para el inciso 2 - Detección de rotación"""
    print("\n" + "="*60)
    print("INCISO 2 - Detección de Ángulo de Rotación")
    print("="*60)
    print("\nCalculando ángulo de rotación entre imagen 3 y 4...")
    print("Método: Mapeo Polar de Espectro FFT")
    print("(Esto puede tomar un momento...)")
    print()
    
    resultado = calcular_angulo_rotacion(3, 4)
    mostrar_resultado_inciso2(resultado)


def menu_inciso3():
    """Menú para el inciso 3 - Detección de escala"""
    print("\n" + "="*60)
    print("INCISO 3 - Detección de Factor de Escala")
    print("="*60)
    print("\nCalculando factor de escala entre imagen 1 y 5...")
    print("Método: FFT + coordenadas log-polares + correlación de fase")
    print("(Esto puede tomar un momento...)")
    print()

    resultado = calcular_factor_escala(1, 5)
    mostrar_resultado_inciso3(resultado)


def menu_inciso4():
    """Menú para el inciso 4 - Coherencia de fase y recuperación de imágenes"""
    from src.inciso4 import (
        buscar_factor_k_optimo, visualizar_resultados_inciso4,
        guardar_imagenes_resultantes
    )
    
    print("\n" + "="*70)
    print("INCISO 4: RECUPERACIÓN DE IMAGEN OBJETIVO")
    print("="*70)
    print("\nProceso: Recuperar imagen usando magnitud de f6 + fase alterada de f7")
    print("Búsqueda automática del factor k óptimo para cada formato")
    print("\n" + "="*70)
    
    # PRIMERA ETAPA: Procesar TIF
    print("\n" + "="*70)
    print("ETAPA 1: ANÁLISIS CON FORMATO TIF")
    print("="*70)
    resultado_tif = buscar_factor_k_optimo(6, 7, rango_k=(0.1, 5.0), pasos=200, es_tif=True)
    
    # SEGUNDA ETAPA: Procesar JPG
    print("\n" + "="*70)
    print("ETAPA 2: ANÁLISIS CON FORMATO JPG")
    print("="*70)
    resultado_jpg = buscar_factor_k_optimo(6, 7, rango_k=(0.1, 5.0), pasos=200, es_tif=False)
    
    # TERCERA ETAPA: Generar gráficos
    print("\n" + "="*70)
    print("ETAPA 3: GENERACIÓN DE GRÁFICOS COMPARATIVOS")
    print("="*70)
    print("\nGenerando visualización de resultados...")
    fig = visualizar_resultados_inciso4(resultado_tif, resultado_jpg)
    
    guardar_imagenes_resultantes(resultado_tif, resultado_jpg)
    
    # Cerrar figuras de matplotlib sin mostrarlas
    plt.close('all')
    
    # Mostrar análisis comparativo
    print("\n" + "="*70)
    print("ANÁLISIS COMPARATIVO")
    print("="*70)
    print(f"\nFactor k óptimo (TIF):  {resultado_tif['k_optimo']:.6f}")
    print(f"Factor k óptimo (JPG):  {resultado_jpg['k_optimo']:.6f}")
    print(f"Diferencia en k:        {abs(resultado_tif['k_optimo'] - resultado_jpg['k_optimo']):.6f}")
    print(f"\nNitidez máxima (TIF):   {resultado_tif['nitidez_maxima']:.4e}")
    print(f"Nitidez máxima (JPG):   {resultado_jpg['nitidez_maxima']:.4e}")
    print(f"Diferencia en nitidez:  {abs(resultado_tif['nitidez_maxima'] - resultado_jpg['nitidez_maxima']):.4e}")
    print(f"\nObservaciones:")
    print(f"  • TIF es compresión LOSSLESS (sin pérdida)")
    print(f"  • JPG usa compresión DCT (con pérdida)")
    print(f"  • La diferencia en k afecta la recuperación de la imagen")
    print("="*70)
    print()


def menu_principal():
    """Menú principal - Selecciona el inciso a probar"""
    print("\n" + "="*60)
    print("ANÁLISIS NUMÉRICO 2026 - TP2")
    print("="*60)
    print("\nSelecciona el inciso que deseas probar:")
    print("  1. Inciso 1 - Cálculo de vectores de traslación")
    print("  2. Inciso 2 - Registro de rotación")
    print("  3. Inciso 3 - Cálculo de factor de escala")
    print("  4. Inciso 4 - Coherencia de fase y recuperación de imágenes")
    print()
    
    opcion = input("Ingresa tu opción (1, 2, 3 o 4): ").strip()
    
    if opcion == "1":
        menu_inciso1()
    elif opcion == "2":
        menu_inciso2()
    elif opcion == "3":
        menu_inciso3()
    elif opcion == "4":
        menu_inciso4()
    else:
        print("\nOpción inválida. Por favor, ingresa 1, 2, 3 o 4.")
        menu_principal()


def main():
    """Programa principal - Análisis de traslación entre imágenes"""
    crear_imagen_log_polar(1, 1.2935)

    # Mostrar menú
    menu_principal()


if __name__ == "__main__":
    main()
