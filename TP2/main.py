from src.inciso1 import calcular_vector_traslacion_fourier
from src.crear_imagenes_cart import crear_imagen_cartesiana  

def main():
    """Programa principal - Análisis de traslación entre imágenes"""
    
    crear_imagen_cartesiana(1, 40, 10)  # Ejemplo de creación de imagen cartesiana

    print("="*60)
    print("CÁLCULO DE VECTORES DE TRASLACIÓN")
    print("="*60)
    print()
    
    # Calcular vector de traslación entre imagen 1 y 2
    print("Calculando vector de traslación entre imagen 1 y 2...")
    print("Método: Espectro Cruzado Normalizado (FFT)")
    print()
    resultado = calcular_vector_traslacion_fourier(1, 2)
    
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


if __name__ == "__main__":
    main()
