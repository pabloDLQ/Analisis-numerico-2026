import cv2
import numpy as np
from src.cargar_imagenes_jpg import cargar_imagen


def calcular_vector_traslacion_fourier(num_imagen1, num_imagen2):
    """
    Calcula el vector de traslación entre dos imágenes usando espectro cruzado normalizado.
    
    Utiliza transformadas de Fourier para detectar el desplazamiento mediante:
    1. FFT de ambas imágenes
    2. Cálculo del espectro cruzado normalizado
    3. Localización del pico de correlación
    
    Args:
        num_imagen1 (int): Número de la primera imagen (1-7)
        num_imagen2 (int): Número de la segunda imagen (1-7)
    
    Returns:
        dict: Diccionario con las siguientes claves:
            - 'dx': Desplazamiento en eje X
            - 'dy': Desplazamiento en eje Y
            - 'vector': Tupla (dx, dy)
            - 'confianza': Métrica de confianza basada en el pico de correlación (0-1)
            - 'pico_correlacion': Valor del pico máximo encontrado
    """
    
    # Cargar imágenes
    img1 = cargar_imagen(num_imagen1)
    img2 = cargar_imagen(num_imagen2)
    
    # Convertir a escala de grises
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Obtener dimensiones
    height, width = gray1.shape
    
    # Calcular transformadas de Fourier
    fft1 = np.fft.fft2(gray1)
    fft2 = np.fft.fft2(gray2)
    
    # Calcular el espectro cruzado
    espectro_cruzado = fft1 * np.conj(fft2)
    
    # Normalizar el espectro cruzado
    magnitud_espectro = np.abs(espectro_cruzado)
    magnitud_espectro[magnitud_espectro == 0] = 1  # Evitar división por cero
    
    espectro_cruzado_normalizado = espectro_cruzado / magnitud_espectro
    
    # Transformada inversa de Fourier para obtener la correlación
    correlacion = np.fft.ifft2(espectro_cruzado_normalizado)
    correlacion = np.abs(correlacion)
    
    # Encontrar el pico de correlación
    dy_pico, dx_pico = np.unravel_index(np.argmax(correlacion), correlacion.shape)
    
    # Convertir a desplazamiento real (tener en cuenta la periodicidad de FFT)
    # Si el desplazamiento es mayor que la mitad de la dimensión, restar la dimensión
    if dy_pico > height / 2:
        dy_pico = dy_pico - height
    if dx_pico > width / 2:
        dx_pico = dx_pico - width
    
    # Calcular confianza basada en el valor del pico
    pico_max = np.max(correlacion)
    
    # Calcular la media de correlación para normalizarla
    media_correlacion = np.mean(correlacion)
    
    # Confianza: razón entre el pico y la media (normalizada)
    if media_correlacion > 0:
        confianza = min(1.0, pico_max / (media_correlacion * 10))
    else:
        confianza = 0.0
    
    return {
        'dx': float(dx_pico),
        'dy': float(dy_pico),
        'vector': (float(dx_pico), float(dy_pico)),
        'confianza': float(confianza),
        'pico_correlacion': float(pico_max),
        'metodo': 'Espectro Cruzado Normalizado (FFT)'
    }


if __name__ == "__main__":
    # Prueba: calcular traslación entre imagen 1 y 2
    resultado = calcular_vector_traslacion_fourier(1, 2)
    print(f"Vector de traslación (imagen 1 → imagen 2):")
    print(f"  dx: {resultado['dx']:.2f} píxeles")
    print(f"  dy: {resultado['dy']:.2f} píxeles")
    print(f"  Pico de correlación: {resultado['pico_correlacion']:.4f}")
    print(f"  Confianza: {resultado['confianza']:.2%}")
    print(f"  Método: {resultado['metodo']}")
