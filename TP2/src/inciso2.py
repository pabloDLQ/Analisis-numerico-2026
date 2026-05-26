import cv2
import numpy as np
from scipy import ndimage
from src.cargar_imagenes_jpg import cargar_imagen


def remuestrear_a_polar(espectro_magnitud, num_radios=None, num_angulos=360):
    """
    Remuestrea un espectro de Fourier centrado en coordenadas cartesianas
    a coordenadas polares usando interpolación.
    
    Args:
        espectro_magnitud (np.ndarray): Espectro de magnitud de Fourier centrado
        num_radios (int): Número de muestras radiales (default: mitad de dimensión)
        num_angulos (int): Número de muestras angulares (default: 360)
    
    Returns:
        np.ndarray: Espectro en coordenadas polares (ρ, θ) con forma (num_radios, num_angulos)
    """
    
    height, width = espectro_magnitud.shape
    
    # Centro del espectro (después de fftshift)
    cy, cx = height / 2, width / 2
    
    # Radio máximo disponible
    radio_max = min(cy, cx)
    
    if num_radios is None:
        num_radios = int(radio_max)
    
    # Crear grilla polar
    # ρ: desde 0 hasta radio_max
    # θ: desde 0 hasta 2π
    radios = np.linspace(0, radio_max - 1, num_radios)
    angulos = np.linspace(0, 2 * np.pi, num_angulos, endpoint=False)
    
    # Meshgrid de la grilla polar
    theta_grid, rho_grid = np.meshgrid(angulos, radios)
    
    # Convertir de polares (ρ, θ) a cartesianas (x, y)
    x = cx + rho_grid * np.cos(theta_grid)
    y = cy + rho_grid * np.sin(theta_grid)
    
    # Remuestrear usando interpolación
    # Usar order=1 para interpolación bilineal
    espectro_polar = ndimage.map_coordinates(
        espectro_magnitud, 
        [y, x], 
        order=1, 
        mode='constant', 
        cval=0.0
    )
    
    return espectro_polar


def calcular_angulo_rotacion_1d(espectro_polar1, espectro_polar2, num_radios_promedio=None):
    """
    Calcula el ángulo de rotación usando correlación 1D para cada radio y promediando.
    
    Args:
        espectro_polar1 (np.ndarray): Espectro polar de la primera imagen (ρ, θ)
        espectro_polar2 (np.ndarray): Espectro polar de la segunda imagen (ρ, θ)
        num_radios_promedio (int): Número de radios a promediar (default: todos)
    
    Returns:
        dict: Diccionario con:
            - 'desplazamiento_theta': Desplazamiento en píxeles de θ
            - 'angulo_rotacion': Ángulo de rotación en grados
            - 'pico_correlacion': Valor máximo de correlación
            - 'confianza': Métrica de confianza (0-1)
    """
    
    num_radios, num_angulos = espectro_polar1.shape
    
    if num_radios_promedio is None:
        num_radios_promedio = num_radios
    else:
        num_radios_promedio = min(num_radios_promedio, num_radios)
    
    # Promediar los radios más significativos (excluyendo el centro muy cercano)
    inicio_radio = max(1, num_radios // 10)  # Comenzar desde 10% del radio máximo
    fin_radio = inicio_radio + num_radios_promedio
    
    perfil1 = np.mean(espectro_polar1[inicio_radio:fin_radio, :], axis=0)
    perfil2 = np.mean(espectro_polar2[inicio_radio:fin_radio, :], axis=0)
    
    # Normalizar
    perfil1 = (perfil1 - np.mean(perfil1)) / (np.std(perfil1) + 1e-10)
    perfil2 = (perfil2 - np.mean(perfil2)) / (np.std(perfil2) + 1e-10)
    
    # Calcular correlación cruzada
    correlacion = np.correlate(perfil1, perfil2, mode='same')
    
    # Encontrar el pico
    pico_idx = np.argmax(correlacion)
    pico_correlacion = correlacion[pico_idx]
    
    # Calcular el desplazamiento (lag)
    centro = num_angulos // 2
    desplazamiento_theta = pico_idx - centro
    
    # Si el desplazamiento es negativo, ajustar sumando num_angulos
    if desplazamiento_theta < -num_angulos / 2:
        desplazamiento_theta += num_angulos
    elif desplazamiento_theta > num_angulos / 2:
        desplazamiento_theta -= num_angulos
    
    # Convertir desplazamiento a ángulo en grados
    # desplazamiento_theta está en píxeles de θ, convertir a radianes y luego a grados
    angulo_radianes = (desplazamiento_theta / num_angulos) * 2 * np.pi
    angulo_grados = np.degrees(angulo_radianes)
    
    # Normalizar el ángulo a [0, 360)
    angulo_grados = angulo_grados % 360
    
    # Calcular confianza normalizando el pico
    media_correlacion = np.mean(correlacion)
    std_correlacion = np.std(correlacion)
    
    if std_correlacion > 0:
        confianza = min(1.0, (pico_correlacion - media_correlacion) / (3 * std_correlacion + 1e-10))
        confianza = max(0.0, confianza)
    else:
        confianza = 0.0
    
    return {
        'desplazamiento_theta': float(desplazamiento_theta),
        'angulo_rotacion': float(angulo_grados),
        'pico_correlacion': float(pico_correlacion),
        'confianza': float(confianza)
    }


def calcular_angulo_rotacion_2d(espectro_polar1, espectro_polar2):
    """
    Calcula el ángulo de rotación usando correlación 2D completa.
    
    Args:
        espectro_polar1 (np.ndarray): Espectro polar de la primera imagen (ρ, θ)
        espectro_polar2 (np.ndarray): Espectro polar de la segunda imagen (ρ, θ)
    
    Returns:
        dict: Diccionario con:
            - 'desplazamiento_theta': Desplazamiento en píxeles de θ
            - 'angulo_rotacion': Ángulo de rotación en grados
            - 'pico_correlacion': Valor máximo de correlación
            - 'confianza': Métrica de confianza (0-1)
    """
    
    num_radios, num_angulos = espectro_polar1.shape
    
    # Normalizar espectros
    esp1_norm = (espectro_polar1 - np.mean(espectro_polar1)) / (np.std(espectro_polar1) + 1e-10)
    esp2_norm = (espectro_polar2 - np.mean(espectro_polar2)) / (np.std(espectro_polar2) + 1e-10)
    
    # Correlación cruzada 2D
    correlacion_2d = ndimage.correlate(esp1_norm, esp2_norm, mode='wrap')
    
    # Buscar el máximo en la dimensión θ (eje horizontal/1)
    # Promediar sobre los radios primero
    correlacion_theta = np.mean(correlacion_2d, axis=0)
    
    # Encontrar el pico
    pico_idx = np.argmax(correlacion_theta)
    pico_correlacion = correlacion_theta[pico_idx]
    
    # Calcular el desplazamiento en θ
    centro = num_angulos // 2
    desplazamiento_theta = pico_idx - centro
    
    # Ajustar si es necesario
    if desplazamiento_theta < -num_angulos / 2:
        desplazamiento_theta += num_angulos
    elif desplazamiento_theta > num_angulos / 2:
        desplazamiento_theta -= num_angulos
    
    # Convertir a ángulo
    angulo_radianes = (desplazamiento_theta / num_angulos) * 2 * np.pi
    angulo_grados = np.degrees(angulo_radianes)
    angulo_grados = angulo_grados % 360
    
    # Confianza
    media_correlacion = np.mean(correlacion_theta)
    std_correlacion = np.std(correlacion_theta)
    
    if std_correlacion > 0:
        confianza = min(1.0, (pico_correlacion - media_correlacion) / (3 * std_correlacion + 1e-10))
        confianza = max(0.0, confianza)
    else:
        confianza = 0.0
    
    return {
        'desplazamiento_theta': float(desplazamiento_theta),
        'angulo_rotacion': float(angulo_grados),
        'pico_correlacion': float(pico_correlacion),
        'confianza': float(confianza)
    }


def calcular_angulo_rotacion(num_imagen1=3, num_imagen2=4, metodo='1d'):
    """
    Calcula el ángulo de rotación entre dos imágenes usando FFT 2D en coordenadas polares.
    
    Procedimiento:
    1. Carga ambas imágenes y las convierte a escala de grises
    2. Calcula la FFT 2D de cada imagen
    3. Realiza fftshift para centrar la componente cero
    4. Obtiene los módulos de los espectros
    5. Remuestrea cada espectro en una grilla polar (ρ, θ)
    6. Calcula la correlación cruzada para encontrar el desplazamiento en θ
    7. Convierte el desplazamiento a ángulo de rotación en grados
    
    Args:
        num_imagen1 (int): Número de la primera imagen (default: 3)
        num_imagen2 (int): Número de la segunda imagen (default: 4)
        metodo (str): Método de correlación ('1d' o '2d', default: '1d')
    
    Returns:
        dict: Diccionario con las siguientes claves:
            - 'angulo_rotacion': Ángulo de rotación detectado en grados
            - 'desplazamiento_theta': Desplazamiento en píxeles de θ
            - 'pico_correlacion': Valor del pico de correlación
            - 'confianza': Métrica de confianza (0-1)
            - 'metodo': Descripción del método utilizado
    """
    
    # Cargar imágenes
    img1 = cargar_imagen(num_imagen1)
    img2 = cargar_imagen(num_imagen2)
    
    # Convertir a escala de grises
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Paso 1: Calcular FFT 2D
    fft1 = np.fft.fft2(gray1)
    fft2 = np.fft.fft2(gray2)
    
    # Paso 2: Desplazar componente cero al centro (fftshift)
    fft1_shifted = np.fft.fftshift(fft1)
    fft2_shifted = np.fft.fftshift(fft2)
    
    # Paso 3: Obtener módulos (magnitudes)
    magnitud_fft1 = np.abs(fft1_shifted)
    magnitud_fft2 = np.abs(fft2_shifted)
    
    # Aplicar logaritmo para mejor visualización (opcional pero recomendado)
    # Sumar 1 para evitar log(0)
    magnitud_fft1_log = np.log1p(magnitud_fft1)
    magnitud_fft2_log = np.log1p(magnitud_fft2)
    
    # Paso 4: Remuestrear en grilla polar
    espectro_polar1 = remuestrear_a_polar(magnitud_fft1_log, num_angulos=360)
    espectro_polar2 = remuestrear_a_polar(magnitud_fft2_log, num_angulos=360)
    
    # Paso 5: Calcular correlación según el método elegido
    if metodo == '1d':
        resultado_correlacion = calcular_angulo_rotacion_1d(espectro_polar1, espectro_polar2)
        metodo_desc = 'Correlación 1D por radio (promediada)'
    elif metodo == '2d':
        resultado_correlacion = calcular_angulo_rotacion_2d(espectro_polar1, espectro_polar2)
        metodo_desc = 'Correlación 2D completa'
    else:
        raise ValueError(f"Método no reconocido: {metodo}. Use '1d' o '2d'.")
    
    # Compilar resultado final
    resultado = {
        'angulo_rotacion': resultado_correlacion['angulo_rotacion'],
        'desplazamiento_theta': resultado_correlacion['desplazamiento_theta'],
        'pico_correlacion': resultado_correlacion['pico_correlacion'],
        'confianza': resultado_correlacion['confianza'],
        'metodo': metodo_desc
    }
    
    return resultado
