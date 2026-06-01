import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.cargar_imagenes_jpg import cargar_imagen, cargar_imagen_tif


def calcular_nitidez(imagen):
    """
    Calcula la métrica de nitidez usando la magnitud del gradiente espacial.
    
    La nitidez se mide como la norma de Frobenius de la matriz de gradientes.
    Imágenes más nítidas tendrán gradientes más pronunciados.
    
    Args:
        imagen (numpy.ndarray): Imagen en escala de grises o color
    
    Returns:
        float: Métrica de nitidez (mayor valor = imagen más nítida)
    """
    
    # Convertir a escala de grises si es necesario
    if len(imagen.shape) == 3:
        img_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        img_gray = imagen.astype(np.float32)
    
    # Calcular gradientes usando el operador de Sobel
    grad_x = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    
    # Calcular la magnitud del gradiente
    magnitud_gradiente = np.sqrt(grad_x**2 + grad_y**2)
    
    # Nitidez = norma de Frobenius de la magnitud del gradiente
    nitidez = np.sum(magnitud_gradiente**2) ** 0.5
    
    return float(nitidez)


def reconstruir_con_magnitud_y_fase_comprimida(num_magnitud, num_fase, k, 
                                                 es_tif=False, tipo_imagen="jpg"):
    """
    Reconstruye una imagen combinando la magnitud de una imagen con la fase comprimida de otra.
    
    Proceso:
    1. Calcula FFT de ambas imágenes
    2. Extrae magnitud de la primera imagen y fase de la segunda
    3. Comprime la fase por factor k (multiplica por k)
    4. Reconstruye usando mag * exp(i * k * fase)
    5. Realiza IFFT para obtener la imagen reconstruida
    
    Args:
        num_magnitud (int): Número de imagen para magnitud
        num_fase (int): Número de imagen para fase
        k (float): Factor de compresión de fase (típicamente entre 0 y 1)
        es_tif (bool): Si True, carga TIF; si False, carga JPG
        tipo_imagen (str): Tipo de imagen ("jpg" o "tif")
    
    Returns:
        numpy.ndarray: Imagen reconstruida en escala de grises
    """
    
    # Cargar imágenes
    if es_tif or tipo_imagen == "tif":
        img_magnitud = cargar_imagen_tif(num_magnitud)
        img_fase = cargar_imagen_tif(num_fase)
    else:
        img_magnitud = cargar_imagen(num_magnitud)
        img_fase = cargar_imagen(num_fase)
    
    # Convertir a escala de grises
    if len(img_magnitud.shape) == 3:
        gray_magnitud = cv2.cvtColor(img_magnitud, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray_magnitud = img_magnitud.astype(np.float32)
    
    if len(img_fase.shape) == 3:
        gray_fase = cv2.cvtColor(img_fase, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray_fase = img_fase.astype(np.float32)
    
    # Calcular FFT
    fft_magnitud = np.fft.fft2(gray_magnitud)
    fft_fase = np.fft.fft2(gray_fase)
    
    # Extraer magnitud y fase
    magnitud = np.abs(fft_magnitud)
    fase = np.angle(fft_fase)
    
    # Aplicar factor de compresión a la fase
    fase_comprimida = k * fase
    
    # Reconstruir espectro: magnitud * exp(i * fase_comprimida)
    espectro_reconstruido = magnitud * np.exp(1j * fase_comprimida)
    
    # Transformada inversa
    imagen_reconstruida = np.fft.ifft2(espectro_reconstruido)
    imagen_reconstruida = np.abs(imagen_reconstruida)
    
    # Normalizar a rango [0, 255]
    imagen_reconstruida = (imagen_reconstruida - np.min(imagen_reconstruida)) / \
                          (np.max(imagen_reconstruida) - np.min(imagen_reconstruida) + 1e-10) * 255
    
    return imagen_reconstruida.astype(np.uint8)


def buscar_factor_k_optimo(num_magnitud, num_fase, rango_k=(0.1, 2.0), 
                           pasos=50, es_tif=False, tipo_imagen="jpg",
                           verbose=True):
    """
    Busca el factor k óptimo que maximiza la nitidez de la imagen reconstruida.
    
    Algoritmo:
    1. Define rango de valores k a probar
    2. Para cada k:
       - Reconstruye imagen con magnitud+fase comprimida
       - Calcula métrica de nitidez (magnitud del gradiente)
    3. Retorna k que maximiza la nitidez
    
    Args:
        num_magnitud (int): Número de imagen para magnitud
        num_fase (int): Número de imagen para fase
        rango_k (tuple): (k_min, k_max) para búsqueda
        pasos (int): Número de valores k a probar
        es_tif (bool): Si True, carga TIF; si False, carga JPG
        tipo_imagen (str): Tipo de imagen ("jpg" o "tif")
        verbose (bool): Si True, imprime progreso
    
    Returns:
        dict: Diccionario con claves:
            - 'k_optimo': Factor k que maximiza nitidez
            - 'nitidez_maxima': Valor de nitidez máxima alcanzada
            - 'imagen_optima': Imagen reconstruida con k óptimo
            - 'valores_k': Lista de valores k probados
            - 'nitideces': Lista de nitideces calculadas
    """
    
    k_min, k_max = rango_k
    valores_k = np.linspace(k_min, k_max, pasos)
    nitideces = []
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Búsqueda de factor k óptimo")
        print(f"{'='*70}")
        print(f"Rango de k: [{k_min:.3f}, {k_max:.3f}]")
        print(f"Pasos: {pasos}")
        print(f"Tipo de imagen: {tipo_imagen.upper()}")
        print(f"{'='*70}")
    
    # Probar cada valor de k
    for i, k in enumerate(valores_k):
        imagen_recon = reconstruir_con_magnitud_y_fase_comprimida(
            num_magnitud, num_fase, k, es_tif=es_tif, tipo_imagen=tipo_imagen
        )
        nitidez = calcular_nitidez(imagen_recon)
        nitideces.append(nitidez)
        
        if verbose and (i + 1) % (pasos // 5) == 0:
            print(f"  Progreso: {i+1}/{pasos} - k={k:.4f}, Nitidez={nitidez:.2e}")
    
    # Encontrar k óptimo
    idx_optimo = np.argmax(nitideces)
    k_optimo = valores_k[idx_optimo]
    nitidez_maxima = nitideces[idx_optimo]
    
    # Reconstruir imagen con k óptimo
    imagen_optima = reconstruir_con_magnitud_y_fase_comprimida(
        num_magnitud, num_fase, k_optimo, es_tif=es_tif, tipo_imagen=tipo_imagen
    )
    
    if verbose:
        print(f"\n{'*'*70}")
        print(f"RESULTADO: Factor k óptimo encontrado")
        print(f"{'*'*70}")
        print(f"  k_óptimo:        {k_optimo:.6f}")
        print(f"  Nitidez máxima:  {nitidez_maxima:.4e}")
        print(f"{'*'*70}\n")
    
    return {
        'k_optimo': float(k_optimo),
        'nitidez_maxima': float(nitidez_maxima),
        'imagen_optima': imagen_optima,
        'valores_k': valores_k.tolist(),
        'nitideces': nitideces
    }


def recuperar_imagen_magnitud(num_magnitud, num_fase, tipo_imagen="jpg"):
    """
    Recupera la imagen objetivo usando magnitud de f6 y fase de f7.
    
    Busca el factor k óptimo para recuperar la imagen con máxima nitidez.
    
    Args:
        num_magnitud (int): Número de imagen con magnitud objetivo
        num_fase (int): Número de imagen con fase alterada
        tipo_imagen (str): Tipo de imagen ("jpg" o "tif")
    
    Returns:
        dict: Resultado de la búsqueda con k_optimo, imagen_optima, etc.
    """
    
    es_tif = (tipo_imagen.lower() == "tif")
    
    resultado = buscar_factor_k_optimo(
        num_magnitud, num_fase,
        rango_k=(0.01, 2.0),
        pasos=100,
        es_tif=es_tif,
        tipo_imagen=tipo_imagen,
        verbose=True
    )
    
    return resultado


def recuperar_imagen_fase(num_magnitud, num_fase, tipo_imagen="jpg"):
    """
    Recupera la imagen objetivo usando fase de f7 y magnitud de f6.
    
    Busca el factor k óptimo para recuperar la imagen con máxima nitidez.
    
    Args:
        num_magnitud (int): Número de imagen con magnitud alterada
        num_fase (int): Número de imagen con fase objetivo
        tipo_imagen (str): Tipo de imagen ("jpg" o "tif")
    
    Returns:
        dict: Resultado de la búsqueda con k_optimo, imagen_optima, etc.
    """
    
    es_tif = (tipo_imagen.lower() == "tif")
    
    resultado = buscar_factor_k_optimo(
        num_magnitud, num_fase,
        rango_k=(0.01, 2.0),
        pasos=100,
        es_tif=es_tif,
        tipo_imagen=tipo_imagen,
        verbose=True
    )
    
    return resultado


def visualizar_resultados_inciso4_tif(resultado_magnitud, resultado_fase):
    """
    Visualiza resultados para formato TIF.
    
    Args:
        resultado_magnitud: Dict con resultado para magnitud (TIF)
        resultado_fase: Dict con resultado para fase (TIF)
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Inciso 4: Coherencia de Fase - Formato TIF', fontsize=14, fontweight='bold')
    
    # Magnitud: Imagen
    axes[0, 0].imshow(resultado_magnitud['imagen_optima'], cmap='gray')
    axes[0, 0].set_title(f'Imagen Reconstruida (Magnitud)\nk óptimo = {resultado_magnitud["k_optimo"]:.4f}')
    axes[0, 0].axis('off')
    
    # Magnitud: Gráfico
    axes[1, 0].plot(resultado_magnitud['valores_k'], resultado_magnitud['nitideces'], 'b-', linewidth=2)
    axes[1, 0].axvline(resultado_magnitud['k_optimo'], color='r', linestyle='--', linewidth=2, label=f'k óptimo = {resultado_magnitud["k_optimo"]:.4f}')
    axes[1, 0].scatter([resultado_magnitud['k_optimo']], [resultado_magnitud['nitidez_maxima']], color='r', s=100, zorder=5)
    axes[1, 0].set_xlabel('Factor k', fontsize=11)
    axes[1, 0].set_ylabel('Nitidez', fontsize=11)
    axes[1, 0].set_title('Búsqueda del Factor k Óptimo (Magnitud)', fontsize=11)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Fase: Imagen
    axes[0, 1].imshow(resultado_fase['imagen_optima'], cmap='gray')
    axes[0, 1].set_title(f'Imagen Reconstruida (Fase)\nk óptimo = {resultado_fase["k_optimo"]:.4f}')
    axes[0, 1].axis('off')
    
    # Fase: Gráfico
    axes[1, 1].plot(resultado_fase['valores_k'], resultado_fase['nitideces'], 'g-', linewidth=2)
    axes[1, 1].axvline(resultado_fase['k_optimo'], color='r', linestyle='--', linewidth=2, label=f'k óptimo = {resultado_fase["k_optimo"]:.4f}')
    axes[1, 1].scatter([resultado_fase['k_optimo']], [resultado_fase['nitidez_maxima']], color='r', s=100, zorder=5)
    axes[1, 1].set_xlabel('Factor k', fontsize=11)
    axes[1, 1].set_ylabel('Nitidez', fontsize=11)
    axes[1, 1].set_title('Búsqueda del Factor k Óptimo (Fase)', fontsize=11)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def visualizar_resultados_inciso4_jpg(resultado_magnitud, resultado_fase):
    """
    Visualiza resultados para formato JPG.
    
    Args:
        resultado_magnitud: Dict con resultado para magnitud (JPG)
        resultado_fase: Dict con resultado para fase (JPG)
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Inciso 4: Coherencia de Fase - Formato JPG', fontsize=14, fontweight='bold')
    
    # Magnitud: Imagen
    axes[0, 0].imshow(resultado_magnitud['imagen_optima'], cmap='gray')
    axes[0, 0].set_title(f'Imagen Reconstruida (Magnitud)\nk óptimo = {resultado_magnitud["k_optimo"]:.4f}')
    axes[0, 0].axis('off')
    
    # Magnitud: Gráfico
    axes[1, 0].plot(resultado_magnitud['valores_k'], resultado_magnitud['nitideces'], 'b-', linewidth=2)
    axes[1, 0].axvline(resultado_magnitud['k_optimo'], color='r', linestyle='--', linewidth=2, label=f'k óptimo = {resultado_magnitud["k_optimo"]:.4f}')
    axes[1, 0].scatter([resultado_magnitud['k_optimo']], [resultado_magnitud['nitidez_maxima']], color='r', s=100, zorder=5)
    axes[1, 0].set_xlabel('Factor k', fontsize=11)
    axes[1, 0].set_ylabel('Nitidez', fontsize=11)
    axes[1, 0].set_title('Búsqueda del Factor k Óptimo (Magnitud)', fontsize=11)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Fase: Imagen
    axes[0, 1].imshow(resultado_fase['imagen_optima'], cmap='gray')
    axes[0, 1].set_title(f'Imagen Reconstruida (Fase)\nk óptimo = {resultado_fase["k_optimo"]:.4f}')
    axes[0, 1].axis('off')
    
    # Fase: Gráfico
    axes[1, 1].plot(resultado_fase['valores_k'], resultado_fase['nitideces'], 'g-', linewidth=2)
    axes[1, 1].axvline(resultado_fase['k_optimo'], color='r', linestyle='--', linewidth=2, label=f'k óptimo = {resultado_fase["k_optimo"]:.4f}')
    axes[1, 1].scatter([resultado_fase['k_optimo']], [resultado_fase['nitidez_maxima']], color='r', s=100, zorder=5)
    axes[1, 1].set_xlabel('Factor k', fontsize=11)
    axes[1, 1].set_ylabel('Nitidez', fontsize=11)
    axes[1, 1].set_title('Búsqueda del Factor k Óptimo (Fase)', fontsize=11)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
