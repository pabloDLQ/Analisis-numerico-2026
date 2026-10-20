import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
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


def reconstruir_imagen_objetivo(num_magnitud, num_fase, k, es_tif=False):
    """
    Reconstruye la imagen objetivo combinando:
    - Magnitud de la imagen num_magnitud (f6)
    - Fase comprimida de la imagen num_fase (f7)
    
    Proceso:
    1. Calcula FFT de ambas imágenes
    2. Extrae magnitud de f6 y fase de f7
    3. Comprime la fase por factor k (multiplica por k)
    4. Reconstruye usando mag * exp(i * k * fase)
    5. Realiza IFFT para obtener la imagen reconstruida
    
    Args:
        num_magnitud (int): Número de imagen para magnitud (típicamente 6)
        num_fase (int): Número de imagen para fase alterada (típicamente 7)
        k (float): Factor de compresión de fase
        es_tif (bool): Si True, carga TIF; si False, carga JPG
    
    Returns:
        numpy.ndarray: Imagen reconstruida en escala de grises [0, 255]
    """
    
    # Cargar imágenes
    if es_tif:
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
    
    # Reconstruir espectro: magnitud * exp(i * k * fase)
    espectro_reconstruido = magnitud * np.exp(1j * fase_comprimida)
    
    # Transformada inversa
    imagen_reconstruida = np.fft.ifft2(espectro_reconstruido)
    imagen_reconstruida = np.abs(imagen_reconstruida)
    
    # Normalizar a rango [0, 255]
    imagen_reconstruida = (imagen_reconstruida - np.min(imagen_reconstruida)) / \
                          (np.max(imagen_reconstruida) - np.min(imagen_reconstruida) + 1e-10) * 255
    
    return imagen_reconstruida.astype(np.uint8)


def buscar_factor_k_optimo(num_magnitud, num_fase, rango_k=(0.1, 5.0), 
                           pasos=200, es_tif=False, verbose=True):
    """
    Busca el factor k óptimo que maximiza la nitidez de la imagen objetivo.
    
    Utiliza magnitud de f6 y fase alterada de f7.
    
    Algoritmo de dos fases:
    1. Búsqueda gruesa: Prueba valores en el rango especificado
    2. Búsqueda fina: Refina alrededor del máximo encontrado
    
    Args:
        num_magnitud (int): Número de imagen para magnitud (típicamente 6)
        num_fase (int): Número de imagen para fase (típicamente 7)
        rango_k (tuple): (k_min, k_max) para búsqueda. Por defecto (0.1, 5.0)
        pasos (int): Número de valores k a probar. Por defecto 200
        es_tif (bool): Si True, carga TIF; si False, carga JPG
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
    tipo_formato = "TIF" if es_tif else "JPG"
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"BÚSQUEDA DE FACTOR k ÓPTIMO - Formato {tipo_formato.upper()}")
        print(f"{'='*70}")
        print(f"Magnitud: Imagen f{num_magnitud}")
        print(f"Fase (alterada): Imagen f{num_fase}")
        print(f"Rango de k: [{k_min:.3f}, {k_max:.3f}]")
        print(f"Pasos: {pasos}")
        print(f"{'='*70}")
    
    # FASE 1: Búsqueda gruesa en el rango especificado
    if verbose:
        print(f"\nFase 1: Búsqueda gruesa ({pasos} puntos)...")
    
    valores_k = np.linspace(k_min, k_max, pasos)
    nitideces = []
    
    for i, k in enumerate(valores_k):
        imagen_recon = reconstruir_imagen_objetivo(
            num_magnitud, num_fase, k, es_tif=es_tif
        )
        nitidez = calcular_nitidez(imagen_recon)
        nitideces.append(nitidez)
        
        if verbose and (i + 1) % (pasos // 5) == 0:
            print(f"  Progreso: {i+1}/{pasos} - k={k:.4f}, Nitidez={nitidez:.4e}")
    
    # Encontrar el máximo aproximado
    idx_aprox = np.argmax(nitideces)
    k_aprox = valores_k[idx_aprox]
    
    # FASE 2: Búsqueda fina alrededor del máximo
    if verbose:
        print(f"\nFase 2: Búsqueda fina alrededor de k={k_aprox:.4f}...")
    
    # Definir rango fino: ±20% del rango original, centrado en k_aprox
    rango_fino = (k_max - k_min) * 0.15  # 15% del rango original
    k_min_fino = max(k_min, k_aprox - rango_fino)
    k_max_fino = min(k_max, k_aprox + rango_fino)
    
    pasos_fino = 100
    valores_k_fino = np.linspace(k_min_fino, k_max_fino, pasos_fino)
    nitideces_finas = []
    
    for i, k in enumerate(valores_k_fino):
        imagen_recon = reconstruir_imagen_objetivo(
            num_magnitud, num_fase, k, es_tif=es_tif
        )
        nitidez = calcular_nitidez(imagen_recon)
        nitideces_finas.append(nitidez)
        
        if verbose and (i + 1) % (pasos_fino // 3) == 0:
            print(f"  Progreso: {i+1}/{pasos_fino} - k={k:.4f}, Nitidez={nitidez:.4e}")
    
    # Encontrar k óptimo en búsqueda fina
    idx_optimo_fino = np.argmax(nitideces_finas)
    k_optimo = valores_k_fino[idx_optimo_fino]
    nitidez_maxima = nitideces_finas[idx_optimo_fino]
    
    # Reconstruir imagen con k óptimo
    imagen_optima = reconstruir_imagen_objetivo(
        num_magnitud, num_fase, k_optimo, es_tif=es_tif
    )
    
    # Combinar resultados de ambas fases para retorno
    valores_k_total = np.concatenate([valores_k, valores_k_fino])
    nitideces_total = nitideces + nitideces_finas
    
    if verbose:
        print(f"\n{'*'*70}")
        print(f"RESULTADO: Factor k óptimo encontrado")
        print(f"{'*'*70}")
        print(f"  k_óptimo:        {k_optimo:.6f}")
        print(f"  Nitidez máxima:  {nitidez_maxima:.4e}")
        print(f"  Rango de búsqueda fina: [{k_min_fino:.6f}, {k_max_fino:.6f}]")
        print(f"{'*'*70}\n")
    
    return {
        'k_optimo': float(k_optimo),
        'nitidez_maxima': float(nitidez_maxima),
        'imagen_optima': imagen_optima,
        'valores_k': valores_k_fino.tolist(),  # Retornar solo búsqueda fina para gráfico
        'nitideces': nitideces_finas,
        'tipo_formato': tipo_formato
    }


def guardar_grafico_comparativo_inciso4(resultado_tif, resultado_jpg):
    """
    Guarda una comparativa de la coherencia de fase para TIF y JPG.
    
    Args:
        resultado_tif: Dict con resultado para formato TIF
        resultado_jpg: Dict con resultado para formato JPG
    """

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])

    eje_curvas = fig.add_subplot(gs[0, :])
    eje_curvas.plot(
        resultado_tif['valores_k'],
        resultado_tif['nitideces'],
        color='blue',
        linewidth=2.5,
        label=f"TIF (k óptimo = {resultado_tif['k_optimo']:.4f})"
    )
    eje_curvas.plot(
        resultado_jpg['valores_k'],
        resultado_jpg['nitideces'],
        color='red',
        linestyle='--',
        linewidth=2.5,
        label=f"JPG (k óptimo = {resultado_jpg['k_optimo']:.4f})"
    )
    eje_curvas.scatter(
        resultado_tif['k_optimo'],
        resultado_tif['nitidez_maxima'],
        color='blue',
        s=100,
        zorder=5
    )
    eje_curvas.scatter(
        resultado_jpg['k_optimo'],
        resultado_jpg['nitidez_maxima'],
        color='red',
        s=100,
        zorder=5
    )
    eje_curvas.set_xlabel(r'Factor de compresión de fase ($k$)')
    eje_curvas.set_ylabel('Nitidez (Norma de Frobenius)')
    eje_curvas.set_title(
        'Efecto de la Compresión en la Coherencia de Fase (TIF vs JPG)'
    )
    eje_curvas.legend()
    eje_curvas.grid(True, alpha=0.3)

    eje_tif = fig.add_subplot(gs[1, 0])
    eje_tif.imshow(resultado_tif['imagen_optima'], cmap='gray')
    eje_tif.set_title('Reconstrucción TIF (Sin pérdida)')
    eje_tif.axis('off')

    eje_jpg = fig.add_subplot(gs[1, 1])
    eje_jpg.imshow(resultado_jpg['imagen_optima'], cmap='gray')
    eje_jpg.set_title('Reconstrucción JPG (Con artefactos de compresión)')
    eje_jpg.axis('off')

    fig.tight_layout()
    ruta_salida = Path(__file__).resolve().parent.parent / 'imagenes creadas'
    ruta_salida.mkdir(parents=True, exist_ok=True)
    archivo_salida = ruta_salida / 'inciso4_comparativa_fase.png'
    fig.savefig(archivo_salida, bbox_inches='tight')
    plt.close(fig)

    print(f'Comparativa de coherencia de fase guardada en: {archivo_salida}')


def guardar_imagenes_resultantes(resultado_tif, resultado_jpg):
    """
    Guarda las imágenes recuperadas en la carpeta 'imagenes creadas'.
    
    Args:
        resultado_tif: Dict con imagen reconstruida (TIF)
        resultado_jpg: Dict con imagen reconstruida (JPG)
    """
    
    ruta_base = Path(__file__).parent.parent
    carpeta_imagenes = ruta_base / "imagenes creadas"
    carpeta_imagenes.mkdir(exist_ok=True)
    
    # Guardar imagen TIF
    ruta_tif = carpeta_imagenes / "inciso4_imagen_objetivo_TIF.jpg"
    cv2.imwrite(str(ruta_tif), resultado_tif['imagen_optima'])
    
    # Guardar imagen JPG
    ruta_jpg = carpeta_imagenes / "inciso4_imagen_objetivo_JPG.jpg"
    cv2.imwrite(str(ruta_jpg), resultado_jpg['imagen_optima'])
    
    print("\n" + "="*70)
    print("IMÁGENES RECUPERADAS GUARDADAS EXITOSAMENTE")
    print("="*70)
    print(f"Ubicación: {carpeta_imagenes}\n")
    print(f"Imagen objetivo (TIF):")
    print(f"  • Archivo: inciso4_imagen_objetivo_TIF.jpg")
    print(f"  • k óptimo: {resultado_tif['k_optimo']:.6f}")
    print(f"  • Nitidez máxima: {resultado_tif['nitidez_maxima']:.4e}\n")
    print(f"Imagen objetivo (JPG):")
    print(f"  • Archivo: inciso4_imagen_objetivo_JPG.jpg")
    print(f"  • k óptimo: {resultado_jpg['k_optimo']:.6f}")
    print(f"  • Nitidez máxima: {resultado_jpg['nitidez_maxima']:.4e}")
    print("="*70)
