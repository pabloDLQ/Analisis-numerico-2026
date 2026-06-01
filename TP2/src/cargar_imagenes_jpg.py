import cv2
import os
from pathlib import Path


def cargar_imagen(numero_imagen):
    """
    Carga una imagen JPG desde la carpeta data/.
    
    Args:
        numero_imagen (int): Número de la imagen a cargar (1-7)
    
    Returns:
        numpy.ndarray: Array con los datos de la imagen cargada
                      Si se usa cv2, retorna en formato BGR
    
    Raises:
        FileNotFoundError: Si la imagen no existe en la carpeta data/
    """
    
    # Construir la ruta de la imagen
    ruta_base = Path(__file__).parent.parent  # Sube dos niveles desde src/cargar_imagenes_jpg.py
    ruta_imagen = ruta_base / "data" / f"imagen{numero_imagen}.jpg"
    
    # Verificar que la ruta existe
    if not ruta_imagen.exists():
        raise FileNotFoundError(f"La imagen no existe en: {ruta_imagen}")
    
    # Cargar la imagen usando OpenCV
    imagen = cv2.imread(str(ruta_imagen))
    
    if imagen is None:
        raise RuntimeError(f"No se pudo cargar la imagen: {ruta_imagen}")
    
    return imagen


def cargar_imagen_rgb(numero_imagen):
    """
    Carga una imagen JPG y la convierte a formato RGB.
    
    Args:
        numero_imagen (int): Número de la imagen a cargar (1-7)
    
    Returns:
        numpy.ndarray: Array con los datos de la imagen en formato RGB
    """
    
    imagen_bgr = cargar_imagen(numero_imagen)
    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    
    return imagen_rgb


def cargar_imagen_tif(numero_imagen, nombre_prefijo="imagen"):
    """
    Carga una imagen TIF desde la carpeta data/.
    
    Args:
        numero_imagen (int): Número de la imagen a cargar
        nombre_prefijo (str): Prefijo del archivo TIF (default: 'imagen')
    
    Returns:
        numpy.ndarray: Array con los datos de la imagen cargada
                      En formato original (puede ser BGR, RGB o escala de grises)
    
    Raises:
        FileNotFoundError: Si la imagen no existe en la carpeta data/
        RuntimeError: Si no se puede cargar la imagen
    """
    
    # Construir la ruta de la imagen
    ruta_base = Path(__file__).parent.parent
    ruta_imagen = ruta_base / "data" / f"{nombre_prefijo}{numero_imagen}.tif"
    
    # Verificar que la ruta existe
    if not ruta_imagen.exists():
        raise FileNotFoundError(f"La imagen no existe en: {ruta_imagen}")
    
    # Cargar la imagen usando OpenCV
    imagen = cv2.imread(str(ruta_imagen), cv2.IMREAD_UNCHANGED)
    
    if imagen is None:
        raise RuntimeError(f"No se pudo cargar la imagen: {ruta_imagen}")
    
    return imagen


# Ejemplo de uso
if __name__ == "__main__":
    # Cargar la imagen 1
    try:
        img = cargar_imagen(1)
        print(f"Imagen cargada exitosamente")
        print(f"Dimensiones: {img.shape}")
        print(f"Tipo de dato: {img.dtype}")
        
        # Cargar en formato RGB
        img_rgb = cargar_imagen_rgb(1)
        print(f"Imagen RGB cargada: {img_rgb.shape}")
        
    except Exception as e:
        print(f"Error: {e}")
