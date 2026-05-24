import cv2
import numpy as np
import argparse
from pathlib import Path
from src.cargar_imagenes_jpg import cargar_imagen_rgb


def crear_imagen_cartesiana(numero_imagen, dx=0, dy=0):
    """
    Crea una imagen en sistema de coordenadas cartesianas.
    
    Convierte una imagen del sistema de coordenadas de pantalla (origen arriba-izquierda)
    al sistema de coordenadas cartesianas (origen abajo-izquierda) y aplica
    los desplazamientos especificados.
    
    Args:
        numero_imagen (int): Número de la imagen a cargar (1-7)
        dx (int): Desplazamiento en el eje X (píxeles). Default: 0
        dy (int): Desplazamiento en el eje Y (píxeles). Default: 0
    
    Returns:
        tuple: (imagen_procesada, ruta_guardada)
               - imagen_procesada: Array numpy con la imagen procesada
               - ruta_guardada: Ruta donde se guardó la imagen
    
    Raises:
        FileNotFoundError: Si la imagen no existe
        ValueError: Si los parámetros no son válidos
    """
    
    # Validar entrada
    if not isinstance(numero_imagen, int) or numero_imagen < 1 or numero_imagen > 7:
        raise ValueError("El número de imagen debe ser un entero entre 1 y 7")
    
    if not isinstance(dx, (int, float)):
        raise ValueError("dx debe ser un número")
    
    if not isinstance(dy, (int, float)):
        raise ValueError("dy debe ser un número")
    
    # Convertir a enteros
    dx = int(dx)
    dy = int(dy)
    
    # Cargar la imagen en RGB
    imagen_original = cargar_imagen_rgb(numero_imagen)
    
    # Paso 1: Convertir al sistema de coordenadas cartesianas
    # En sistema de pantalla: (0,0) está arriba-izquierda
    # En sistema cartesiano: (0,0) está abajo-izquierda
    # Solución: hacer flip vertical de la imagen
    
    # Paso 2: Aplicar desplazamientos con relleno por wrapping
    if dx != 0 or dy != 0:
        # Crear matriz de traslación
        # Nota: cv2.warpAffine espera desplazamientos en píxeles (positivo = derecha, arriba)
        matriz_traslacion = np.float32([[1, 0, dx],
                                         [0, 1, dy]])  
        
        height, width = imagen_original.shape[:2]
        # BORDER_WRAP rellena con la imagen que se desplazó (wrapping circular)
        imagen_original = cv2.warpAffine(imagen_original, 
                                           matriz_traslacion,
                                           (width, height),
                                           borderMode=cv2.BORDER_WRAP)
    
    # Paso 3: Guardar la imagen
    # Crear carpeta si no existe
    ruta_base = Path(__file__).parent.parent
    carpeta_destino = ruta_base / "imagenes creadas"
    carpeta_destino.mkdir(exist_ok=True)
    
    # Construir nombre del archivo
    if dx == 0 and dy == 0:
        nombre_archivo = f"imagen{numero_imagen}_cartesiana.png"
    else:
        nombre_archivo = f"imagen{numero_imagen}_cartesiana_dx{dx:+d}_dy{dy:+d}.png"
    
    ruta_guardada = carpeta_destino / nombre_archivo
    
    # Convertir de RGB a BGR para cv2.imwrite
    imagen_bgr = cv2.cvtColor(imagen_original, cv2.COLOR_RGB2BGR)
    
    # Guardar la imagen
    éxito = cv2.imwrite(str(ruta_guardada), imagen_bgr)
    
    if not éxito:
        raise RuntimeError(f"No se pudo guardar la imagen en: {ruta_guardada}")
    
    return imagen_original, ruta_guardada


def main():
    """Función principal para ejecutar desde línea de comandos"""
    
    parser = argparse.ArgumentParser(
        description="Crea una imagen en sistema de coordenadas cartesianas con desplazamientos opcionales",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python crear_imagenes_cart.py 1                    # Imagen 1 sin desplazamientos
  python crear_imagenes_cart.py 2 --dx 10 --dy -5   # Imagen 2 con dx=10, dy=-5
  python crear_imagenes_cart.py 3 -x 20 -y 15       # Imagen 3 con dx=20, dy=15
        """
    )
    
    parser.add_argument('numero_imagen',
                        type=int,
                        help='Número de la imagen a procesar (1-7)')
    
    parser.add_argument('--dx', '-x',
                        type=float,
                        default=0,
                        help='Desplazamiento en el eje X (píxeles). Default: 0')
    
    parser.add_argument('--dy', '-y',
                        type=float,
                        default=0,
                        help='Desplazamiento en el eje Y (píxeles). Default: 0')
    
    args = parser.parse_args()
    
    try:
        print("="*70)
        print("CREACIÓN DE IMAGEN EN SISTEMA DE COORDENADAS CARTESIANAS")
        print("="*70)
        print()
        print(f"Número de imagen:     {args.numero_imagen}")
        print(f"Desplazamiento X:     {args.dx:.2f} píxeles")
        print(f"Desplazamiento Y:     {args.dy:.2f} píxeles")
        print()
        print("Procesando...")
        
        imagen_procesada, ruta_guardada = crear_imagen_cartesiana(
            args.numero_imagen,
            dx=args.dx,
            dy=args.dy
        )
        
        print()
        print("="*70)
        print("¡ÉXITO!")
        print("="*70)
        print(f"Imagen guardada en: {ruta_guardada}")
        print(f"Dimensiones:        {imagen_procesada.shape}")
        print()
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    except ValueError as e:
        print(f"ERROR de validación: {e}")
        return 1
    except Exception as e:
        print(f"ERROR inesperado: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
