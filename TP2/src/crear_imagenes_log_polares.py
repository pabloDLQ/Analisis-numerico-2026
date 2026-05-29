import argparse
from pathlib import Path

import cv2
import numpy as np

from src.cargar_imagenes_jpg import cargar_imagen


def crear_imagen_log_polar(numero_imagen, k):
    """
    Crea una nueva imagen aplicando una escala radial polar de factor k.

    La imagen resultante conserva el mismo tamaño que la original, pero cada
    punto se remapea usando coordenadas polares con radio r * k.

    Args:
        numero_imagen (int): Número de la imagen a cargar (1-7)
        k (float): Factor de escala radial. Debe ser mayor que 0.

    Returns:
        tuple: (imagen_procesada, ruta_guardada)
    """

    if not isinstance(numero_imagen, int) or numero_imagen < 1 or numero_imagen > 7:
        raise ValueError("El número de imagen debe ser un entero entre 1 y 7")

    if not isinstance(k, (int, float)):
        raise ValueError("k debe ser un número")

    k = float(k)
    if k <= 0:
        raise ValueError("k debe ser mayor que 0")

    imagen_original = cargar_imagen(numero_imagen)
    height, width = imagen_original.shape[:2]

    centro_x = (width - 1) / 2.0
    centro_y = (height - 1) / 2.0

    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32)
    )

    dx = grid_x - centro_x
    dy = grid_y - centro_y

    radio_destino = np.sqrt(dx * dx + dy * dy)
    angulo = np.arctan2(dy, dx)

    radio_origen = radio_destino / k
    mapa_x = centro_x + radio_origen * np.cos(angulo)
    mapa_y = centro_y + radio_origen * np.sin(angulo)

    imagen_procesada = cv2.remap(
        imagen_original,
        mapa_x.astype(np.float32),
        mapa_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    directorio_salida = Path(__file__).parent.parent / "imagenes creadas"
    directorio_salida.mkdir(exist_ok=True)

    nombre_archivo = f"imagen_{numero_imagen}_log_polar_k{k:.3f}.jpg"
    ruta_guardada = directorio_salida / nombre_archivo

    if not cv2.imwrite(str(ruta_guardada), imagen_procesada):
        raise RuntimeError(f"No se pudo guardar la imagen en: {ruta_guardada}")

    return imagen_procesada, ruta_guardada


def main():
    """Ejecuta el script desde línea de comandos."""

    parser = argparse.ArgumentParser(
        description="Crea una imagen con escala radial polar r*k",
    )
    parser.add_argument("numero_imagen", type=int, help="Número de la imagen a procesar (1-7)")
    parser.add_argument("k", type=float, help="Factor de escala radial polar")

    args = parser.parse_args()

    try:
        print("=" * 70)
        print("CREACIÓN DE IMAGEN CON ESCALA RADIAL POLAR")
        print("=" * 70)
        print()
        print(f"Número de imagen: {args.numero_imagen}")
        print(f"Factor k:         {args.k:.4f}")
        print()
        print("Procesando...")

        imagen_procesada, ruta_guardada = crear_imagen_log_polar(args.numero_imagen, args.k)

        print()
        print("=" * 70)
        print("¡ÉXITO!")
        print("=" * 70)
        print(f"Imagen guardada en: {ruta_guardada}")
        print(f"Dimensiones:        {imagen_procesada.shape}")
        print()
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"ERROR: {e}")
        return 1
    except Exception as e:
        print(f"ERROR inesperado: {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())