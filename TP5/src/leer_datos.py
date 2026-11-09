# Path representa la ubicación del archivo de datos.
from pathlib import Path

# NumPy permite abrir archivos NPZ y copiar sus arreglos.
import numpy as np


def leer_techos(ruta_archivo: Path) -> dict[str, np.ndarray]:
    """Lee los contornos de los techos desde un archivo NPZ."""
    # Se abre el archivo y se copia cada contorno para trabajar fuera del contexto.
    with np.load(ruta_archivo, allow_pickle=False) as datos:
        return {clave: datos[clave].copy() for clave in datos.files}
