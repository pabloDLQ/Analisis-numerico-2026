from pathlib import Path

import numpy as np


def leer_techos(ruta_archivo: Path) -> dict[str, np.ndarray]:
    """Lee los contornos de los techos desde un archivo NPZ."""
    with np.load(ruta_archivo, allow_pickle=False) as datos:
        return {clave: datos[clave].copy() for clave in datos.files}
