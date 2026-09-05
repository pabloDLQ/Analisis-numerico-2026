"""Estructura central de datos para la trayectoria del dron.

Este archivo define la dataclass Trajectory, que centraliza la salida de cada
etapa del pipeline: posicion en el mapa, escala relativa, flujo optico,
estadisticos de validacion y estado de registro. La idea es que cada modulo
trabaje con una misma representacion de datos para evitar que el algoritmo se
pierda en estructuras inconsistentes.

La dataclass sirve como contenedor unificado: cada funcion del sistema agrega o
lee informacion coherente desde la misma estructura, sin mezclar tipos ni
formatos distintos a lo largo de la ejecucion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Trajectory:
    frame: np.ndarray
    time_s: np.ndarray
    x_map: np.ndarray
    y_map: np.ndarray
    z: np.ndarray
    u: np.ndarray
    v: np.ndarray
    scale_step: np.ndarray
    valid_points: np.ndarray
    lk_error: np.ndarray
    map_inliers: np.ndarray
    registered: np.ndarray
