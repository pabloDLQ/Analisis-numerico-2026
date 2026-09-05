"""Deteccion de puntos de interes para la estimacion de flujo optico.

Este modulo selecciona esquinas con cv2.goodFeaturesToTrack, porque la matriz
A^T A que aparece en Lucas-Kanade requiere textura local suficiente para que el
problema sea bien condicionado. Cuando la escena es plana o muy uniforme,
las esquinas no son confiables y la estimacion del flujo se vuelve inestable.

La funcion detect_features hace una primera seleccion de regiones con buena
estructura local para que la ecuacion normal del seguimiento tenga una solucion
numericamente estable y no dependa de puntos demasiados planos o sin gradiente.
"""

from __future__ import annotations

import cv2
import numpy as np


def detect_features(gray: np.ndarray, max_corners: int = 300) -> np.ndarray:
    points = cv2.goodFeaturesToTrack(gray, max_corners, 0.01, 8, blockSize=7)
    return np.empty((0, 2), np.float32) if points is None else points.reshape(-1, 2)
