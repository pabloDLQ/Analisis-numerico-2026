"""Flujo optico por Lucas-Kanade explicitamente resuelto.

Este modulo implementa la estimacion local del desplazamiento entre dos frames
consecutivos. El principio matematico es la aproximacion de Taylor de primer
orden del campo de intensidad, que lleva al sistema lineal de minimos cuadrados
(A^T A) u = A^T b. Se filtran ademas outliers con una medida robusta basada en
MAD para evitar que un par de puntos malos arruinen todo el frame.

La logica del arreglo es clara: para cada punto de interes se arma una ventana
local, se calculan gradientes espaciales y se resuelve la ecuacion normal que
minimiza el error entre la imagen anterior y la actual. Luego se descarta la
parte de flujo que resulta imposible o demasiado anomala.
"""

from __future__ import annotations

import cv2
import numpy as np


def lucas_kanade(previous: np.ndarray, current: np.ndarray, points: np.ndarray, radius: int = 5):
    """Estima el desplazamiento local de puntos de interes entre dos frames.

    La idea de esta funcion es resolver, para cada ventana alrededor de un punto,
    el sistema de minimos cuadrados que surge de la aproximacion lineal de la
    intensidad en torno a la posicion anterior. En otras palabras, se busca el
    vector de movimiento (u, v) que mejor explica la diferencia entre la imagen
    previa y la actual.

    La construccion del problema sigue la forma canonica del algoritmo
    Lucas-Kanade: A * d ≈ b, con A = [Ix, Iy] y b = I_t. La solucion se obtiene
    mediante la ecuacion normal (A.T A) d = A.T b, usando la matriz de Gram y
    descartando ventanas mal condicionadas o con movimiento imposible.
    """
    ix = cv2.Sobel(previous, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(previous, cv2.CV_32F, 0, 1, ksize=3)
    temporal = current.astype(np.float32) - previous.astype(np.float32)
    height, width = previous.shape
    old_valid, new_valid, flows, errors = [], [], [], []
    for point in points.reshape(-1, 2):
        x, y = np.round(point).astype(int)
        x0, x1, y0, y1 = x - radius, x + radius + 1, y - radius, y + radius + 1
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            continue
        matrix = np.column_stack((ix[y0:y1, x0:x1].ravel(), iy[y0:y1, x0:x1].ravel()))
        difference = temporal[y0:y1, x0:x1].ravel()
        normal = matrix.T @ matrix
        if np.linalg.det(normal) <= 1e-3:
            continue
        flow = np.linalg.solve(normal, matrix.T @ difference)
        if not np.isfinite(flow).all() or np.linalg.norm(flow) > 20:
            continue
        old_valid.append(point)
        new_valid.append(point + flow)
        flows.append(flow)
        errors.append(np.sqrt(np.mean((matrix @ flow - difference) ** 2)))
    return (
        np.asarray(old_valid, np.float32).reshape(-1, 2),
        np.asarray(new_valid, np.float32).reshape(-1, 2),
        np.asarray(flows, np.float32).reshape(-1, 2),
        float(np.median(errors)) if errors else float("inf"),
    )


def robust_mask(flows: np.ndarray) -> np.ndarray:
    """Filtra desplazamientos anomales dentro de un conjunto de estimaciones.

    Dado que el flujo optico puede producir algunos vectores claramente errados,
    esta funcion mide la distancia de cada estimacion respecto de la mediana del
    conjunto y aplica una regla robusta basada en MAD. La idea es aceptar los
    puntos que permanecen cerca del comportamiento dominante del grupo y rechazar
    los outliers que no son compatibles con el movimiento general del frame.
    """
    if len(flows) == 0:
        return np.zeros(0, dtype=bool)
    distances = np.linalg.norm(flows - np.median(flows, axis=0), axis=1)
    median = np.median(distances)
    mad = np.median(np.abs(distances - median))
    return distances <= max(2.5, median + 3 * 1.4826 * mad)


def relative_scale(old: np.ndarray, new: np.ndarray) -> float:
    """Calcula un factor de escala relativo entre dos configuraciones de puntos.

    Esta medicion intenta resumir si el conjunto de puntos se ha expandido o
    contraido respecto de una referencia local. Se toma el centroide de cada
    nube, se calculan radios respecto de ese centro y luego se compara la
    distribucion de distancias entre ambos conjuntos. El resultado es un factor
    multiplicativo que se usa para actualizar la escala relativa Z(t) del vuelo.
    """
    if len(old) < 3:
        return 1.0
    old_center, new_center = np.median(old, axis=0), np.median(new, axis=0)
    old_radius = np.linalg.norm(old - old_center, axis=1)
    new_radius = np.linalg.norm(new - new_center, axis=1)
    usable = old_radius > 1
    ratios = new_radius[usable] / old_radius[usable]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0.5) & (ratios < 2)]
    return float(np.median(ratios)) if len(ratios) else 1.0
