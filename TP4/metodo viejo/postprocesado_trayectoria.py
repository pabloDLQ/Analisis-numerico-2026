"""Limpieza y suavizado final de la trayectoria estimada.

Este modulo elimina saltos bruscos, picos aislados y ruido estructural en las
coordenadas X/Y del mapa. La idea no es cambiar el movimiento real del dron,
sino remover artefactos de registro, homografia inestable y ruido numerico que
aparecen al proyectar la trayectoria sobre el mapa.

La funcion principal detecta desplazamientos anomales, reemplaza valores
volatiles por interpolacion y luego aplica un suavizado conservador para que la
trayectoria quede ordenada sin perder la forma general del vuelo.
"""

from __future__ import annotations

import numpy as np

from .datos_trayectoria import Trajectory


def _fix_xy_outliers(trajectory: Trajectory, jump_threshold: float = 55.0, min_inliers: int = 18):
    x = trajectory.x_map.astype(float).copy()
    y = trajectory.y_map.astype(float).copy()
    inliers = trajectory.map_inliers.astype(int)

    jumps = np.hypot(np.diff(x), np.diff(y))
    bad = np.zeros(len(x), dtype=bool)

    # Se descartan saltos bruscos cuando el registro del mapa tiene poca
    # confianza o cuando la variacion de posicion excede los limites plausibles.
    if len(jumps) > 0:
        conflict = np.where((jumps > jump_threshold) & (inliers[1:] < min_inliers))[0] + 1
        bad[conflict] = True

    if len(jumps) > 10:
        base = float(np.median(jumps))
        mad = float(np.median(np.abs(jumps - base)))
        dynamic_limit = max(jump_threshold, base + 4.0 * 1.4826 * mad)
        high = np.where(jumps > dynamic_limit)[0] + 1
        bad[high] = True

    # Se limpian picos aislados donde un salto se corrige al frame siguiente y
    # deja la trayectoria con un patron no fisico de ida y vuelta instantanea.
    for i in range(1, len(x) - 1):
        left = np.hypot(x[i] - x[i - 1], y[i] - y[i - 1])
        right = np.hypot(x[i + 1] - x[i], y[i + 1] - y[i])
        bridge = np.hypot(x[i + 1] - x[i - 1], y[i + 1] - y[i - 1])
        if left > 35.0 and right > 35.0 and bridge < 25.0:
            bad[i] = True

    for i in range(1, len(x) - 1):
        left = np.hypot(x[i] - x[i - 1], y[i] - y[i - 1])
        right = np.hypot(x[i + 1] - x[i], y[i + 1] - y[i])
        if inliers[i] < min_inliers and (left > 35.0 or right > 35.0):
            bad[i] = True

    bad[0] = False
    bad[-1] = False
    good = ~bad
    if good.sum() < 2:
        return x, y

    idx = np.arange(len(x), dtype=float)
    x = np.interp(idx, idx[good], x[good])
    y = np.interp(idx, idx[good], y[good])

    # Luego de eliminar los valores sospechosos, se aplica un filtro mediano y
    # suavizado lineal para quitar dientes o ruido de alta frecuencia sin perder
    # la estructura general de la trayectoria.
    for _ in range(2):
        x_pad = np.pad(x, (3, 3), mode="edge")
        y_pad = np.pad(y, (3, 3), mode="edge")
        x = np.array([np.median(x_pad[i:i + 7]) for i in range(len(x))], dtype=float)
        y = np.array([np.median(y_pad[i:i + 7]) for i in range(len(y))], dtype=float)

    kernel = np.array([1, 2, 3, 4, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    x_pad = np.pad(x, (3, 3), mode="edge")
    y_pad = np.pad(y, (3, 3), mode="edge")
    x = np.convolve(x_pad, kernel, mode="valid")
    y = np.convolve(y_pad, kernel, mode="valid")
    return x, y
