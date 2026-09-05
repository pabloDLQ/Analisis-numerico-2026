"""Exportacion y visualizacion de los resultados de la trayectoria."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def mostrar_resultados(data, mapa, salida, zoom_inicial, inliers_inicial, parametros):
    """Guarda el CSV y las graficas obtenidas por el pipeline."""
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    data = np.asarray(data, dtype=float)
    speed = calcular_rapidez_instantanea(data)
    altitude = 1.0 / np.maximum(data[:, 4], 1e-12)
    altitude /= altitude[0]

    csv_path = salida / "trayectoria_dron.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "frame", "tiempo_s", "x_mapa_px", "y_mapa_px", "zoom_relativo",
            "altitud_relativa", "rapidez_instantanea_px_s", "flujo_x_px",
            "flujo_y_px", "puntos_validos", "error_LK",
        ])
        writer.writerows(np.column_stack((data[:, :5], altitude, speed, data[:, 5:])))

    figure, axis = plt.subplots(figsize=(9, 6))
    axis.imshow(cv2.cvtColor(mapa, cv2.COLOR_BGR2RGB))
    axis.plot(data[:, 2], data[:, 3], color="#e8590c", linewidth=2)
    axis.scatter(data[0, 2], data[0, 3], color="#2b8a3e", label="Inicio", zorder=3)
    axis.set_title("Trayectoria del dron sobre el mapa")
    axis.set_xlabel("X del mapa (pixeles)")
    axis.set_ylabel("Y del mapa (pixeles)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(salida / "trayectoria_mapa.png", dpi=150)
    plt.close(figure)

    # Se representa la trayectoria en coordenadas cartesianas locales.
    x_cartesian = data[:, 2] - data[0, 2]
    y_cartesian = -(data[:, 3] - data[0, 3])
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(x_cartesian, y_cartesian, color="#e8590c", linewidth=2)
    axis.scatter(0.0, 0.0, color="#2b8a3e", label="Inicio y fin", zorder=3)
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_title("Trayectoria del dron en coordenadas cartesianas")
    axis.set_xlabel("Desplazamiento X (pixeles)")
    axis.set_ylabel("Desplazamiento Y (pixeles)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(salida / "trayectoria_cartesiana.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(data[:, 1], data[:, 4], color="#2b8a3e")
    axis.set_title("Zoom relativo en funcion del tiempo")
    axis.set_xlabel("Tiempo (s)")
    axis.set_ylabel("Zoom relativo Z(t)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(salida / "zoom_relativo.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(data[:, 1], altitude, color="#9c36b5")
    axis.set_title("Altitud relativa en funcion del tiempo")
    axis.set_xlabel("Tiempo (s)")
    axis.set_ylabel("Altitud relativa H(t)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(salida / "altitud_relativa.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(data[:, 1], np.hypot(data[:, 5], data[:, 6]), color="#1c7ed6")
    axis.set_title("Magnitud del flujo optico de Lucas-Kanade")
    axis.set_xlabel("Tiempo (s)")
    axis.set_ylabel("Flujo (pixeles/frame)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(salida / "flujo_optico.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(data[:, 1], speed, color="#d9480f")
    axis.set_title("Rapidez escalar instantanea")
    axis.set_xlabel("Tiempo (s)")
    axis.set_ylabel("Rapidez (pixeles/s)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(salida / "rapidez_instantanea.png", dpi=150)
    plt.close(figure)

    (salida / "parametros_extraccion.txt").write_text(
        f"zoom inicial mapa/frame: {zoom_inicial:.6f}\n"
        f"inliers del registro inicial: {inliers_inicial}\n"
        f"parametros: {parametros}\n",
        encoding="utf-8",
    )
    return csv_path


def calcular_rapidez_instantanea(data):
    """Calcula la rapidez a partir de la trayectoria mediante diferencias."""
    time = data[:, 1]
    velocity_x = np.gradient(data[:, 2], time)
    velocity_y = np.gradient(data[:, 3], time)
    return np.hypot(velocity_x, velocity_y)
