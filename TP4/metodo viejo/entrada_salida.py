"""Carga de datos y exportacion de resultados.

Esta capa no resuelve la dinamica del vuelo: solo lee el video, lee el mapa,
genera el CSV con la trayectoria y guarda las graficas principales. La separacion
permite reutilizar la parte matematica del tracking sin mezclar la logica de
persistencia y visualizacion con la cinemática.

Aqui se centraliza toda la parte de archivos y graficos: se escriben los datos
procesados en formato tabular y se generan las visualizaciones principales para
que la interpretacion del resultado quede separada de la logica numerica.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = ROOT / "data" / "vuelo_dron.mp4"
DEFAULT_MAP = ROOT / "data" / "mapa_satelital_completo.jpg"
DEFAULT_OUTPUT = ROOT / "resultados"


def save_results(trajectory, map_path=DEFAULT_MAP, output_dir=DEFAULT_OUTPUT):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "trayectoria_mapa.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "frame", "tiempo_s", "X_mapa_px", "Y_mapa_px", "Z_relativa",
            "u_LK", "v_LK", "escala_frame", "puntos_validos", "error_LK",
            "inliers_mapa", "registrado"
        ])
        writer.writerows(zip(
            trajectory.frame, trajectory.time_s, trajectory.x_map, trajectory.y_map,
            trajectory.z, trajectory.u, trajectory.v, trajectory.scale_step,
            trajectory.valid_points, trajectory.lk_error, trajectory.map_inliers,
            trajectory.registered
        ))
    y_cartesian = -trajectory.y_map
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(trajectory.x_map, y_cartesian, color="#e8590c")
    axis.set_aspect("equal")
    axis.set_title("Trayectoria X-Y sobre el mapa satelital (eje Y cartesiano)")
    axis.set_xlabel("X en mapa (pixeles)"); axis.set_ylabel("Y cartesiana (pixeles)"); axis.grid(alpha=0.25)
    figure.tight_layout(); figure.savefig(output_dir / "trayectoria_xy_mapa.png", dpi=150); plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(trajectory.time_s, trajectory.z, color="#2b8a3e")
    axis.set_title("Escala de altitud/zoom relativo Z(t)")
    axis.set_xlabel("Tiempo (s)"); axis.set_ylabel("Z(t) relativa"); axis.grid(alpha=0.25)
    figure.tight_layout(); figure.savefig(output_dir / "escala_z.png", dpi=150); plt.close(figure)
    return csv_path
