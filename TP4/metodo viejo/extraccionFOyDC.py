"""Orquestador del pipeline de trazado de trayectoria del dron.

Orden:
1) Extraccion: detectar puntos, estimar flujo optico y registrar cada frame con
   el mapa para reconstruir la trayectoria inicial en coordenadas del mundo.
2) Postprocesado: limpiar saltos y ruido en X/Y y reconstruir Z(t) de forma
   estable para que la semantica cinematica sea consistente.
3) Exportacion: guardar el CSV y las graficas finales para que los resultados
   queden listos para analisis.

La implementacion concreta de cada etapa se separa en otros archivos dentro de
src/, pero la ejecucion general se mantiene aqui para que el orden sea
legible y no se pierda entre funciones cargadas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .entrada_salida import DEFAULT_MAP, DEFAULT_OUTPUT, DEFAULT_VIDEO, save_results
from .pipeline_trayectoria import extract_trajectory, postprocess_trajectory


def ejecutar_pipeline(salida: Path = DEFAULT_OUTPUT, paso_registro: int = 5):
    video = DEFAULT_VIDEO
    mapa = DEFAULT_MAP

    # Orden de ejecucion del pipeline:
    # 1. EXTRAER: se calcula la trayectoria inicial usando flujo optico y registro
    #    al mapa. Esto genera los datos crudos, aunque aun puedan tener ruido.
    # 2. POSTPROCESAR: se corrigen saltos geométricos y se reconstruye la escala
    #    Z(t) de forma robusta para estabilizar la cinématica del dron.
    # 3. EXPORTAR: se guardan los resultados finales en CSV y graficos para analisis.
    trajectory = extract_trajectory(video, mapa, registration_step=paso_registro)
    trajectory = postprocess_trajectory(trajectory)
    csv_path = save_results(trajectory, mapa, salida)

    print(f"Frames procesados: {len(trajectory.frame)}")
    print(f"Frames con registro de mapa: {int(trajectory.registered.sum())}")
    print(
        f"Cierre: "
        f"{np.linalg.norm(trajectory.x_map[-1:] - trajectory.x_map[:1]) + np.linalg.norm(trajectory.y_map[-1:] - trajectory.y_map[:1]):.3f} pixeles"
    )
    print(csv_path.resolve())

    return csv_path
