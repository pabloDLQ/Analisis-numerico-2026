"""Pipeline principal de extraccion y limpieza de la trayectoria del dron.

Este modulo orquesta el orden de ejecucion del algoritmo completo:

1) Lectura del mapa y del video.
2) Deteccion de puntos de interes.
3) Flujo optico por Lucas-Kanade entre frames consecutivos.
4) Registro geométrico del frame contra el mapa satelital.
5) Estimacion y suavizado de la escala relativa Z(t).
6) Limpieza de la trayectoria en X/Y para eliminar saltos y ruido.
7) Exportacion de resultados (CSV y graficos).

La idea es que el flujo principal quede legible y que cada etapa quede
encapsulada en un archivo reutilizable, con responsabilidades separadas.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .datos_trayectoria import Trajectory
from .deteccion_puntos import detect_features
from .entrada_salida import DEFAULT_MAP, DEFAULT_OUTPUT, DEFAULT_VIDEO, save_results
from .escala_relativa import _rebuild_relative_scale
from .flujo_optico_lk import lucas_kanade, relative_scale, robust_mask
from .postprocesado_trayectoria import _fix_xy_outliers
from .registro_mapa import register_frame


def extract_trajectory(video_path=DEFAULT_VIDEO, map_path=DEFAULT_MAP, max_corners=300, registration_step=5):
    # Se abre el mapa y el video para validar que existan. Luego se toma el
    # primer frame como referencia inicial para comparar con los siguientes y
    # construir la trayectoria desde una base estable.
    map_gray = cv2.imread(str(map_path), cv2.IMREAD_GRAYSCALE)
    if map_gray is None:
        raise FileNotFoundError(f"No se pudo abrir el mapa: {map_path}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 1.0
    ok, frame = capture.read()
    if not ok:
        capture.release()
        raise ValueError("El video no contiene frames legibles")
    previous = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    # Se seleccionan esquinas con suficiente textura para que la solucion del
    # sistema de Lucas-Kanade sea numericamente confiable y no dependa de zonas
    # planas o mal condicionadas.
    points = detect_features(previous, max_corners)
    orb = cv2.ORB_create(nfeatures=2500, fastThreshold=8)
    map_keypoints, map_descriptors = orb.detectAndCompute(map_gray, None)
    if map_descriptors is None:
        capture.release()
        raise ValueError("No se encontraron caracteristicas en el mapa")

    # Se preparan las variables que sostendran la posicion global del dron, el
    # centro del frame y el estado de registro durante todo el recorrido.
    image_height, image_width = previous.shape
    center = np.float32([[[image_width / 2, image_height / 2]]])
    map_position = np.array([np.nan, np.nan])
    start_position = None
    homography = None
    rows = [[0, 0.0, np.nan, np.nan, 1.0, 0, 0, 1.0, len(points), 0, 0, 0]]
    z_value = 1.0
    frame_number = 0

    # Para cada frame nuevo se estima el desplazamiento local entre imagenes,
    # se filtran outliers y se actualiza la escala relativa. Si el tracking se
    # vuelve poco estable, se re-detectan puntos de interes para recuperar la
    # continuidad del movimiento.
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_number += 1
        current = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)

        old_valid, new_valid, flows, lk_error = lucas_kanade(previous, current, points)
        mask = robust_mask(flows)
        if mask.sum() >= 8:
            old_inliers, new_inliers = old_valid[mask], new_valid[mask]
            flow = np.median(flows[mask], axis=0)
            scale_step = relative_scale(old_inliers, new_inliers)
            z_value *= scale_step
            u, v = map(float, flow)
            points = new_inliers
        else:
            old_inliers, new_inliers = old_valid[:0], new_valid[:0]
            scale_step, u, v = 1.0, 0.0, 0.0

        if len(points) < 25:
            points = detect_features(current, max_corners)

        # El registro contra el mapa se hace cada cierto numero de frames usando
        # ORB y homografia robusta. Cuando la alineacion es valida, se proyecta el
        # centro del frame para recuperar la posicion global sobre la imagen satelital.
        map_inliers = 0
        if frame_number % registration_step == 0 or homography is None:
            new_homography, map_inliers = register_frame(current, map_gray, orb, map_keypoints, map_descriptors)
            if new_homography is not None:
                homography = new_homography
        if homography is not None:
            projected = cv2.perspectiveTransform(center, homography)[0, 0]
            if np.isfinite(projected).all():
                map_position = projected.astype(float)
                if start_position is None:
                    start_position = map_position.copy()

        # Se acumula la informacion de cada frame para poder reconstruir luego la
        # trayectoria completa en X, Y, Z, velocidad y estadisticos de confianza.
        rows.append([
            frame_number, frame_number / fps, map_position[0], map_position[1],
            z_value, u, -v, scale_step, len(new_inliers), lk_error, map_inliers,
            int(homography is not None)
        ])
        previous = current

    # Se cierra la trayectoria y se ajusta el desplazamiento final para que la
    # posicion de aterrizaje sea consistente con la pose inicial del dron.
    capture.release()
    data = np.asarray(rows, float)
    if start_position is None:
        raise RuntimeError("No se pudo registrar ningun frame contra el mapa")
    data[0, 2:4] = start_position
    closure = data[-1, 2:4] - start_position
    data[:, 2:4] -= np.linspace(0, 1, len(data))[:, None] * closure
    return Trajectory(*[data[:, i].astype(dtype) for i, dtype in enumerate([int, float, float, float, float, float, float, float, int, float, int, int])])


def postprocess_trajectory(trajectory: Trajectory) -> Trajectory:
    # La trayectoria inicial puede contener saltos y ruido de registro. Aqui se
    # limpia la geometra en X/Y y se reconstruye la escala relativa Z(t) con una
    # version mas estable para la interpretacion cinematica final.
    x_fixed, y_fixed = _fix_xy_outliers(trajectory)
    z_fixed = _rebuild_relative_scale(trajectory)
    return Trajectory(
        frame=trajectory.frame,
        time_s=trajectory.time_s,
        x_map=x_fixed,
        y_map=y_fixed,
        z=z_fixed,
        u=trajectory.u,
        v=trajectory.v,
        scale_step=trajectory.scale_step,
        valid_points=trajectory.valid_points,
        lk_error=trajectory.lk_error,
        map_inliers=trajectory.map_inliers,
        registered=trajectory.registered,
    )


def main():
    # Este bloque solo sirve para ejecutar el pipeline desde la terminal. La
    # logica principal del algoritmo ya se encuentra encapsulada en las funciones
    # de extraccion y postprocesado, y aqui solo se parsean los parametros de uso.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--mapa", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--salida", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--paso-registro", type=int, default=5)
    args = parser.parse_args()

    trajectory = extract_trajectory(args.video, args.mapa, registration_step=args.paso_registro)
    trajectory = postprocess_trajectory(trajectory)
    csv_path = save_results(trajectory, args.mapa, args.salida)
    print(f"Frames procesados: {len(trajectory.frame)}")
    print(f"Frames con registro de mapa: {int(trajectory.registered.sum())}")
    print(f"Cierre: {np.linalg.norm(trajectory.x_map[-1:] - trajectory.x_map[:1]) + np.linalg.norm(trajectory.y_map[-1:] - trajectory.y_map[:1]):.3f} pixeles")
    print(csv_path.resolve())


if __name__ == "__main__":
    main()
