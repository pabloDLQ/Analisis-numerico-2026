"""Extraccion de trayectoria de un dron mediante Lucas-Kanade."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .mostrar_resultados import mostrar_resultados


# -----------------------------------------------------------------------------
# Parametros computacionales editables del algoritmo.
# -----------------------------------------------------------------------------
PARAMETROS = {
    "winSize": (21, 21),
    "maxLevel": 3,
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    "qualityLevel": 0.01,
    "minDistance": 8,
    "blockSize": 7,
    "maxCorners": 500,
    "min_puntos_validos": 12,
}

ROOT = Path(__file__).resolve().parents[1]
VIDEO_DEFAULT = ROOT / "data" / "vuelo_dron.mp4"
MAPA_DEFAULT = ROOT / "data" / "mapa_satelital_completo.jpg"
SALIDA_DEFAULT = ROOT / "resultados"


# -----------------------------------------------------------------------------
# Deteccion de puntos con textura suficiente para resolver Lucas-Kanade.
# -----------------------------------------------------------------------------
def detectar_puntos(gray: np.ndarray) -> np.ndarray:
    puntos = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=PARAMETROS["maxCorners"],
        qualityLevel=PARAMETROS["qualityLevel"],
        minDistance=PARAMETROS["minDistance"],
        blockSize=PARAMETROS["blockSize"],
    )
    if puntos is None:
        return np.empty((0, 2), dtype=np.float32)
    return puntos.reshape(-1, 2).astype(np.float32)


# -----------------------------------------------------------------------------
# Seguimiento de puntos mediante el algoritmo piramidal de Lucas-Kanade.
# -----------------------------------------------------------------------------
def seguir_lucas_kanade(previous, current, points):
    if len(points) == 0:
        return points, points, np.empty((0,), dtype=bool), float("inf")

    old = points.reshape(-1, 1, 2)
    new, status, errors = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        old,
        None,
        winSize=PARAMETROS["winSize"],
        maxLevel=PARAMETROS["maxLevel"],
        criteria=PARAMETROS["criteria"],
    )
    if new is None or status is None:
        empty = np.zeros(len(points), dtype=bool)
        return points, points.copy(), empty, float("inf")

    valid = status.reshape(-1).astype(bool)
    new_flat = new.reshape(-1, 2)
    displacement = new_flat - points
    valid &= np.isfinite(new_flat).all(axis=1)
    valid &= np.linalg.norm(displacement, axis=1) < 80.0

    # MAD: conserva el movimiento dominante y descarta correspondencias anomales.
    if valid.sum() >= 8:
        median_flow = np.median(displacement[valid], axis=0)
        distances = np.linalg.norm(displacement - median_flow, axis=1)
        median_distance = np.median(distances[valid])
        mad = np.median(np.abs(distances[valid] - median_distance))
        valid &= distances <= max(3.0, median_distance + 3.0 * 1.4826 * mad)

    error_values = errors.reshape(-1)[valid] if errors is not None else np.array([])
    error = float(np.median(error_values)) if len(error_values) else float("inf")
    return points, new_flat, valid, error


# -----------------------------------------------------------------------------
# Estimacion del zoom relativo a partir de la expansion/contraccion de puntos.
# -----------------------------------------------------------------------------
def estimar_zoom_relativo(old_points, new_points) -> float:
    if len(old_points) < 4:
        return 1.0
    old_center = np.median(old_points, axis=0)
    new_center = np.median(new_points, axis=0)
    old_radius = np.linalg.norm(old_points - old_center, axis=1)
    new_radius = np.linalg.norm(new_points - new_center, axis=1)
    usable = old_radius > 2.0
    ratios = new_radius[usable] / old_radius[usable]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0.7) & (ratios < 1.4)]
    return float(np.median(ratios)) if len(ratios) else 1.0


# -----------------------------------------------------------------------------
# Interpolacion cubica local implementada.
# -----------------------------------------------------------------------------
def interpolar_cubica(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    indexes = np.arange(len(result), dtype=float)
    good = np.flatnonzero(valid & np.isfinite(result))
    if len(good) < 4:
        return np.interp(indexes, good, result[good]) if len(good) >= 2 else result

    for index in np.flatnonzero(~valid | ~np.isfinite(result)):
        nearby = good[np.argsort(np.abs(good - index))[:4]]
        nearby.sort()
        coefficients = np.polyfit(nearby.astype(float), result[nearby], 3)
        result[index] = np.polyval(coefficients, float(index))
    return result


# -----------------------------------------------------------------------------
# Cierre de la trayectoria: reparte linealmente el error final para que el
# recorrido comience y termine exactamente en la misma coordenada del mapa.
# -----------------------------------------------------------------------------
def cerrar_trayectoria(data: np.ndarray) -> np.ndarray:
    closed = data.copy()
    correction = closed[-1, 2:4] - closed[0, 2:4]
    fraction = np.linspace(0.0, 1.0, len(closed))[:, None]
    closed[:, 2:4] -= fraction * correction
    closed[-1, 2:4] = closed[0, 2:4]
    return closed


# -----------------------------------------------------------------------------
# Registro del frame 0 contra el mapa mediante caracteristicas de OpenCV.
# -----------------------------------------------------------------------------
def registrar_frame_inicial(frame_gray, mapa_gray):
    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=5000)
        norm_type = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(nfeatures=5000)
        norm_type = cv2.NORM_HAMMING

    key_frame, descriptors_frame = detector.detectAndCompute(frame_gray, None)
    key_map, descriptors_map = detector.detectAndCompute(mapa_gray, None)
    if descriptors_frame is None or descriptors_map is None:
        raise RuntimeError("No se encontraron caracteristicas para registrar el frame 0")

    matcher = cv2.BFMatcher(norm_type)
    pairs = matcher.knnMatch(descriptors_frame, descriptors_map, k=2)
    good = [first for first, second in pairs if first.distance < 0.75 * second.distance]
    if len(good) < 4:
        raise RuntimeError("No hay suficientes correspondencias entre el frame 0 y el mapa")

    source = np.float32([key_frame[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    destination = np.float32([key_map[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(source, destination, cv2.RANSAC, 5.0)
    if homography is None or mask is None or int(mask.sum()) < 4:
        raise RuntimeError("La homografia del frame 0 no es confiable")
    return homography, int(mask.sum())


# -----------------------------------------------------------------------------
# Escritura de CSV y generacion de las graficas solicitadas.
# -----------------------------------------------------------------------------
def guardar_resultados(rows, mapa, salida, zoom_inicial, inliers_inicial):
    return mostrar_resultados(
        rows, mapa, salida, zoom_inicial, inliers_inicial, PARAMETROS
    )


# -----------------------------------------------------------------------------
# Pipeline completo: carga, registro inicial, seguimiento y exportacion.
# -----------------------------------------------------------------------------
def ejecutar_extraccion(video_path=VIDEO_DEFAULT, mapa_path=MAPA_DEFAULT, salida=SALIDA_DEFAULT):
    mapa = cv2.imread(str(mapa_path), cv2.IMREAD_COLOR)
    capture = cv2.VideoCapture(str(video_path))
    if mapa is None:
        raise FileNotFoundError(f"No se pudo abrir el mapa: {mapa_path}")
    if not capture.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 1.0
    ok, frame = capture.read()
    if not ok:
        capture.release()
        raise ValueError("El video no contiene frames legibles")

    previous = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mapa_gray = cv2.cvtColor(mapa, cv2.COLOR_BGR2GRAY)
    homography, initial_inliers = registrar_frame_inicial(previous, mapa_gray)
    height, width = previous.shape
    center = np.array([[[width / 2.0, height / 2.0]]], dtype=np.float32)
    corners = np.array([[[0, 0], [width, 0], [width, height], [0, height]]], dtype=np.float32)
    projected_corners = cv2.perspectiveTransform(corners, homography)[0]
    initial_area = abs(cv2.contourArea(projected_corners.astype(np.float32)))
    initial_zoom = float(np.sqrt(initial_area / (width * height)))
    start_position = cv2.perspectiveTransform(center, homography)[0, 0].astype(float)

    points = detectar_puntos(previous)
    cumulative_flow = np.zeros(2, dtype=float)
    zoom = 1.0
    rows = [[0, 0.0, start_position[0], start_position[1], zoom, 0.0, 0.0, len(points), 0.0]]
    valid_rows = [True]
    frame_number = 0

    # El flujo de la escena tiene signo opuesto al desplazamiento de la camara.
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_number += 1
        current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        old_points, new_points, valid, lk_error = seguir_lucas_kanade(previous, current, points)
        valid_count = int(valid.sum())
        reliable = valid_count >= PARAMETROS["min_puntos_validos"]
        if reliable:
            tracked_old = old_points[valid]
            tracked_new = new_points[valid]
            flow = np.median(tracked_new - tracked_old, axis=0)
            cumulative_flow += flow
            zoom *= np.clip(estimar_zoom_relativo(tracked_old, tracked_new), 0.97, 1.03)
            points = tracked_new
        else:
            flow = np.zeros(2, dtype=float)
            points = np.empty((0, 2), dtype=np.float32)
        if len(points) < PARAMETROS["maxCorners"] // 3:
            points = detectar_puntos(current)

        map_point = cv2.perspectiveTransform(
            (center - cumulative_flow.reshape(1, 1, 2)).astype(np.float32), homography
        )[0, 0]
        rows.append([
            frame_number, frame_number / fps, map_point[0], map_point[1], zoom,
            flow[0], flow[1], valid_count, lk_error,
        ])
        valid_rows.append(reliable)
        previous = current

    capture.release()
    data = np.asarray(rows, dtype=float)
    reliable_array = np.asarray(valid_rows, dtype=bool)
    for column in (2, 3, 4):
        data[:, column] = interpolar_cubica(data[:, column], reliable_array)
    data = cerrar_trayectoria(data)
    csv_path = guardar_resultados(data, mapa, Path(salida), initial_zoom, initial_inliers)
    print(f"Frames procesados: {len(data)}")
    print(f"Inliers del registro inicial: {initial_inliers}")
    print(f"Coordenada inicial en el mapa: ({data[0, 2]:.2f}, {data[0, 3]:.2f})")
    print(f"Zoom inicial mapa/frame: {initial_zoom:.6f}")
    print(f"Resultados: {csv_path.resolve()}")
    return data


if __name__ == "__main__":
    ejecutar_extraccion()