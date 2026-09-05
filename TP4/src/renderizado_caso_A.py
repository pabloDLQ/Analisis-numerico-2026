"""Renderizado del caso A sobre el mapa satelital."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .modelado_trayectoria import (
    RUTA_CSV as RUTA_CSV_MODELO,
    SALTO_NODOS,
    cargar_datos_y_nodos,
    spline_cubico_natural,
)


# -----------------------------------------------------------------------------
# Rutas y parametros editables del renderizado.
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RUTA_CSV = RUTA_CSV_MODELO
RUTA_MAPA = ROOT / "data" / "mapa_satelital_completo.jpg"
RUTA_VIDEO_ORIGINAL = ROOT / "data" / "vuelo_dron.mp4"
DIRECTORIO_SALIDA = ROOT / "resultados" / "resultados_ej3"
RUTA_VIDEO = DIRECTORIO_SALIDA / "renderizado_caso_A.mp4"
FPS_SALIDA = 30.0
ANCHO_PANEL = 600
ALTO_PANEL = 600


# -----------------------------------------------------------------------------
# Carga la trayectoria modelada y el zoom del CSV de extraccion.
# -----------------------------------------------------------------------------
def cargar_trayectoria_y_zoom(ruta_csv: Path):
    t, x, y, _, t_nodos, x_nodos, y_nodos, _ = cargar_datos_y_nodos(
        ruta_csv, SALTO_NODOS
    )
    zoom = np.genfromtxt(
        ruta_csv,
        delimiter=",",
        skip_header=1,
        usecols=(4,),
        dtype=float,
    )
    zoom = np.atleast_1d(zoom)
    if len(t) < 2 or len(zoom) != len(t):
        raise ValueError("La trayectoria y el zoom no tienen la misma cantidad de muestras")
    if not np.all(np.isfinite(t)) or not np.all(np.diff(t) > 0):
        raise ValueError("Los tiempos deben ser finitos y estrictamente crecientes")
    if not np.all(np.isfinite(zoom)) or np.any(zoom <= 0):
        raise ValueError("El zoom relativo debe contener valores positivos y finitos")

    x_spline = spline_cubico_natural(t, t_nodos, x_nodos)
    y_spline = spline_cubico_natural(t, t_nodos, y_nodos)
    if not np.all(np.isfinite(x_spline)) or not np.all(np.isfinite(y_spline)):
        raise ValueError("El spline produjo coordenadas no finitas")
    return t, x, y, x_spline, y_spline, zoom


# -----------------------------------------------------------------------------
# Obtiene la escala base del mapa a partir de la huella del frame 0 original.
# -----------------------------------------------------------------------------
def obtener_referencia_frame0(ruta_video: Path, mapa):
    video = cv2.VideoCapture(str(ruta_video))
    ok, frame0 = video.read()
    video.release()
    if not ok or frame0 is None:
        raise ValueError("No se pudo leer el frame 0 del video original")
    if frame0.shape[1] != ANCHO_PANEL or frame0.shape[0] != ALTO_PANEL:
        raise ValueError("El video original no tiene la resolucion esperada de 600x600")

    gray_frame = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    gray_map = cv2.cvtColor(mapa, cv2.COLOR_BGR2GRAY)
    detector = cv2.SIFT_create(nfeatures=5000)
    key_frame, descriptors_frame = detector.detectAndCompute(gray_frame, None)
    key_map, descriptors_map = detector.detectAndCompute(gray_map, None)
    if descriptors_frame is None or descriptors_map is None:
        raise RuntimeError("No se pudieron detectar caracteristicas en el frame 0 o mapa")

    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptors_frame, descriptors_map, k=2)
    good = [first for first, second in pairs if first.distance < 0.75 * second.distance]
    if len(good) < 4:
        raise RuntimeError("No hay correspondencias suficientes para fijar la escala")
    source = np.float32([key_frame[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    destination = np.float32([key_map[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(source, destination, cv2.RANSAC, 5.0)
    if homography is None or mask is None or int(mask.sum()) < 4:
        raise RuntimeError("La homografia del frame 0 no es confiable")

    corners = np.float32([[[0, 0], [ANCHO_PANEL, 0], [ANCHO_PANEL, ALTO_PANEL], [0, ALTO_PANEL]]])
    projected = cv2.perspectiveTransform(corners, homography)[0]
    base_width = float(np.ptp(projected[:, 0]))
    base_height = float(np.ptp(projected[:, 1]))
    center = projected.mean(axis=0)
    return frame0, center, base_width, base_height


# -----------------------------------------------------------------------------
# Remuestrea el spline por longitud de arco para que el desplazamiento entre
# frames sea uniforme y la simulacion tenga rapidez espacial constante.
# -----------------------------------------------------------------------------
def remuestrear_velocidad_constante(t, x, y, zoom, t_nodos, x_nodos, y_nodos):
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    zoom = np.asarray(zoom, dtype=float)
    if len(t) < 2 or not (len(t) == len(x) == len(y) == len(zoom)):
        raise ValueError("Las muestras temporales, espaciales y de zoom deben coincidir")

    # La tabla densa reduce el error de cuerda entre frames y representa la
    # longitud del spline, no la de los 750 puntos originales solamente.
    t_denso = np.linspace(t[0], t[-1], max(10000, 100 * len(t)))
    x_denso = spline_cubico_natural(t_denso, t_nodos, x_nodos)
    y_denso = spline_cubico_natural(t_denso, t_nodos, y_nodos)
    tramos = np.hypot(np.diff(x_denso), np.diff(y_denso))
    longitud = np.concatenate(([0.0], np.cumsum(tramos)))
    if longitud[-1] <= 0.0:
        raise ValueError("La trayectoria no tiene longitud suficiente para renderizarse")

    # La velocidad espacial se fija a partir de la longitud total y la
    # duracion total. Cada instante recibe exactamente s(t) = v0 * (t - t0).
    velocidad_constante = longitud[-1] / (t[-1] - t[0])
    distancia_objetivo = velocidad_constante * (t - t[0])
    x_uniforme = np.interp(distancia_objetivo, longitud, x_denso)
    y_uniforme = np.interp(distancia_objetivo, longitud, y_denso)
    zoom_uniforme = zoom.copy()
    return x_uniforme, y_uniforme, zoom_uniforme, velocidad_constante


# -----------------------------------------------------------------------------
# Comprueba la condicion v(t) = v0 en las muestras discretas del video.
# -----------------------------------------------------------------------------
def validar_velocidad_constante(t, velocidad_constante):
    tiempos = np.asarray(t, dtype=float)
    longitudes_objetivo = velocidad_constante * (tiempos - tiempos[0])
    velocidades = np.diff(longitudes_objetivo) / np.diff(tiempos)
    if not np.allclose(velocidades, velocidad_constante, rtol=1e-12, atol=1e-12):
        raise RuntimeError("La parametrizacion de longitud de arco no es constante")
    return float(velocidad_constante)


# -----------------------------------------------------------------------------
# Utilidades de transformacion y dibujo de puntos en un panel de video.
# -----------------------------------------------------------------------------
def transformar_puntos(points, x0, y0, width, height, panel_width, panel_height):
    points = np.asarray(points, dtype=float)
    transformed = np.empty_like(points)
    transformed[:, 0] = panel_width * (points[:, 0] - x0) / width
    transformed[:, 1] = panel_height * (points[:, 1] - y0) / height
    return np.rint(transformed).astype(np.int32)


def dibujar_curva(panel, points, color, thickness=2):
    if len(points) >= 2:
        cv2.polylines(panel, [points.reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)


def transformar_puntos_globales(points, scale, offset_x, offset_y):
    points = np.asarray(points, dtype=float)
    transformed = points * scale
    transformed[:, 0] += offset_x
    transformed[:, 1] += offset_y
    return np.rint(transformed).astype(np.int32)


# -----------------------------------------------------------------------------
# Construye un frame unicamente con la simulacion local del dron.
# -----------------------------------------------------------------------------
def construir_frame(mapa, frame0, x, y, zoom, indice, base_width, base_height):
    if indice == 0:
        return frame0.copy()

    current_zoom = float(zoom[indice])
    crop_width = min(mapa.shape[1], base_width / max(current_zoom, 1e-6))
    crop_height = min(mapa.shape[0], base_height / max(current_zoom, 1e-6))
    center_x, center_y = x[indice], y[indice]
    x0 = float(np.clip(center_x - crop_width / 2.0, 0.0, mapa.shape[1] - crop_width))
    y0 = float(np.clip(center_y - crop_height / 2.0, 0.0, mapa.shape[0] - crop_height))
    x1, y1 = x0 + crop_width, y0 + crop_height
    roi = mapa[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))]
    panel_local = cv2.resize(roi, (ANCHO_PANEL, ALTO_PANEL), interpolation=cv2.INTER_AREA)

    return panel_local


# -----------------------------------------------------------------------------
# Genera el video MP4 completo y verifica que el archivo pueda reabrirse.
# -----------------------------------------------------------------------------
def renderizar_caso_A(ruta_csv=RUTA_CSV, ruta_mapa=RUTA_MAPA, ruta_video=RUTA_VIDEO):
    mapa = cv2.imread(str(ruta_mapa), cv2.IMREAD_COLOR)
    if mapa is None:
        raise FileNotFoundError(f"No se pudo abrir el mapa: {ruta_mapa}")
    frame0, _, base_width, base_height = obtener_referencia_frame0(
        RUTA_VIDEO_ORIGINAL, mapa
    )
    t, _, _, x_spline, y_spline, zoom = cargar_trayectoria_y_zoom(Path(ruta_csv))
    _, _, _, _, t_nodos, x_nodos, y_nodos, _ = cargar_datos_y_nodos(
        Path(ruta_csv), SALTO_NODOS
    )
    x, y, zoom, velocidad_constante = remuestrear_velocidad_constante(
        t, x_spline, y_spline, zoom, t_nodos, x_nodos, y_nodos
    )
    zoom = zoom / zoom[0]
    velocidad_constante = validar_velocidad_constante(t, velocidad_constante)

    ruta_video = Path(ruta_video)
    ruta_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(ruta_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS_SALIDA,
        (ANCHO_PANEL, ALTO_PANEL),
    )
    if not writer.isOpened():
        raise RuntimeError(f"No se pudo abrir el escritor de video: {ruta_video}")

    try:
        for indice in range(len(t)):
            frame = construir_frame(
                mapa, frame0, x, y, zoom, indice, base_width, base_height
            )
            if frame.shape != (ALTO_PANEL, ANCHO_PANEL, 3) or frame.dtype != np.uint8:
                raise RuntimeError("El frame generado tiene dimensiones o tipo invalidos")
            writer.write(frame)
    finally:
        writer.release()

    video = cv2.VideoCapture(str(ruta_video))
    ok, first_frame = video.read()
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    if not ok or first_frame is None:
        raise RuntimeError("El video generado no se pudo reabrir")
    if (width, height) != (ANCHO_PANEL, ALTO_PANEL):
        raise RuntimeError("El video generado tiene una resolucion inesperada")
    if frame_count != len(t):
        raise RuntimeError(f"El video contiene {frame_count} frames; se esperaban {len(t)}")
    print(f"Video generado: {ruta_video.resolve()}")
    print(f"Frames: {frame_count} | Resolucion: {width}x{height} | FPS: {FPS_SALIDA:.1f}")
    print(f"Rapidez espacial constante: {velocidad_constante:.6f} pixeles/s")
    return ruta_video


if __name__ == "__main__":
    renderizar_caso_A()
