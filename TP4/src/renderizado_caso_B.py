"""Renderizado del caso B con zoom constante y rapidez constante."""

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
from .renderizado_caso_A import (
    ALTO_PANEL,
    ANCHO_PANEL,
    FPS_SALIDA,
    RUTA_MAPA,
    RUTA_VIDEO_ORIGINAL,
    construir_frame,
    obtener_referencia_frame0,
    validar_velocidad_constante,
)


ROOT = Path(__file__).resolve().parents[1]
RUTA_CSV = RUTA_CSV_MODELO
DIRECTORIO_SALIDA = ROOT / "resultados" / "resultados_ej3"
RUTA_VIDEO = DIRECTORIO_SALIDA / "renderizado_caso_B.mp4"
ZOOM_CONSTANTE = 1.0


def cargar_trayectoria(ruta_csv: Path):
    """Carga X e Y y las evalua mediante el spline cubico natural."""
    t, _, _, _, t_nodos, x_nodos, y_nodos, _ = cargar_datos_y_nodos(
        ruta_csv, SALTO_NODOS
    )
    if len(t) < 2 or not np.all(np.isfinite(t)) or not np.all(np.diff(t) > 0):
        raise ValueError("Los tiempos deben ser finitos y estrictamente crecientes")

    x_spline = spline_cubico_natural(t, t_nodos, x_nodos)
    y_spline = spline_cubico_natural(t, t_nodos, y_nodos)
    if not np.all(np.isfinite(x_spline)) or not np.all(np.isfinite(y_spline)):
        raise ValueError("El spline produjo coordenadas no finitas")
    return t, t_nodos, x_nodos, y_nodos, x_spline, y_spline


def remuestrear_posicion_rapidez_constante(
    t, t_nodos, x_nodos, y_nodos, cantidad_muestras_densas=10000
):
    """Reparametriza la trayectoria para que su rapidez espacial sea constante."""
    t = np.asarray(t, dtype=float)
    if len(t) < 2 or not np.all(np.diff(t) > 0):
        raise ValueError("Se necesitan al menos dos tiempos estrictamente crecientes")

    t_denso = np.linspace(
        t[0], t[-1], max(cantidad_muestras_densas, 100 * len(t))
    )
    x_denso = spline_cubico_natural(t_denso, t_nodos, x_nodos)
    y_denso = spline_cubico_natural(t_denso, t_nodos, y_nodos)
    longitud = np.concatenate(([0.0], np.cumsum(np.hypot(
        np.diff(x_denso), np.diff(y_denso)
    ))))
    if longitud[-1] <= 0.0:
        raise ValueError("La trayectoria no tiene longitud suficiente")

    velocidad_constante = longitud[-1] / (t[-1] - t[0])
    distancia = velocidad_constante * (t - t[0])
    x_uniforme = np.interp(distancia, longitud, x_denso)
    y_uniforme = np.interp(distancia, longitud, y_denso)
    return x_uniforme, y_uniforme, velocidad_constante


def renderizar_caso_B(ruta_csv=RUTA_CSV, ruta_mapa=RUTA_MAPA, ruta_video=RUTA_VIDEO):
    """Genera el video del caso B: Z(t)=1 y v(t)=v0."""
    if ZOOM_CONSTANTE != 1.0:
        raise ValueError("El caso B requiere ZOOM_CONSTANTE = 1.0")

    mapa = cv2.imread(str(ruta_mapa), cv2.IMREAD_COLOR)
    if mapa is None:
        raise FileNotFoundError(f"No se pudo abrir el mapa: {ruta_mapa}")
    frame0, _, base_width, base_height = obtener_referencia_frame0(
        RUTA_VIDEO_ORIGINAL, mapa
    )
    t, t_nodos, x_nodos, y_nodos, _, _ = cargar_trayectoria(Path(ruta_csv))
    x, y, velocidad_constante = remuestrear_posicion_rapidez_constante(
        t, t_nodos, x_nodos, y_nodos
    )
    validar_velocidad_constante(t, velocidad_constante)
    zoom = np.full(len(t), ZOOM_CONSTANTE, dtype=float)

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
    print(f"Zoom constante: Z = {ZOOM_CONSTANTE:.1f}")
    print(f"Rapidez espacial constante: {velocidad_constante:.6f} pixeles/s")
    return ruta_video


if __name__ == "__main__":
    renderizar_caso_B()