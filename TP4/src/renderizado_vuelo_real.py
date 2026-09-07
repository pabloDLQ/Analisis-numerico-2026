"""Renderiza un vuelo con velocidad limitada por dinamica y curvatura."""

from pathlib import Path

import cv2
import numpy as np

from .modelado_trayectoria import SALTO_NODOS, cargar_datos_y_nodos, spline_cubico_natural
from .renderizado_caso_A import (
    ALTO_PANEL,
    ANCHO_PANEL,
    FPS_SALIDA,
    RUTA_VIDEO_ORIGINAL,
    construir_frame,
    obtener_referencia_frame0,
)
from .trayectoria_dron import ejecutar_extraccion


# ---------------------------------------------------------------------
# Configuracion del vuelo
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RUTA_CSV = ROOT / "resultados" / "resultados_ej1" / "trayectoria_dron.csv"
RUTA_MAPA = ROOT / "data" / "mapa_satelital_completo.jpg"
DIRECTORIO_SALIDA = ROOT / "resultados" / "resultados_ej4"
RUTA_VIDEO = DIRECTORIO_SALIDA / "vuelo_dinamica_real.mp4"

MUESTRAS_DENSAS = 20000
V_MAX = 450.0
A_MAX = 150.0
AC_MAX = 300.0


# ---------------------------------------------------------------------
# Carga de la trayectoria
# ---------------------------------------------------------------------
def cargar_trayectoria(ruta_csv):
    datos = cargar_datos_y_nodos(ruta_csv, SALTO_NODOS)
    t, _, _, _, t_nodos, x_nodos, y_nodos, _ = datos
    zoom = np.genfromtxt(
        ruta_csv,
        delimiter=",",
        skip_header=1,
        usecols=(4,),
        dtype=float,
    )
    zoom = np.atleast_1d(zoom)
    if len(t) < 2 or len(zoom) != len(t):
        raise ValueError("La trayectoria y el zoom deben tener la misma cantidad de muestras")
    if not np.all(np.isfinite(zoom)) or np.any(zoom <= 0):
        raise ValueError("El zoom debe contener valores positivos y finitos")

    return t, t_nodos, x_nodos, y_nodos, zoom


# ---------------------------------------------------------------------
# Barrido cinemático
# ---------------------------------------------------------------------
def remuestrear_dinamica_real(t, t_nodos, x_nodos, y_nodos, zoom):
    t = np.asarray(t, dtype=float)
    zoom = np.asarray(zoom, dtype=float)
    if len(t) < 2 or len(zoom) != len(t) or not np.all(np.diff(t) > 0):
        raise ValueError("Los tiempos deben ser crecientes y compatibles con el zoom")

    t_denso = np.linspace(t[0], t[-1], MUESTRAS_DENSAS)
    x_denso = spline_cubico_natural(t_denso, t_nodos, x_nodos)
    y_denso = spline_cubico_natural(t_denso, t_nodos, y_nodos)

    # La curvatura limita la velocidad máxima permitida en cada nodo
    dx = np.gradient(x_denso, t_denso)
    dy = np.gradient(y_denso, t_denso)
    ddx = np.gradient(dx, t_denso)
    ddy = np.gradient(dy, t_denso)
    kappa = np.abs(dx * ddy - dy * ddx) / np.maximum(
        (dx**2 + dy**2) ** 1.5,
        1e-8,
    )
    v_curva = np.sqrt(AC_MAX / np.maximum(kappa, 1e-6))
    v_limite = np.minimum(V_MAX, v_curva)

    ds = np.hypot(np.diff(x_denso), np.diff(y_denso))
    v_perfil = np.zeros(len(x_denso), dtype=float)

    # Primera pasada (Forward): acelera desde el reposo hasta el limite local
    v_perfil[0] = 0.0
    for i in range(1, len(x_denso)):
        v_perfil[i] = min(
            v_limite[i],
            np.sqrt(v_perfil[i - 1] ** 2 + 2.0 * A_MAX * ds[i - 1]),
        )

    # Segunda pasada (Backward): asegura el frenado hacia las curvas y el final
    v_perfil[-1] = 0.0
    for i in range(len(x_denso) - 2, -1, -1):
        v_perfil[i] = min(
            v_perfil[i],
            np.sqrt(v_perfil[i + 1] ** 2 + 2.0 * A_MAX * ds[i]),
        )

    # Integración temporal física usando la velocidad media del segmento
    v_media = (v_perfil[:-1] + v_perfil[1:]) / 2.0
    dt_real = ds / np.maximum(v_media, 1.0)
    t_fisico = np.concatenate(([0.0], np.cumsum(dt_real)))
    
    zoom_denso = np.interp(t_denso, t, zoom)
    tiempo_uniforme = np.arange(0.0, t_fisico[-1] + 0.5 / FPS_SALIDA, 1.0 / FPS_SALIDA)

    x_uniforme = np.interp(tiempo_uniforme, t_fisico, x_denso)
    y_uniforme = np.interp(tiempo_uniforme, t_fisico, y_denso)
    zoom_uniforme = np.interp(tiempo_uniforme, t_fisico, zoom_denso)
    return x_uniforme, y_uniforme, zoom_uniforme, tiempo_uniforme


# ---------------------------------------------------------------------
# Renderizado y reevaluacion
# ---------------------------------------------------------------------
def renderizar_vuelo_real():
    t, t_nodos, x_nodos, y_nodos, zoom = cargar_trayectoria(RUTA_CSV)
    x, y, zoom, tiempo = remuestrear_dinamica_real(
        t,
        t_nodos,
        x_nodos,
        y_nodos,
        zoom,
    )

    mapa = cv2.imread(str(RUTA_MAPA), cv2.IMREAD_COLOR)
    if mapa is None:
        raise FileNotFoundError(f"No se pudo abrir el mapa: {RUTA_MAPA}")
    frame0, _, base_width, base_height = obtener_referencia_frame0(
        RUTA_VIDEO_ORIGINAL,
        mapa,
    )

    DIRECTORIO_SALIDA.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(RUTA_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS_SALIDA,
        (ANCHO_PANEL, ALTO_PANEL),
    )
    if not writer.isOpened():
        raise RuntimeError(f"No se pudo abrir el escritor de video: {RUTA_VIDEO}")

    try:
        zoom = zoom / zoom[0]
        for indice in range(len(tiempo)):
            frame = construir_frame(
                mapa,
                frame0,
                x,
                y,
                zoom,
                indice,
                base_width,
                base_height,
            )
            if frame.shape != (ALTO_PANEL, ANCHO_PANEL, 3) or frame.dtype != np.uint8:
                raise RuntimeError("El frame generado tiene dimensiones o tipo invalidos")
            writer.write(frame)
    finally:
        writer.release()

    print(f"Video de dinamica real generado: {RUTA_VIDEO.resolve()}")
    print("Iniciando extraccion de resultados del vuelo dinamico...")
    ejecutar_extraccion(
        video_path=RUTA_VIDEO,
        mapa_path=RUTA_MAPA,
        salida=DIRECTORIO_SALIDA,
        etiqueta_resultados="Resultados del vuelo dinamico",
    )
    return RUTA_VIDEO


if __name__ == "__main__":
    renderizar_vuelo_real()