"""Renderiza un vuelo libre de ruido con velocidad limitada por dinamica y curvatura."""

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
# Configuracion del vuelo real y Parametros Fisicos
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RUTA_CSV = ROOT / "resultados" / "resultados_ej1" / "trayectoria_dron.csv"
RUTA_MAPA = ROOT / "data" / "mapa_satelital_completo.jpg"
DIRECTORIO_SALIDA = ROOT / "resultados" / "resultados_ej4"
DIRECTORIO_GRAFICOS = DIRECTORIO_SALIDA / "graficos"
RUTA_VIDEO = DIRECTORIO_SALIDA / "vuelo_dinamica_real.mp4"

MUESTRAS_DENSAS = 20000

# Calculo previo de escala (Desacople visual-cinematico basado en Caso B)
V_MAX_OBSERVADA_PX = 350.0       # Velocidad pico aislada en el Ejercicio 3 Caso B (px/s)
V_MAX_REAL_MS_REF = 7.0          # Velocidad de crucero real para mapeo UAV (m/s)
CALCULO_ESCALA = V_MAX_OBSERVADA_PX / V_MAX_REAL_MS_REF  # Da como resultado 50.0 px/m

# Parametros cinemáticos del mundo real (basados en ArduPilot/PX4)
ESCALA_PX_M = CALCULO_ESCALA     # Factor de conversión experimental (px/m)
V_MAX_MS = 7.0                   # Velocidad de crucero máxima (m/s)
A_MAX_MS2 = 3.0                  # Límite de aceleración/frenado tangencial (m/s^2)
ANGULO_ALABEO_MAX = 31.45        # Inclinación máxima permitida del chasis (grados)
GRAVEDAD = 9.81                  # Aceleración gravitatoria (m/s^2)

# Conversión analítica al dominio de la imagen (pixeles/frame)
V_MAX = V_MAX_MS * ESCALA_PX_M
A_MAX = A_MAX_MS2 * ESCALA_PX_M
# Ac = g * tan(theta). Con theta = 31.45°, Ac resulta exactamente en ~6 m/s^2
AC_MAX_MS2 = GRAVEDAD * np.tan(np.radians(ANGULO_ALABEO_MAX))
AC_MAX = AC_MAX_MS2 * ESCALA_PX_M


# ---------------------------------------------------------------------
# Carga de la trayectoria base
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
    
    return t, t_nodos, x_nodos, y_nodos, zoom


# ---------------------------------------------------------------------
# Barrido cinematico real sobre el spline suavizado
# ---------------------------------------------------------------------
def remuestrear_dinamica_real(t, t_nodos, x_nodos, y_nodos, zoom):
    t = np.asarray(t, dtype=float)
    zoom = np.asarray(zoom, dtype=float)

    # 1. Trazamos el recorrido sin ruido usando el modelo local
    t_denso = np.linspace(t[0], t[-1], MUESTRAS_DENSAS)
    x_denso = spline_cubico_natural(t_denso, t_nodos, x_nodos)
    y_denso = spline_cubico_natural(t_denso, t_nodos, y_nodos)

    # 2. La curvatura limita la velocidad máxima permitida en cada punto
    dx = np.gradient(x_denso, t_denso)
    dy = np.gradient(y_denso, t_denso)
    ddx = np.gradient(dx, t_denso)
    ddy = np.gradient(dy, t_denso)
    kappa = np.abs(dx * ddy - dy * ddx) / np.maximum((dx**2 + dy**2) ** 1.5, 1e-8)
    
    v_curva = np.sqrt(AC_MAX / np.maximum(kappa, 1e-6))
    v_limite = np.minimum(V_MAX, v_curva)

    ds = np.hypot(np.diff(x_denso), np.diff(y_denso))
    v_perfil = np.zeros(len(x_denso), dtype=float)

    # 3. Primera pasada (Forward): acelera desde el reposo hasta el limite local
    v_perfil[0] = 0.0
    for i in range(1, len(x_denso)):
        v_perfil[i] = min(
            v_limite[i],
            np.sqrt(v_perfil[i - 1] ** 2 + 2.0 * A_MAX * ds[i - 1]),
        )

    # 4. Segunda pasada (Backward): asegura el frenado hacia las curvas y el final
    v_perfil[-1] = 0.0
    for i in range(len(x_denso) - 2, -1, -1):
        v_perfil[i] = min(
            v_perfil[i],
            np.sqrt(v_perfil[i + 1] ** 2 + 2.0 * A_MAX * ds[i]),
        )

    # 5. Integración temporal física usando la velocidad media del segmento
    v_media = (v_perfil[:-1] + v_perfil[1:]) / 2.0
    dt_real = ds / np.maximum(v_media, 1.0)
    t_fisico = np.concatenate(([0.0], np.cumsum(dt_real)))
    
    # 6. Remuestreo uniforme (30 FPS) para generar el video sintético
    zoom_denso = np.interp(t_denso, t, zoom)
    tiempo_uniforme = np.arange(0.0, t_fisico[-1] + 0.5 / FPS_SALIDA, 1.0 / FPS_SALIDA)

    x_uniforme = np.interp(tiempo_uniforme, t_fisico, x_denso)
    y_uniforme = np.interp(tiempo_uniforme, t_fisico, y_denso)
    zoom_uniforme = np.interp(tiempo_uniforme, t_fisico, zoom_denso)
    
    return x_uniforme, y_uniforme, zoom_uniforme, tiempo_uniforme


# ---------------------------------------------------------------------
# Renderizado y Extracción
# ---------------------------------------------------------------------
def renderizar_vuelo_real():
    t, t_nodos, x_nodos, y_nodos, zoom = cargar_trayectoria(RUTA_CSV)

    x, y, zoom, tiempo = remuestrear_dinamica_real(t, t_nodos, x_nodos, y_nodos, zoom)

    mapa = cv2.imread(str(RUTA_MAPA), cv2.IMREAD_COLOR)
    if mapa is None:
        raise FileNotFoundError(f"No se pudo abrir el mapa: {RUTA_MAPA}")
    
    frame0, _, base_width, base_height = obtener_referencia_frame0(RUTA_VIDEO_ORIGINAL, mapa)

    DIRECTORIO_SALIDA.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(RUTA_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS_SALIDA,
        (ANCHO_PANEL, ALTO_PANEL),
    )

    try:
        zoom = zoom / zoom[0]
        for indice in range(len(tiempo)):
            frame = construir_frame(
                mapa, frame0, x, y, zoom, indice, base_width, base_height
            )
            writer.write(frame)
    finally:
        writer.release()

    print(f"Video solicitado generado en: {RUTA_VIDEO.parent.resolve()}")

    ejecutar_extraccion(
        video_path=RUTA_VIDEO,
        mapa_path=RUTA_MAPA,
        salida=DIRECTORIO_SALIDA,
        etiqueta_resultados="Resultados vuelo real",
        mostrar_resultado=False,
        salida_graficos=DIRECTORIO_GRAFICOS,
    )
    print(f"Graficas generadas en: {DIRECTORIO_GRAFICOS.resolve()}")


if __name__ == "__main__":
    renderizar_vuelo_real()