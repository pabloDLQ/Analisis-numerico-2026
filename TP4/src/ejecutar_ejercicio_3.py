"""Orquesta el renderizado y la reevaluacion de los casos A y B."""

from pathlib import Path

from .renderizado_caso_A import renderizar_caso_A
from .renderizado_caso_B import renderizar_caso_B
from .trayectoria_dron import ejecutar_extraccion


# ---------------------------------------------------------------------
# Rutas de entrada y salida
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RUTA_MAPA = ROOT / "data" / "mapa_satelital_completo.jpg"
RUTA_CSV_MODELO = ROOT / "resultados" / "resultados_ej1" / "trayectoria_dron.csv"

DIR_EJ3 = ROOT / "resultados" / "resultados_ej3"
DIR_CASO_A = DIR_EJ3 / "caso_A"
DIR_CASO_B = DIR_EJ3 / "caso_B"
DIR_GRAFICOS_A = DIR_CASO_A / "graficos_caso_A"
DIR_GRAFICOS_B = DIR_CASO_B / "graficos_caso_B"
VIDEO_A = DIR_CASO_A / "renderizado_caso_A.mp4"
VIDEO_B = DIR_CASO_B / "renderizado_caso_B.mp4"


# ---------------------------------------------------------------------
# Pipeline del ejercicio 3
# ---------------------------------------------------------------------
def main():
    DIR_CASO_A.mkdir(parents=True, exist_ok=True)
    DIR_GRAFICOS_A.mkdir(parents=True, exist_ok=True)
    renderizar_caso_A(
        ruta_csv=RUTA_CSV_MODELO,
        ruta_mapa=RUTA_MAPA,
        ruta_video=VIDEO_A,
    )

    ejecutar_extraccion(
        video_path=VIDEO_A,
        mapa_path=RUTA_MAPA,
        salida=DIR_GRAFICOS_A,
        etiqueta_resultados="Resultados del caso A",
    )

    DIR_CASO_B.mkdir(parents=True, exist_ok=True)
    DIR_GRAFICOS_B.mkdir(parents=True, exist_ok=True)
    renderizar_caso_B(
        ruta_csv=RUTA_CSV_MODELO,
        ruta_mapa=RUTA_MAPA,
        ruta_video=VIDEO_B,
    )

    ejecutar_extraccion(
        video_path=VIDEO_B,
        mapa_path=RUTA_MAPA,
        salida=DIR_GRAFICOS_B,
        etiqueta_resultados="Resultados del caso B",
    )


if __name__ == "__main__":
    main()
