from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .calcular_areas import (
    centroide_contorno,
    parametrizar_contorno,
)


def dibujar_techos_sobre_mapa(
    techos: dict[str, np.ndarray],
    areas_m2: dict[str, float],
    ruta_imagen: Path,
    ruta_salida: Path,
) -> None:
    """Dibuja los techos y sus superficies sobre la imagen satelital."""
    if not ruta_imagen.is_file():
        raise FileNotFoundError(
            f"No se encontró la imagen satelital en: {ruta_imagen}"
        )

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    imagen = plt.imread(ruta_imagen)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(imagen, origin="upper")

    colores = ("tab:red", "tab:blue", "tab:green", "tab:orange")
    for indice, (clave, contorno) in enumerate(techos.items()):
        puntos = parametrizar_contorno(contorno)[1]
        area_m2 = areas_m2[clave]
        centroide_x, centroide_y = centroide_contorno(contorno)
        color = colores[indice % len(colores)]
        y_min = np.min(puntos[:, 1])
        y_max = np.max(puntos[:, 1])
        margen_rotulo = max(15.0, 0.05 * (y_max - y_min))

        if indice == 0:
            # En la imagen, el eje y crece hacia abajo: debajo del
            # edificio corresponde a un valor de y mayor.
            posicion_rotulo = (centroide_x, y_max + margen_rotulo)
            alineacion_vertical = "top"
        elif indice == 1:
            # Por encima del edificio corresponde a un valor de y menor.
            posicion_rotulo = (centroide_x, y_min - margen_rotulo)
            alineacion_vertical = "bottom"
        else:
            posicion_rotulo = (centroide_x, centroide_y)
            alineacion_vertical = "center"

        ax.fill(
            puntos[:, 0],
            puntos[:, 1],
            color=color,
            alpha=0.35,
            edgecolor=color,
            linewidth=2,
            label=f"Edificio {indice + 1}",
        )
        ax.text(
            *posicion_rotulo,
            f"Edificio {indice + 1}\n{area_m2:.2f} m²",
            ha="center",
            va=alineacion_vertical,
            color="white",
            fontsize=10,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "black",
                "alpha": 0.7,
            },
        )

    ax.set_title("Superficie de los techos sobre imagen satelital")
    ax.set_xlabel("Coordenada x (píxeles)")
    ax.set_ylabel("Coordenada y (píxeles)")
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Imagen guardada en: {ruta_salida}")
