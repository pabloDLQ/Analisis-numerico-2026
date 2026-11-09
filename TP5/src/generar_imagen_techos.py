# Path se usa para comprobar y construir las rutas de imágenes.
from pathlib import Path

# Matplotlib genera la imagen anotada y NumPy procesa coordenadas.
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
    # Se verifica que exista la imagen base antes de crear el gráfico.
    if not ruta_imagen.is_file():
        raise FileNotFoundError(
            f"No se encontró la imagen satelital en: {ruta_imagen}"
        )

    # Se crea la carpeta de salida si todavía no existe.
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    # Se carga la imagen y se prepara un eje con el mismo sistema de coordenadas.
    imagen = plt.imread(ruta_imagen)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(imagen, origin="upper")

    # Cada edificio se dibuja con un color distinto para facilitar su lectura.
    colores = ("tab:red", "tab:blue", "tab:green", "tab:orange")
    for indice, (clave, contorno) in enumerate(techos.items()):
        # Se obtienen el contorno cerrado, su área y su centroide.
        puntos = parametrizar_contorno(contorno)[1]
        area_m2 = areas_m2[clave]
        centroide_x, centroide_y = centroide_contorno(contorno)
        color = colores[indice % len(colores)]
        y_min = np.min(puntos[:, 1])
        y_max = np.max(puntos[:, 1])
        margen_rotulo = max(15.0, 0.05 * (y_max - y_min))

        # Se elige la posición de la etiqueta para evitar cubrir cada edificio.
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

        # Se rellena el polígono del techo y se dibuja su borde.
        ax.fill(
            puntos[:, 0],
            puntos[:, 1],
            color=color,
            alpha=0.35,
            edgecolor=color,
            linewidth=2,
            label=f"Edificio {indice + 1}",
        )
        # Se agrega una etiqueta con el número del edificio y su superficie.
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

    # Se configuran títulos, ejes, leyenda y proporción del gráfico.
    ax.set_title("Superficie de los techos sobre imagen satelital")
    ax.set_xlabel("Coordenada x (píxeles)")
    ax.set_ylabel("Coordenada y (píxeles)")
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()
    # Se guarda la figura en alta resolución y se libera la memoria.
    fig.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Imagen guardada en: {ruta_salida}")
