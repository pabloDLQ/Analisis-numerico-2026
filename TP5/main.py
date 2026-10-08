from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ESCALA_METROS_POR_PIXEL = 0.5073


def parametrizar_contorno(contorno: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cierra un contorno discreto y le asigna un parámetro normalizado."""
    puntos = np.asarray(contorno, dtype=float)

    if puntos.ndim != 2 or puntos.shape[1] != 2:
        raise ValueError("El contorno debe tener forma (n, 2).")
    if puntos.shape[0] < 3:
        raise ValueError("Se necesitan al menos tres puntos.")

    # La curva poligonal se recorre con t_i = i/(n-1), incluyendo
    # el último punto coincidente con el primero para cerrar el contorno.
    if not np.array_equal(puntos[0], puntos[-1]):
        puntos = np.vstack((puntos, puntos[0]))
    parametro = np.linspace(0.0, 1.0, puntos.shape[0])
    return parametro, puntos


def area_gauss_pixeles_cuadrados(contorno: np.ndarray) -> float:
    """Calcula el área de un contorno mediante Green/Shoelace."""
    _, puntos = parametrizar_contorno(contorno)
    x = puntos[:, 0]
    y = puntos[:, 1]

    # Para cada segmento de la curva poligonal, la integral de Green
    # 1/2 * integral (x dy - y dx) produce la fórmula del área de Gauss.
    area_orientada = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    return float(abs(area_orientada))


def validar_area_rectangulo() -> None:
    """Comprueba la fórmula con un rectángulo de 100 px por 50 px."""
    rectangulo = np.array(
        [[0, 0], [100, 0], [100, 50], [0, 50]],
        dtype=float,
    )
    area_esperada = 100 * 50
    area_obtenida = area_gauss_pixeles_cuadrados(rectangulo)

    if area_obtenida != area_esperada:
        raise AssertionError(
            f"Validación incorrecta: se esperaban {area_esperada} px², "
            f"pero se obtuvieron {area_obtenida} px²."
        )

    print(
        "Validación rectángulo 100 x 50 px: "
        f"{area_obtenida:.0f} px² (esperada: {area_esperada} px²)"
    )


def centroide_contorno(contorno: np.ndarray) -> tuple[float, float]:
    """Calcula el centroide de un polígono mediante la fórmula de Gauss."""
    _, puntos = parametrizar_contorno(contorno)
    x = puntos[:, 0]
    y = puntos[:, 1]
    termino = x[:-1] * y[1:] - x[1:] * y[:-1]
    area_orientada = 0.5 * np.sum(termino)

    if np.isclose(area_orientada, 0.0):
        return tuple(np.mean(puntos[:-1], axis=0))

    centroide_x = np.sum((x[:-1] + x[1:]) * termino) / (6.0 * area_orientada)
    centroide_y = np.sum((y[:-1] + y[1:]) * termino) / (6.0 * area_orientada)
    return float(centroide_x), float(centroide_y)


def dibujar_techos_sobre_mapa(
) -> None:
    """Dibuja los techos y sus superficies sobre la imagen satelital."""
    ruta_base = Path(__file__).parent
    ruta_datos = ruta_base / "data" / "techos_campus_utn.npz"
    ruta_imagen = ruta_base / "data" / "mapa_satelital_completo.jpg"
    carpeta_resultados = ruta_base / "resultados" / "ejercicio1"
    ruta_salida = carpeta_resultados / "techos_sobre_mapa.png"

    if not ruta_imagen.is_file():
        raise FileNotFoundError(
            f"No se encontró la imagen satelital en: {ruta_imagen}"
        )

    carpeta_resultados.mkdir(parents=True, exist_ok=True)
    imagen = plt.imread(ruta_imagen)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(imagen, origin="upper")

    colores = ("tab:red", "tab:blue", "tab:green", "tab:orange")
    with np.load(ruta_datos, allow_pickle=False) as datos:
        for indice, clave in enumerate(datos.files):
            contorno = datos[clave]
            puntos = parametrizar_contorno(contorno)[1]
            area_m2 = area_gauss_pixeles_cuadrados(contorno)
            area_m2 *= ESCALA_METROS_POR_PIXEL**2
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


def analizar_techos() -> None:
    """Calcula el área de cada techo y el área total en metros cuadrados."""
    ruta_archivo = Path(__file__).parent / "data" / "techos_campus_utn.npz"
    escala_area = ESCALA_METROS_POR_PIXEL**2

    with np.load(ruta_archivo, allow_pickle=False) as datos:
        areas_totales_m2 = 0.0

        print(f"Archivo: {ruta_archivo}")
        print("Áreas de los techos:")
        for clave in datos.files:
            area_px2 = area_gauss_pixeles_cuadrados(datos[clave])
            area_m2 = area_px2 * escala_area
            areas_totales_m2 += area_m2
            print(f"- {clave}: {area_m2:.2f} m² ({area_px2:.2f} px²)")

        print(f"Área total: {areas_totales_m2:.2f} m²")


if __name__ == "__main__":
    validar_area_rectangulo()
    analizar_techos()
    dibujar_techos_sobre_mapa()