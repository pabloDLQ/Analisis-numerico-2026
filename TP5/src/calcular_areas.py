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
        centroide = np.mean(puntos[:-1], axis=0)
        return float(centroide[0]), float(centroide[1])

    centroide_x = np.sum((x[:-1] + x[1:]) * termino) / (6.0 * area_orientada)
    centroide_y = np.sum((y[:-1] + y[1:]) * termino) / (6.0 * area_orientada)
    return float(centroide_x), float(centroide_y)


def analizar_techos(techos: dict[str, np.ndarray]) -> dict[str, float]:
    """Calcula e informa las áreas de los techos en metros cuadrados."""
    escala_area = ESCALA_METROS_POR_PIXEL**2
    areas_totales_m2 = 0.0
    areas_m2 = {}

    print("Áreas de los techos:")
    for clave, contorno in techos.items():
        area_px2 = area_gauss_pixeles_cuadrados(contorno)
        area_m2 = area_px2 * escala_area
        areas_m2[clave] = area_m2
        areas_totales_m2 += area_m2
        print(f"- {clave}: {area_m2:.2f} m² ({area_px2:.2f} px²)")

    print(f"Área total: {areas_totales_m2:.2f} m²")
    return areas_m2
