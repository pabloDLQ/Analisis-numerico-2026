# NumPy permite manipular los puntos y realizar las sumas vectorizadas.
import numpy as np


# Conversión de píxeles de la imagen a metros.
ESCALA_METROS_POR_PIXEL = 0.5073


def parametrizar_contorno(contorno: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cierra un contorno discreto y le asigna un parámetro normalizado."""
    # Se convierte la entrada a punto flotante para uniformar los cálculos.
    puntos = np.asarray(contorno, dtype=float)

    # Se verifica que cada punto tenga exactamente coordenadas x e y.
    if puntos.ndim != 2 or puntos.shape[1] != 2:
        raise ValueError("El contorno debe tener forma (n, 2).")
    # Un polígono necesita al menos tres puntos para definir una superficie.
    if puntos.shape[0] < 3:
        raise ValueError("Se necesitan al menos tres puntos.")

    # La curva poligonal se recorre con t_i = i/(n-1), incluyendo
    # el último punto coincidente con el primero para cerrar el contorno.
    # Se agrega el primer punto al final si el contorno aún no está cerrado.
    if not np.array_equal(puntos[0], puntos[-1]):
        puntos = np.vstack((puntos, puntos[0]))
    # Se asigna a cada punto un parámetro equiespaciado entre 0 y 1.
    parametro = np.linspace(0.0, 1.0, puntos.shape[0])
    # Se devuelven el parámetro y las coordenadas del contorno cerrado.
    return parametro, puntos


def area_gauss_pixeles_cuadrados(contorno: np.ndarray) -> float:
    """Calcula el área de un contorno mediante Green/Shoelace."""
    # Se obtiene el polígono cerrado para aplicar la fórmula de Gauss.
    _, puntos = parametrizar_contorno(contorno)
    # Se separan las coordenadas para aplicar la fórmula de área.
    x = puntos[:, 0]
    y = puntos[:, 1]

    # Para cada segmento de la curva poligonal, la integral de Green
    # 1/2 * integral (x dy - y dx) produce la fórmula del área de Gauss.
    # Se calcula el área orientada sumando el aporte de cada lado.
    area_orientada = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    # El valor absoluto elimina la dependencia de la orientación del contorno.
    return float(abs(area_orientada))


def validar_area_rectangulo() -> None:
    """Comprueba la fórmula con un rectángulo de 100 px por 50 px."""
    # Se construye un rectángulo cuyo resultado esperado es conocido.
    rectangulo = np.array(
        [[0, 0], [100, 0], [100, 50], [0, 50]],
        dtype=float,
    )
    # El área teórica se usa como referencia para validar el cálculo.
    area_esperada = 100 * 50
    # Se calcula el área del rectángulo usando la función que se está probando.
    area_obtenida = area_gauss_pixeles_cuadrados(rectangulo)

    # Una diferencia indica que la implementación del área no es correcta.
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
    # Se reutilizan los puntos cerrados para aplicar la fórmula del centroide.
    _, puntos = parametrizar_contorno(contorno)
    # Se extraen las coordenadas y los términos cruzados de cada lado.
    x = puntos[:, 0]
    y = puntos[:, 1]
    termino = x[:-1] * y[1:] - x[1:] * y[:-1]
    # El área orientada se necesita en el denominador de la fórmula del centroide.
    area_orientada = 0.5 * np.sum(termino)

    # Para un polígono degenerado se usa el promedio de sus vértices.
    if np.isclose(area_orientada, 0.0):
        centroide = np.mean(puntos[:-1], axis=0)
        return float(centroide[0]), float(centroide[1])

    # Se calculan por separado las coordenadas x e y del centroide.
    centroide_x = np.sum((x[:-1] + x[1:]) * termino) / (6.0 * area_orientada)
    centroide_y = np.sum((y[:-1] + y[1:]) * termino) / (6.0 * area_orientada)
    return float(centroide_x), float(centroide_y)


def analizar_techos(techos: dict[str, np.ndarray]) -> dict[str, float]:
    """Calcula e informa las áreas de los techos en metros cuadrados."""
    # La escala lineal se eleva al cuadrado porque se convierten áreas.
    escala_area = ESCALA_METROS_POR_PIXEL**2
    areas_totales_m2 = 0.0
    # El diccionario conservará la correspondencia entre techo y área calculada.
    areas_m2 = {}

    print("Áreas de los techos:")
    # Se procesa cada techo, convirtiendo su área de píxeles cuadrados a m².
    for clave, contorno in techos.items():
        # Primero se obtiene el área geométrica en píxeles cuadrados.
        area_px2 = area_gauss_pixeles_cuadrados(contorno)
        # Luego se aplica la conversión de escala a metros cuadrados.
        area_m2 = area_px2 * escala_area
        areas_m2[clave] = area_m2
        areas_totales_m2 += area_m2
        print(f"- {clave}: {area_m2:.2f} m² ({area_px2:.2f} px²)")

    print(f"Área total: {areas_totales_m2:.2f} m²")
    # Se devuelven las áreas individuales para usarlas en otros módulos.
    return areas_m2
