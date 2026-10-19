import numpy as np

from .calculo_fotovoltaico import calcular_area_total_techos
from .calculo_irradiancia_diaria import simpson_un_tercio


def intensidad_lluvia(t: np.ndarray | float) -> np.ndarray | float:
    """Evalúa la intensidad de lluvia i(t) en milímetros por hora."""
    return 45 * t * np.exp(-t)


def caudal_volumetrico(
    t: np.ndarray | float,
    area_total_m2: float,
) -> np.ndarray | float:
    """Evalúa el caudal de escorrentía en metros cúbicos por hora."""
    return area_total_m2 * 0.90 * (intensidad_lluvia(t) / 1000)


def newton_cotes_abierta_3_puntos(
    funcion,
    a: float,
    b: float,
) -> float:
    """Aproxima una integral con Newton-Cotes abierta de tres puntos."""
    paso = (b - a) / 4
    nodos_interiores = a + paso * np.array([1, 2, 3])
    valores = np.asarray(funcion(nodos_interiores), dtype=float)

    return float(
        4
        * paso
        / 3
        * (2 * valores[0] - valores[1] + 2 * valores[2])
    )


def ejecutar_calculo_pluvial() -> None:
    """Calcula e imprime el volumen drenado por ambos métodos."""
    limite_inferior = 0.0
    limite_superior = 3.0
    cantidad_intervalos = 12
    area_total_m2 = calcular_area_total_techos()
    funcion_caudal = lambda tiempo: caudal_volumetrico(
        tiempo,
        area_total_m2,
    )

    volumen_abierto = newton_cotes_abierta_3_puntos(
        funcion_caudal,
        limite_inferior,
        limite_superior,
    )
    volumen_simpson = simpson_un_tercio(
        funcion_caudal,
        limite_inferior,
        limite_superior,
        cantidad_intervalos,
    )

    print("Volumen total drenado:")
    print(
        "- Newton-Cotes abierta (3 puntos interiores): "
        f"{volumen_abierto:.4f} m³"
    )
    print(
        "- Simpson 1/3 compuesto (N = 12): "
        f"{volumen_simpson:.4f} m³"
    )


if __name__ == "__main__":
    ejecutar_calculo_pluvial()
