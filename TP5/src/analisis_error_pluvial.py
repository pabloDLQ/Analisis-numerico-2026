import numpy as np

from .calculo_fotovoltaico import calcular_area_total_techos
from .calculo_irradiancia_diaria import simpson_un_tercio
from .calculo_pluvial import (
    caudal_volumetrico,
    newton_cotes_abierta_3_puntos,
)


def volumen_analitico_exacto(area_total_m2: float) -> float:
    """Calcula el volumen exacto drenado entre 0 y 3 horas."""
    return (
        area_total_m2
        * 0.90
        * (45.0 / 1000.0)
        * (1.0 - 4.0 * np.exp(-3.0))
    )


def ejecutar_analisis_error() -> None:
    """Compara los volúmenes numéricos con la solución analítica exacta."""
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
    volumen_exacto = volumen_analitico_exacto(area_total_m2)

    error_absoluto_abierto = abs(volumen_exacto - volumen_abierto)
    error_absoluto_simpson = abs(volumen_exacto - volumen_simpson)
    error_relativo_abierto = error_absoluto_abierto / volumen_exacto * 100
    error_relativo_simpson = error_absoluto_simpson / volumen_exacto * 100

    print("\nANÁLISIS DE ERROR (Inciso b)")
    print(f"Volumen exacto: {volumen_exacto:.4f} m³")
    print("\nNewton-Cotes abierta (3 puntos interiores):")
    print(f"- Error absoluto: {error_absoluto_abierto:.4f} m³")
    print(f"- Error relativo: {error_relativo_abierto:.4f} %")
    print("\nSimpson 1/3 compuesto (N = 12):")
    print(f"- Error absoluto: {error_absoluto_simpson:.4f} m³")
    print(f"- Error relativo: {error_relativo_simpson:.4f} %")


if __name__ == "__main__":
    ejecutar_analisis_error()
