# NumPy se utiliza para evaluar la expresión analítica del volumen.
import numpy as np

# Se reutilizan las funciones de cálculo de áreas e integración del proyecto.
# También se importa el modelo de caudal y la regla de Newton-Cotes pluvial.
from .calculo_fotovoltaico import calcular_area_total_techos
from .calculo_irradiancia_diaria import simpson_un_tercio
from .calculo_pluvial import (
    caudal_volumetrico,
    newton_cotes_abierta_3_puntos,
)


def volumen_analitico_exacto(area_total_m2: float) -> float:
    """Calcula el volumen exacto drenado entre 0 y 3 horas."""
    # Esta expresión corresponde a la integral exacta del caudal del modelo.
    # Se multiplica la superficie por la eficiencia y la integral de la lluvia.
    return (
        area_total_m2
        * 0.90
        * (45.0 / 1000.0)
        * (1.0 - 4.0 * np.exp(-3.0))
    )


def ejecutar_analisis_error() -> None:
    """Compara los volúmenes numéricos con la solución analítica exacta."""
    # Se fijan los límites temporales y la cantidad de subintervalos de Simpson.
    limite_inferior = 0.0
    limite_superior = 3.0
    cantidad_intervalos = 12
    # El área de captación se obtiene a partir de los contornos de los techos.
    area_total_m2 = calcular_area_total_techos()
    # Se crea una función que deja fijo el área para integrar solo respecto del tiempo.
    # La función anónima permite pasar el caudal a los métodos de integración.
    funcion_caudal = lambda tiempo: caudal_volumetrico(
        tiempo,
        area_total_m2,
    )

    # Se calculan dos aproximaciones numéricas al volumen acumulado.
    volumen_abierto = newton_cotes_abierta_3_puntos(
        funcion_caudal,
        limite_inferior,
        limite_superior,
    )
    # Simpson utiliza una malla uniforme de 12 intervalos.
    volumen_simpson = simpson_un_tercio(
        funcion_caudal,
        limite_inferior,
        limite_superior,
        cantidad_intervalos,
    )
    # Se calcula el valor de referencia mediante la solución analítica.
    volumen_exacto = volumen_analitico_exacto(area_total_m2)

    # Se comparan ambas aproximaciones con el valor exacto.
    error_absoluto_abierto = abs(volumen_exacto - volumen_abierto)
    error_absoluto_simpson = abs(volumen_exacto - volumen_simpson)
    # El error relativo expresa la diferencia como porcentaje del valor exacto.
    error_relativo_abierto = error_absoluto_abierto / volumen_exacto * 100
    error_relativo_simpson = error_absoluto_simpson / volumen_exacto * 100

    # Se presentan los errores absoluto y relativo de cada método.
    print("\nANÁLISIS DE ERROR (Inciso b)")
    print(f"Volumen exacto: {volumen_exacto:.4f} m³")
    print("\nNewton-Cotes abierta (3 puntos interiores):")
    print(f"- Error absoluto: {error_absoluto_abierto:.4f} m³")
    print(f"- Error relativo: {error_relativo_abierto:.4f} %")
    print("\nSimpson 1/3 compuesto (N = 12):")
    print(f"- Error absoluto: {error_absoluto_simpson:.4f} m³")
    print(f"- Error relativo: {error_relativo_simpson:.4f} %")


# Se ejecuta el análisis solo cuando este archivo se lanza directamente.
if __name__ == "__main__":
    ejecutar_analisis_error()
