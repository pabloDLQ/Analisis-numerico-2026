# NumPy permite evaluar el modelo con tiempos escalares o arreglos.
import numpy as np

# Se reutiliza el área total calculada para el dimensionamiento de techos.
from .calculo_fotovoltaico import calcular_area_total_techos
# Se reutiliza la implementación de Simpson 1/3 del módulo de irradiancia.
from .calculo_irradiancia_diaria import simpson_un_tercio


def intensidad_lluvia(t: np.ndarray | float) -> np.ndarray | float:
    """Evalúa la intensidad de lluvia i(t) en milímetros por hora."""
    # Se aplica la función de intensidad definida para la tormenta.
    # La exponencial representa la disminución de la intensidad con el tiempo.
    return 45 * t * np.exp(-t)


def caudal_volumetrico(
    t: np.ndarray | float,
    area_total_m2: float,
) -> np.ndarray | float:
    """Evalúa el caudal de escorrentía en metros cúbicos por hora."""
    # Se transforma mm/h a m/h y se multiplica por el área y la eficiencia.
    return area_total_m2 * 0.90 * (intensidad_lluvia(t) / 1000)


def newton_cotes_abierta_3_puntos(
    funcion,
    a: float,
    b: float,
) -> float:
    """Aproxima una integral con Newton-Cotes abierta de tres puntos."""
    # La regla abierta usa tres nodos interiores separados por este paso.
    paso = (b - a) / 4
    nodos_interiores = a + paso * np.array([1, 2, 3])
    # Se evalúa el caudal en los tres nodos interiores de la regla.
    valores = np.asarray(funcion(nodos_interiores), dtype=float)

    # Se aplica la combinación ponderada de Newton-Cotes abierta.
    return float(
        4
        * paso
        / 3
        * (2 * valores[0] - valores[1] + 2 * valores[2])
    )


def ejecutar_calculo_pluvial() -> None:
    """Calcula e imprime el volumen drenado por ambos métodos."""
    # Se define el intervalo temporal de la tormenta y la malla de Simpson.
    limite_inferior = 0.0
    limite_superior = 3.0
    cantidad_intervalos = 12
    # El caudal depende del área total de captación de los techos.
    area_total_m2 = calcular_area_total_techos()
    # Se fija el área para obtener una función que dependa solo del tiempo.
    funcion_caudal = lambda tiempo: caudal_volumetrico(
        tiempo,
        area_total_m2,
    )

    # Se integra primero con Newton-Cotes abierta usando tres puntos interiores.
    volumen_abierto = newton_cotes_abierta_3_puntos(
        funcion_caudal,
        limite_inferior,
        limite_superior,
    )
    # Se integra también con Simpson compuesto para comparar ambos métodos.
    volumen_simpson = simpson_un_tercio(
        funcion_caudal,
        limite_inferior,
        limite_superior,
        cantidad_intervalos,
    )

    # Se muestran los volúmenes obtenidos por ambos métodos de integración.
    print("Volumen total drenado:")
    print(
        "- Newton-Cotes abierta (3 puntos interiores): "
        f"{volumen_abierto:.4f} m³"
    )
    print(
        "- Simpson 1/3 compuesto (N = 12): "
        f"{volumen_simpson:.4f} m³"
    )


# Se ejecuta el cálculo pluvial solo cuando el archivo se ejecuta directamente.
if __name__ == "__main__":
    ejecutar_calculo_pluvial()
