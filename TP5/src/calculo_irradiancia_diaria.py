import numpy as np


def irradiancia_solar(t: np.ndarray | float) -> np.ndarray | float:
    """Evalúa G(t) en W/m² para t expresado en horas."""
    return (
        1000
        * np.sin(np.pi * (t - 6) / 14)
        * np.exp(-0.01 * (t - 13) ** 2)
    )


def irradiancia_nublado(t: np.ndarray | float) -> np.ndarray | float:
    """Evalúa G_nublado(t) en W/m²."""
    return 0.25 * irradiancia_solar(t)


def irradiancia_oscilante(t: np.ndarray | float) -> np.ndarray | float:
    """Evalúa G_oscilante(t) en W/m² con nubosidad oscilante."""
    factor_nube = (
        0.60
        + 0.35
        * np.sin(2.2 * np.pi * (t - 6))
        * np.cos(1.1 * np.pi * t)
    )
    factor_nube_recortado = np.maximum(
        0.20,
        np.minimum(0.95, factor_nube),
    )
    return irradiancia_solar(t) * factor_nube_recortado


def simpson_un_tercio(
    funcion,
    limite_inferior: float,
    limite_superior: float,
    cantidad_intervalos: int,
) -> float:
    """Aproxima una integral usando la Regla de Simpson 1/3 compuesta."""
    if cantidad_intervalos <= 0 or cantidad_intervalos % 2 != 0:
        raise ValueError(
            "La cantidad de intervalos debe ser positiva y par."
        )
    if limite_superior <= limite_inferior:
        raise ValueError(
            "El límite superior debe ser mayor que el límite inferior."
        )

    paso = (limite_superior - limite_inferior) / cantidad_intervalos
    puntos = np.linspace(
        limite_inferior,
        limite_superior,
        cantidad_intervalos + 1,
    )
    valores = np.asarray(funcion(puntos), dtype=float)

    suma_extremos = valores[0] + valores[-1]
    suma_indices_impares = np.sum(valores[1:-1:2])
    suma_indices_pares = np.sum(valores[2:-1:2])

    return float(
        paso
        / 3
        * (
            suma_extremos
            + 4 * suma_indices_impares
            + 2 * suma_indices_pares
        )
    )


def calcular_irradancia_diaria(cantidad_intervalos: int = 14) -> float:
    """Calcula H_dia entre las 6:00 y las 20:00 en kWh/m²·día."""
    integral_wh_por_m2 = simpson_un_tercio(
        irradiancia_solar,
        limite_inferior=6.0,
        limite_superior=20.0,
        cantidad_intervalos=cantidad_intervalos,
    )
    return integral_wh_por_m2 / 1000


def estimar_error_truncamiento(
    cantidad_intervalos: int = 14,
) -> tuple[float, float]:
    """Estima el error absoluto y el orden empírico de Simpson 1/3.

    Se comparan tres aproximaciones sucesivas, con pasos h, h/2 y h/4.
    La estimación de Runge se aplica a la aproximación más fina.
    """
    if cantidad_intervalos <= 0 or cantidad_intervalos % 2 != 0:
        raise ValueError(
            "La cantidad de intervalos debe ser positiva y par."
        )

    aproximacion_h = calcular_irradancia_diaria(cantidad_intervalos)
    aproximacion_h_dos = calcular_irradancia_diaria(2 * cantidad_intervalos)
    aproximacion_h_cuatro = calcular_irradancia_diaria(
        4 * cantidad_intervalos
    )

    diferencia_h = abs(aproximacion_h - aproximacion_h_dos)
    diferencia_h_dos = abs(aproximacion_h_dos - aproximacion_h_cuatro)

    if np.isclose(diferencia_h_dos, 0.0):
        raise ValueError(
            "No se puede estimar el orden: las aproximaciones no difieren."
        )

    orden_empirico = np.log2(diferencia_h / diferencia_h_dos)
    error_absoluto_estimado = diferencia_h_dos / (
        2**orden_empirico - 1
    )

    return float(error_absoluto_estimado), float(orden_empirico)


def ejecutar_calculo_irradiancia() -> None:
    """Ejecuta el análisis completo de irradiancia solar."""
    from .graficos_irradiancia import (
        comparar_escenarios_nubosidad,
        generar_perfil_diario_irradiancia,
    )

    cantidad_intervalos = 14
    irradiancia_diaria = calcular_irradancia_diaria(cantidad_intervalos)
    error_estimado, orden_empirico = estimar_error_truncamiento(
        cantidad_intervalos
    )
    print(
        "Irradiancia diaria acumulada entre las 6:00 y las 20:00: "
        f"{irradiancia_diaria:.6f} kWh/m² por día"
    )
    print(
        "Error absoluto de truncamiento estimado: "
        f"{error_estimado:.12f} kWh/m² por día"
    )
    print(
        "Orden empírico de convergencia: "
        f"{orden_empirico:.6f}"
    )
    comparar_escenarios_nubosidad()
    generar_perfil_diario_irradiancia()


if __name__ == "__main__":
    ejecutar_calculo_irradiancia()
