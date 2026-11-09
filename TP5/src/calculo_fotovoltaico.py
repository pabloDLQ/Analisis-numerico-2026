# Path permite ubicar el archivo de datos sin depender de la carpeta actual.
from pathlib import Path

# NumPy permite evaluar la potencia para tiempos escalares o vectoriales.
import numpy as np

# Se importan la escala y el cálculo geométrico de áreas de los techos.
from .calcular_areas import (
    ESCALA_METROS_POR_PIXEL,
    area_gauss_pixeles_cuadrados,
)
# Se importan los modelos de irradiancia y el integrador de Simpson.
from .calculo_irradiancia_diaria import (
    irradiancia_nublado,
    irradiancia_oscilante,
    irradiancia_solar,
    simpson_un_tercio,
)
# Se importa la función que lee los contornos desde el archivo NPZ.
from .leer_datos import leer_techos


# Parámetros físicos y de diseño utilizados para dimensionar la instalación.
PORCENTAJE_SUPERFICIE_UTILIZABLE = 0.35
POTENCIA_PANEL_WP = 550.0
LARGO_PANEL_M = 2.28
ANCHO_PANEL_M = 1.13
FACTOR_RENDIMIENTO_GLOBAL = 0.80
DEMANDA_CAMPUS_KWH_DIA = 4500.0

# Ruta del archivo que contiene los contornos de los techos.
RUTA_DATOS = (
    Path(__file__).parent.parent
    / "data"
    / "techos_campus_utn.npz"
)


def calcular_area_total_techos(
    ruta_datos: Path = RUTA_DATOS,
) -> float:
    """Calcula la superficie total de los techos en metros cuadrados."""
    # Se convierten las áreas calculadas en píxeles cuadrados a metros cuadrados.
    escala_area = ESCALA_METROS_POR_PIXEL**2
    # Se cargan todos los contornos para calcular su superficie total.
    techos = leer_techos(ruta_datos)

    # Se suman las áreas de todos los contornos del archivo de datos.
    return sum(
        area_gauss_pixeles_cuadrados(contorno) * escala_area
        for contorno in techos.values()
    )


def calcular_instalacion_fotovoltaica(
    area_total_techos_m2: float,
) -> tuple[float, float, int, float]:
    """Calcula superficie útil, área de paneles, cantidad y potencia pico."""
    # Un área negativa indicaría datos de entrada inválidos.
    if area_total_techos_m2 < 0:
        raise ValueError("El área total de los techos no puede ser negativa.")

    # Solo se considera instalable el porcentaje disponible de los techos.
    area_utilizable_m2 = (
        area_total_techos_m2 * PORCENTAJE_SUPERFICIE_UTILIZABLE
    )
    # Se calcula el área de un panel y cuántos caben completamente.
    area_panel_m2 = LARGO_PANEL_M * ANCHO_PANEL_M
    # La división entera descarta la fracción de panel que no puede instalarse.
    cantidad_paneles = int(area_utilizable_m2 // area_panel_m2)
    # La potencia pico total se convierte de Wp a kWp.
    potencia_pico_kwp = cantidad_paneles * POTENCIA_PANEL_WP / 1000

    # Se devuelven los cuatro valores necesarios para informar el diseño.
    return (
        area_utilizable_m2,
        area_panel_m2,
        cantidad_paneles,
        potencia_pico_kwp,
    )


def potencia_electrica_instantanea(
    irradiancia,
    potencia_pico_kwp: float,
    tiempo: np.ndarray | float,
) -> np.ndarray | float:
    """Calcula PAC(t) en kW para una irradiancia dada.

    Se considera la relación PAC(t) = Ppico * G(t) / 1000 * PR,
    donde G(t) está expresada en W/m².
    """
    # Se aplica la relación entre irradiancia, potencia pico y rendimiento.
    # El resultado conserva la forma de la función de irradiancia recibida.
    return (
        potencia_pico_kwp
        * irradiancia(tiempo)
        / 1000
        * FACTOR_RENDIMIENTO_GLOBAL
    )


def calcular_energia_neta_diaria(
    funcion_irradiancia,
    potencia_pico_kwp: float,
    cantidad_intervalos: int = 5000,
) -> float:
    """Integra PAC(t) y devuelve la energía diaria en kWh/día."""
    # Se transforma la función de irradiancia en una función de potencia AC.
    funcion_potencia = lambda tiempo: potencia_electrica_instantanea(
        funcion_irradiancia,
        potencia_pico_kwp,
        tiempo,
    )
    # Se integra la potencia entre el amanecer y el final de la producción solar.
    # La integral de kW durante horas produce la energía en kWh.
    return simpson_un_tercio(
        funcion_potencia,
        limite_inferior=6.0,
        limite_superior=20.0,
        cantidad_intervalos=cantidad_intervalos,
    )


def calcular_energias_netas_diarias(
    potencia_pico_kwp: float,
    cantidad_intervalos: int = 5000,
) -> dict[str, float]:
    """Calcula la energía diaria neta para los tres escenarios."""
    # Se asocia cada escenario de nubosidad con su modelo de irradiancia.
    escenarios = {
        "Día despejado": irradiancia_solar,
        "Día nublado": irradiancia_nublado,
        "Día oscilante": irradiancia_oscilante,
    }
    # Se integra la energía de cada escenario y se devuelve una tabla de resultados.
    return {
        nombre: calcular_energia_neta_diaria(
            funcion_irradiancia,
            potencia_pico_kwp,
            cantidad_intervalos,
        )
        for nombre, funcion_irradiancia in escenarios.items()
    }


def calcular_porcentajes_autoconsumo(
    energias_netas: dict[str, float],
    demanda_kwh_dia: float = DEMANDA_CAMPUS_KWH_DIA,
) -> dict[str, float]:
    """Calcula el porcentaje diario de demanda cubierto por el sistema."""
    # La demanda debe ser positiva para que el porcentaje tenga sentido.
    if demanda_kwh_dia <= 0:
        raise ValueError("La demanda debe ser mayor que cero.")

    # Se compara la energía generada con la demanda diaria del campus.
    return {
        escenario: energia / demanda_kwh_dia * 100
        for escenario, energia in energias_netas.items()
    }


def ejecutar_calculo_fotovoltaico() -> None:
    """Imprime los resultados del dimensionamiento fotovoltaico."""
    # La importación local evita cargar Matplotlib si solo se usan cálculos.
    from .graficos_fotovoltaicos import generar_grafico_potencia_ac

    # Se obtiene el área disponible y se dimensiona el parque solar.
    area_total_m2 = calcular_area_total_techos()
    (
        area_utilizable_m2,
        area_panel_m2,
        cantidad_paneles,
        potencia_pico_kwp,
    ) = calcular_instalacion_fotovoltaica(area_total_m2)

    # Se imprimen los parámetros principales de la instalación calculada.
    print("Dimensionamiento del parque solar:")
    print(f"- Superficie total de techos: {area_total_m2:.2f} m²")
    print(
        "- Superficie disponible (35 %): "
        f"{area_utilizable_m2:.2f} m²"
    )
    print(
        "- Superficie de cada panel: "
        f"{area_panel_m2:.4f} m²"
    )
    print(f"- Cantidad entera de paneles: {cantidad_paneles}")
    print(f"- Potencia pico instalada: {potencia_pico_kwp:.2f} kWp")
    print(
        "- Factor de rendimiento global considerado para etapas "
        f"posteriores: {FACTOR_RENDIMIENTO_GLOBAL:.2f}"
    )
    # Se genera el gráfico de potencia y se calculan las energías diarias.
    generar_grafico_potencia_ac(potencia_pico_kwp)
    # Se calculan las energías netas para cada escenario de nubosidad.
    energias_netas = calcular_energias_netas_diarias(potencia_pico_kwp)
    print("\nEnergía neta generada por día:")
    print(
        "- Método de integración: Simpson 1/3 compuesta "
        "(N = 5000)"
    )
    # Se imprime una línea por cada escenario evaluado.
    for escenario, energia in energias_netas.items():
        print(f"- {escenario}: {energia:.6f} kWh/día")

    # Se calcula qué proporción de la demanda podría cubrirse.
    porcentajes_autoconsumo = calcular_porcentajes_autoconsumo(
        energias_netas
    )
    print(
        "\nPorcentaje de autoconsumo cubierto "
        f"(demanda: {DEMANDA_CAMPUS_KWH_DIA:.0f} kWh/día):"
    )
    # Se informa el porcentaje de demanda cubierto en cada escenario.
    for escenario, porcentaje in porcentajes_autoconsumo.items():
        print(f"- {escenario}: {porcentaje:.2f} %")


# Se ejecuta el dimensionamiento solo al iniciar este módulo directamente.
if __name__ == "__main__":
    ejecutar_calculo_fotovoltaico()
