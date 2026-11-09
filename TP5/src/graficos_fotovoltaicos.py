# Path permite definir la carpeta donde se guardará el gráfico.
from pathlib import Path

# Matplotlib y NumPy se usan para construir y muestrear el gráfico.
import matplotlib.pyplot as plt
import numpy as np

from .calculo_fotovoltaico import potencia_electrica_instantanea
from .calculo_irradiancia_diaria import (
    irradiancia_nublado,
    irradiancia_oscilante,
    irradiancia_solar,
)


# Carpeta de resultados correspondiente al ejercicio fotovoltaico.
RUTA_RESULTADOS = (
    Path(__file__).parent.parent
    / "resultados"
    / "ejercicio3"
)


def generar_grafico_potencia_ac(
    potencia_pico_kwp: float,
) -> None:
    """Grafica PAC(t) para los tres escenarios de nubosidad."""
    # Se crea una malla fina de horas para obtener curvas suaves.
    tiempo = np.linspace(6.0, 20.0, 2001)
    # Se definen los modelos de irradiancia, etiquetas y colores del gráfico.
    escenarios = (
        (
            "Día despejado: PAC(t)",
            irradiancia_solar,
            "tab:orange",
        ),
        (
            "Día nublado: PAC(t)",
            irradiancia_nublado,
            "tab:blue",
        ),
        (
            "Día oscilante: PAC(t)",
            irradiancia_oscilante,
            "tab:green",
        ),
    )

    # Se prepara la carpeta y el nombre del archivo de salida.
    RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    ruta_salida = RUTA_RESULTADOS / "potencia_ac_escenarios.png"

    # Se crea una figura y se dibuja la potencia de cada escenario.
    figura, eje = plt.subplots(figsize=(12, 6))
    for etiqueta, funcion_irradiancia, color in escenarios:
        # Se evalúa la potencia instantánea a lo largo de todo el día.
        potencia_ac = potencia_electrica_instantanea(
            funcion_irradiancia,
            potencia_pico_kwp,
            tiempo,
        )
        eje.plot(
            tiempo,
            potencia_ac,
            label=etiqueta,
            color=color,
            linewidth=1.8,
        )

    # Se configuran los textos, límites, grilla y leyenda del gráfico.
    eje.set_title("Potencia eléctrica instantánea entregada a la red")
    eje.set_xlabel("Hora del día, t (h)")
    eje.set_ylabel("Potencia AC (kW)")
    eje.set_xlim(6.0, 20.0)
    eje.set_ylim(bottom=0.0)
    eje.grid(True, alpha=0.3)
    eje.legend()
    figura.tight_layout()
    # Se guarda la figura y se cierra para liberar recursos.
    figura.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    plt.close(figura)
    print(f"Gráfico de potencia AC guardado en: {ruta_salida}")
