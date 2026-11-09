# Path permite definir la carpeta de resultados.
from pathlib import Path

# Matplotlib dibuja las tablas y curvas; NumPy genera la malla temporal.
import matplotlib.pyplot as plt
import numpy as np

from .calculo_irradiancia_diaria import (
    irradiancia_nublado,
    irradiancia_oscilante,
    irradiancia_solar,
    simpson_un_tercio,
)


# Carpeta de resultados correspondiente al ejercicio de irradiancia.
RUTA_RESULTADOS = (
    Path(__file__).parent.parent
    / "resultados"
    / "ejercicio2"
)


def comparar_escenarios_nubosidad() -> None:
    """Genera la tabla de energía para los escenarios de nubosidad."""
    # Se comparan distintas cantidades de intervalos para observar convergencia.
    lista_N = [10, 50, 100, 500, 1000, 5000]
    filas_tabla = []

    print("\nImpacto de la nubosidad:")
    print(
        f"{'N':>6} | "
        f"{'Día nublado (kWh/m²·día)':>28} | "
        f"{'Día oscilante (kWh/m²·día)':>30}"
    )
    print("-" * 72)

    # Se integra cada escenario y se almacenan sus resultados para la tabla.
    for cantidad_intervalos in lista_N:
        if cantidad_intervalos % 2 != 0:
            cantidad_intervalos += 1

        energia_nublada = simpson_un_tercio(
            irradiancia_nublado,
            limite_inferior=6.0,
            limite_superior=20.0,
            cantidad_intervalos=cantidad_intervalos,
        ) / 1000
        energia_oscilante = simpson_un_tercio(
            irradiancia_oscilante,
            limite_inferior=6.0,
            limite_superior=20.0,
            cantidad_intervalos=cantidad_intervalos,
        ) / 1000

        filas_tabla.append(
            [
                cantidad_intervalos,
                f"{energia_nublada:.9f}",
                f"{energia_oscilante:.9f}",
            ]
        )
        print(
            f"{cantidad_intervalos:6d} | "
            f"{energia_nublada:28.9f} | "
            f"{energia_oscilante:30.9f}"
        )

    # Se crea la carpeta de salida y se prepara la tabla como una figura.
    RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    ruta_salida = RUTA_RESULTADOS / "comparacion_nubosidad.png"

    # Se construye una tabla visual con los valores calculados.
    figura, eje = plt.subplots(figsize=(12, 3.5))
    eje.axis("off")
    tabla = eje.table(
        cellText=filas_tabla,
        colLabels=[
            "N",
            "Día nublado (kWh/m²·día)",
            "Día oscilante (kWh/m²·día)",
        ],
        cellLoc="center",
        loc="center",
        colWidths=[0.12, 0.38, 0.42],
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(11)
    tabla.scale(1, 1.8)
    eje.set_title(
        "Impacto de la nubosidad",
        fontsize=14,
        fontweight="bold",
        pad=18,
    )
    figura.tight_layout()
    # Se guarda la tabla y se cierra la figura.
    figura.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    plt.close(figura)
    print(f"Tabla guardada en: {ruta_salida}")


def generar_perfil_diario_irradiancia() -> None:
    """Genera el perfil continuo de irradiancia de los tres escenarios."""
    # Se muestrea el intervalo diurno para dibujar perfiles continuos.
    tiempo = np.linspace(6.0, 20.0, 2001)

    # Se prepara la ruta de salida y el lienzo del gráfico.
    RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    ruta_salida = RUTA_RESULTADOS / "perfil_diario_irradiancia.png"

    # Se grafica la irradiancia de cada condición de nubosidad.
    figura, eje = plt.subplots(figsize=(12, 6))
    eje.plot(
        tiempo,
        irradiancia_solar(tiempo),
        label="Día despejado: G(t)",
        color="tab:orange",
        linewidth=2,
    )
    eje.plot(
        tiempo,
        irradiancia_nublado(tiempo),
        label="Día nublado: G_nublado(t)",
        color="tab:blue",
        linewidth=2,
    )
    eje.plot(
        tiempo,
        irradiancia_oscilante(tiempo),
        label="Cúmulos oscilantes: G_oscilante(t)",
        color="tab:green",
        linewidth=1.5,
    )
    # Se agregan etiquetas, límites, grilla y leyenda.
    eje.set_title("Perfil diario de irradiancia")
    eje.set_xlabel("Hora del día, t (h)")
    eje.set_ylabel("Irradiancia solar (W/m²)")
    eje.set_xlim(6.0, 20.0)
    eje.set_ylim(bottom=0.0)
    eje.grid(True, alpha=0.3)
    eje.legend()
    figura.tight_layout()
    # Se guarda el perfil y se libera la figura.
    figura.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    plt.close(figura)
    print(f"Perfil diario guardado en: {ruta_salida}")
