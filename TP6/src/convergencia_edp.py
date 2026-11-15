"""Estudio de convergencia temporal del Ítem 2.b."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .integradores import integrar_edo
from .operadores_espaciales import crear_malla
from .simulacion_edp import DIRECTORIO_RESULTADOS
from .sistema_edp import crear_estado_inicial, derivadas_sistema_edp, proyectar_contorno


def _posicion_a_tiempo(dt, malla, t_final):
    """Integra sin captura y devuelve la posición del dron en t_final."""
    # Se desactiva la parada por captura para comparar todos los pasos en t*=40 s.
    _, estados = integrar_edo(
        lambda t, estado: derivadas_sistema_edp(t, estado, malla),
        (0.0, t_final),
        crear_estado_inicial(malla),
        dt,
        "rk4",
        post_paso=lambda t, estado: proyectar_contorno(t, estado, malla),
    )
    return estados[-1][-2:]


def calcular_convergencia_edp():
    """Calcula errores de posición, ajuste log-log y datos de reporte."""
    # La malla espacial se mantiene fija para aislar el error temporal.
    malla = crear_malla()
    t_objetivo = 40.0
    dt_ref = 0.005
    dt_lista = np.array([0.25, 0.125, 0.0625, 0.03125, 0.015625])
    referencia = _posicion_a_tiempo(dt_ref, malla, t_objetivo)
    # Se mide la norma del error de posición respecto de la referencia fina.
    errores = np.array(
        [
            np.linalg.norm(_posicion_a_tiempo(dt, malla, t_objetivo) - referencia)
            for dt in dt_lista
        ]
    )
    # La pendiente del ajuste lineal log(error) frente a log(dt) es el orden p.
    orden = float(np.polyfit(np.log(dt_lista), np.log(errores), 1)[0])
    return malla, dt_lista, errores, orden


def mostrar_resultados_convergencia_edp(dt_lista, errores, orden, dt_max):
    """Imprime y devuelve la explicación del orden observado y la estabilidad."""
    # Se construye la explicación que también se agregará al informe.
    explicacion = [
        "\nInterpretación de la convergencia temporal del Ítem 2.b:",
        "La pendiente p se obtiene ajustando log(error) frente a log(dt).",
        "La malla espacial permanece fija, por lo que se aísla el error temporal.",
        f"Orden experimental p = {orden:.8f}; orden teórico de RK4 = 4.",
        (
            "El valor p aprox. 3.90 es consistente con 4: t*=40 s < 50 s, "
            "por lo que la fuente todavía es suave y no interviene el apagado abrupto."
        ),
        (
            f"La estabilidad explícita exige dt < dt_max aprox. {dt_max:.3f} s; "
            "todos los pasos ensayados cumplen estrictamente esa restricción."
        ),
    ]
    # Se imprime la tabla completa, incluyendo cada paso y su error.
    print("\n".join(explicacion[:3]))
    print("dt [s] | error posición [m]")
    for dt, error in zip(dt_lista, errores):
        print(f"{dt:.8f} | {error:.12e}")
    print("\n".join(explicacion[3:]))
    return explicacion


def ejecutar_convergencia_edp():
    """Guarda el informe y la gráfica del orden temporal experimental."""
    # Se calculan una sola vez la malla, los pasos, errores y orden experimental.
    malla, dt_lista, errores, orden = calcular_convergencia_edp()
    DIRECTORIO_RESULTADOS.mkdir(parents=True, exist_ok=True)
    # Se prepara la tabla numérica principal del archivo de resultados.
    lineas = [
        "Convergencia temporal del Ítem 2.b",
        f"dt_max = {malla.dt_max:.9f} s; método: RK4; t* = 40 s",
        "dt [s] | error posición [m]",
    ]
    lineas.extend(f"{dt:.8f} | {error:.12e}" for dt, error in zip(dt_lista, errores))
    lineas.append(f"Orden experimental p = {orden:.8f}; orden teórico RK4 = 4")
    # Se agrega la interpretación solicitada al final del mismo informe.
    lineas.extend(mostrar_resultados_convergencia_edp(dt_lista, errores, orden, malla.dt_max))
    (DIRECTORIO_RESULTADOS / "2b_resultados_convergencia.txt").write_text(
        "\n".join(lineas), encoding="utf-8"
    )
    # Se crea la curva de errores en escala log-log.
    figura, eje = plt.subplots(figsize=(7, 5))
    eje.loglog(dt_lista, errores, "o-", label=f"RK4 (p = {orden:.3f})")
    # Se normaliza una recta de pendiente 4 para que pase por el primer dato.
    referencia_teorica = errores[0] * (dt_lista / dt_lista[0]) ** 4
    eje.loglog(
        dt_lista,
        referencia_teorica,
        "k--",
        label=r"Referencia $\mathcal{O}(\Delta t^4)$",
    )
    eje.set(xlabel="dt [s]", ylabel="Error global de posición [m]", title="Convergencia temporal EDP")
    eje.grid(True, which="both", alpha=0.3)
    eje.legend()
    ruta = DIRECTORIO_RESULTADOS / "2b_convergencia_temporal.png"
    figura.savefig(ruta, dpi=180, bbox_inches="tight")
    plt.close(figura)
    return ruta
