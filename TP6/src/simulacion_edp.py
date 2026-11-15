"""Simulación, captura y visualización del Ítem 2.a."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .integradores import integrar_edo
from .operadores_espaciales import aplicar_condiciones_contorno, calcular_baricentro, crear_malla
from .sistema_edp import (
    DISTANCIA_CAPTURA_EDP,
    crear_estado_inicial,
    derivadas_sistema_edp,
    distancia_captura,
    proyectar_contorno,
)

DIRECTORIO_RESULTADOS = Path(__file__).resolve().parent.parent / "resultados" / "item2"


def _simular(dt, malla, detener):
    """Integra una corrida y devuelve historia, posiciones y captura interpolada."""
    # El callback devuelve False cuando se solicita una corrida sin captura.
    callback = lambda t, estado: distancia_captura(t, estado, malla) if detener else False
    t_out, estados = integrar_edo(
        lambda t, estado: derivadas_sistema_edp(t, estado, malla),
        (0.0, 200.0),
        crear_estado_inicial(malla),
        dt,
        "rk4",
        post_paso=lambda t, estado: proyectar_contorno(t, estado, malla),
        condicion_parada=callback,
    )
    posiciones = []
    for t, estado in zip(t_out, estados):
        C = aplicar_condiciones_contorno(
            estado[:-2].reshape(malla.Ny, malla.Nx), t, malla
        )
        _, xc, yc = calcular_baricentro(C, malla.X, malla.Y, malla.pesos_simpson)
        posiciones.append((xc, yc, estado[-2], estado[-1]))
    posiciones = np.asarray(posiciones)
    captura = None
    if detener and len(t_out) > 1:
        for i in range(1, len(t_out)):
            d0 = np.linalg.norm(posiciones[i - 1, :2] - posiciones[i - 1, 2:])
            d1 = np.linalg.norm(posiciones[i, :2] - posiciones[i, 2:])
            if d1 <= DISTANCIA_CAPTURA_EDP:
                alpha = (d0 - DISTANCIA_CAPTURA_EDP) / (d0 - d1) if d0 != d1 else 1.0
                captura = (
                    t_out[i - 1] + alpha * (t_out[i] - t_out[i - 1]),
                    posiciones[i - 1, 2:] + alpha * (posiciones[i, 2:] - posiciones[i - 1, 2:]),
                )
                break
    return t_out, estados, posiciones, captura


def ejecutar_simulacion_2a():
    """Ejecuta los pasos pedidos, guarda tabla y genera mapas de captura."""
    # Se crea una única malla compartida por todas las corridas temporales.
    malla = crear_malla()
    # Se conservan los pasos solicitados, todos estrictamente menores que dt_max.
    dt_lista = (0.25, 0.10, 0.05)
    if any(dt >= malla.dt_max for dt in dt_lista):
        raise ValueError("Todos los pasos deben ser estrictamente menores que dt_max.")
    DIRECTORIO_RESULTADOS.mkdir(parents=True, exist_ok=True)
    # Se integran las tres resoluciones y se conserva la corrida más fina.
    corridas = [_simular(dt, malla, True) for dt in dt_lista]
    lineas = [f"dt_max = {malla.dt_max:.9f} s", "dt [s] | t_captura [s] | x_d [m] | y_d [m]"]
    for dt, corrida in zip(dt_lista, corridas):
        captura = corrida[3]
        lineas.append(f"{dt:.5f} | {captura[0]:.8f} | {captura[1][0]:.8f} | {captura[1][1]:.8f}")
    (DIRECTORIO_RESULTADOS / "2a_resultados_captura.txt").write_text("\n".join(lineas), encoding="utf-8")
    # La última corrida corresponde a dt=0.05, por lo que se reutiliza sin recalcularla.
    t_out, estados, posiciones, captura = corridas[-1]
    # Se eligen seis instantes equiespaciados hasta la captura.
    tiempos = np.linspace(0.0, t_out[-1], 6)
    figura, ejes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    # Los niveles fijos permiten comparar colores entre todos los paneles.
    niveles = np.linspace(0.0, 105.0, 22)
    ultimo = None
    for eje, instante in zip(ejes.ravel(), tiempos):
        # Se busca el estado almacenado más próximo al instante representativo.
        indice = int(np.argmin(np.abs(t_out - instante)))
        # Se reconstruye la concentración bidimensional del panel actual.
        C = aplicar_condiciones_contorno(
            estados[indice][:-2].reshape(malla.Ny, malla.Nx), t_out[indice], malla
        )
        # Se usa la misma escala de colores en los seis mapas.
        ultimo = eje.contourf(malla.X, malla.Y, C, levels=niveles, cmap="viridis")
        # Se dibujan las trayectorias acumuladas hasta el instante actual.
        eje.plot(posiciones[: indice + 1, 0], posiciones[: indice + 1, 1], "w-", label="Baricentro")
        eje.plot(posiciones[: indice + 1, 2], posiciones[: indice + 1, 3], "r-", label="Dron")
        # Se marca la posición instantánea de cada trayectoria con un punto.
        eje.plot(posiciones[indice, 0], posiciones[indice, 1], "wo", markersize=5)
        eje.plot(posiciones[indice, 2], posiciones[indice, 3], "ro", markersize=5)
        # Se fijan etiquetas, límites y proporción física idénticos en cada panel.
        eje.set(title=f"t = {t_out[indice]:.2f} s", xlabel="x [m]", ylabel="y [m]", xlim=(0, 130), ylim=(0, 130))
        eje.set_aspect("equal")
        eje.grid(alpha=0.25)
    # Se marca la captura exacta en el último panel y se incluye en su leyenda.
    if captura is not None:
        ejes.ravel()[-1].plot(captura[1][0], captura[1][1], "ko", label="Captura")
    if ultimo is not None:
        # La barra representa la escala común de concentración de todos los paneles.
        figura.colorbar(ultimo, ax=ejes.ravel().tolist(), label="C [g/m³]")
    # La leyenda se concentra en el último panel para evitar repetirla seis veces.
    ejes.ravel()[-1].legend(loc="upper right", fontsize=8)
    ruta = DIRECTORIO_RESULTADOS / "2a_evolucion_y_captura.png"
    figura.savefig(ruta, dpi=180)
    plt.close(figura)
    print("\n".join(lineas))
    return ruta
