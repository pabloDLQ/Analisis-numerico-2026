"""Simulación y visualización del sistema de dispersión y persecución."""

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Se obtiene la carpeta que contiene este archivo para construir rutas portables.
DIRECTORIO_SRC = Path(__file__).resolve().parent


def cargar_modulo(nombre, ruta):
    """Carga un módulo Python desde una ruta, incluso si su nombre comienza con un dígito."""

    # Se crea una especificación de importación asociada al archivo solicitado.
    especificacion = importlib.util.spec_from_file_location(nombre, ruta)
    # Se informa explícitamente si Python no pudo crear la especificación.
    if especificacion is None or especificacion.loader is None:
        raise ImportError("No se pudo construir la especificación del módulo.")
    # Se crea un objeto módulo vacío a partir de la especificación.
    modulo = importlib.util.module_from_spec(especificacion)
    # Se ejecuta el código fuente dentro del objeto módulo recién creado.
    especificacion.loader.exec_module(modulo)
    # Se devuelve el módulo cargado para acceder a sus funciones y constantes.
    return modulo


# Se carga la definición del sistema dinámico del inciso 1.a.
modulo_sistema = cargar_modulo("sistema_dinamico", DIRECTORIO_SRC / "1a_sistema_dinamico.py")
# Se extrae la función de derivadas requerida por el integrador.
derivadas_sistema = modulo_sistema.derivadas_sistema
# Se carga el módulo propio de integradores, sin usar scipy.integrate.
modulo_integradores = cargar_modulo("integradores", DIRECTORIO_SRC / "1b_integradores.py")
# Se extrae la función maestra que permite seleccionar RK4.
integrar_edo = modulo_integradores.integrar_edo


def concentracion_gaussiana(x, y, xc, yc, sigma, masa):
    """Evalúa la concentración gaussiana bidimensional de la nube."""

    # Se valida que el ancho gaussiano sea físicamente positivo.
    if sigma <= 0.0:
        raise ValueError("sigma debe ser estrictamente positivo.")
    # Se calcula el factor de normalización de una gaussiana bidimensional.
    normalizacion = masa / (2.0 * np.pi * sigma**2)
    # Se calcula el exponente asociado a la distancia cuadrática al centro.
    exponente = -((x - xc) ** 2 + (y - yc) ** 2) / (2.0 * sigma**2)
    # Se devuelve la concentración evaluada punto a punto sobre la malla.
    return normalizacion * np.exp(exponente)


def ejecutar_simulacion_1b():
    """Integra el sistema y genera la figura de los seis instantes solicitados."""

    # Se fija el paso fino para resolver la dinámica con buena precisión temporal.
    dt = 0.05
    # Se integra desde cero hasta sesenta segundos con el método RK4.
    t_out, X_out = integrar_edo(derivadas_sistema, (0.0, 60.0), modulo_sistema.X0, dt, "rk4")
    # Se define la malla espacial común a todos los subgráficos.
    x = np.linspace(0.0, 160.0, 161)
    # Se define la segunda coordenada espacial con los mismos límites fijos.
    y = np.linspace(0.0, 160.0, 161)
    # Se construye la malla cartesiana sobre la cual se evaluará C(x,y).
    X_malla, Y_malla = np.meshgrid(x, y)
    # Se seleccionan los seis tiempos representativos, todos presentes en la grilla temporal.
    tiempos_figura = np.array([0.0, 12.0, 24.0, 36.0, 48.0, 60.0])
    # Se crea una figura organizada en dos filas y tres columnas.
    figura, ejes = plt.subplots(2, 3, figsize=(17.0, 10.0), constrained_layout=True)
    # Se calcula un nivel máximo común para que los colores sean comparables entre paneles.
    masa_maxima = float(np.max(X_out[:, 3]))
    sigma_minima = float(np.min(X_out[:, 2]))
    concentracion_maxima = masa_maxima / (2.0 * np.pi * sigma_minima**2)
    # Se construyen niveles igualmente espaciados para todos los mapas de contorno.
    niveles = np.linspace(0.0, concentracion_maxima, 21)
    # Se reserva el último objeto de contorno para construir una barra de color compartida.
    ultimo_contorno = None

    # Se recorre cada eje junto con el instante que debe representar.
    for eje, instante in zip(ejes.ravel(), tiempos_figura):
        # Se obtiene el índice temporal correspondiente al instante solicitado.
        indice = int(np.argmin(np.abs(t_out - instante)))
        # Se recuperan las variables de estado de la nube y del dron hasta ese instante.
        estado_hasta_ahora = X_out[: indice + 1]
        # Se extraen las coordenadas del centro de la nube para dibujar su historia.
        xc_historia, yc_historia = estado_hasta_ahora[:, 0], estado_hasta_ahora[:, 1]
        # Se extraen las coordenadas del dron para dibujar su historia.
        xd_historia, yd_historia = estado_hasta_ahora[:, 4], estado_hasta_ahora[:, 5]
        # Se extraen las variables escalares de la nube en el instante actual.
        xc, yc, sigma, masa, xd, yd = X_out[indice]
        # Se evalúa la distribución gaussiana sobre toda la malla espacial.
        concentracion = concentracion_gaussiana(X_malla, Y_malla, xc, yc, sigma, masa)
        # Se pinta el mapa de concentración con niveles comunes a todos los paneles.
        ultimo_contorno = eje.contourf(X_malla, Y_malla, concentracion, levels=niveles, cmap="viridis")
        # Se superpone la trayectoria histórica del centro de la nube.
        eje.plot(xc_historia, yc_historia, color="white", linewidth=2.0, label="Trayectoria nube")
        # Se superpone la trayectoria histórica del dron.
        eje.plot(xd_historia, yd_historia, color="red", linewidth=1.8, label="Trayectoria dron")
        # Se marca la posición actual del centro de la nube.
        eje.plot(xc, yc, marker="o", color="cyan", markersize=7, label="Nube actual")
        # Se marca la posición actual del dron.
        eje.plot(xd, yd, marker="^", color="yellow", markeredgecolor="black", markersize=7, label="Dron actual")
        # Se fija el dominio espacial pedido para cada panel.
        eje.set_xlim(0.0, 160.0)
        # Se fija el mismo límite vertical para facilitar la comparación.
        eje.set_ylim(0.0, 160.0)
        # Se etiqueta el eje horizontal en metros.
        eje.set_xlabel("x [m]")
        # Se etiqueta el eje vertical en metros.
        eje.set_ylabel("y [m]")
        # Se agrega una cuadrícula tenue para leer posiciones sobre el mapa.
        eje.grid(True, alpha=0.3)
        # Se identifica el tiempo mostrado en el título.
        eje.set_title(f"t = {t_out[indice]:.0f} s")
        # Se evita repetir la leyenda completa en cada panel.
        if eje is ejes.ravel()[0]:
            eje.legend(loc="upper left", fontsize=8)

    # Se agrega una barra de color común, cuyo rango representa concentración.
    if ultimo_contorno is not None:
        figura.colorbar(ultimo_contorno, ax=ejes.ravel().tolist(), label="C(x, y, t)")
    # Se calcula la carpeta raíz del proyecto a partir de la carpeta src actual.
    directorio_raiz = DIRECTORIO_SRC.parent
    # Se define la carpeta destinada a almacenar las figuras generadas.
    directorio_graficos = directorio_raiz / "graficos"
    # Se crea la carpeta de gráficos y sus padres si todavía no existen.
    directorio_graficos.mkdir(parents=True, exist_ok=True)
    # Se define el nombre solicitado para la imagen de la simulación 1.b.
    ruta_salida = directorio_graficos / "1b_evolucion_concentracion_y_trayectorias.png"
    # Se guarda la figura en formato PNG con resolución adecuada para informes.
    figura.savefig(ruta_salida, dpi=200, bbox_inches="tight")
    # Se cierra la figura para liberar memoria y evitar mostrarla por pantalla.
    plt.close(figura)
    # Se devuelve la ruta para que otros módulos puedan informar o verificar la salida.
    return ruta_salida
