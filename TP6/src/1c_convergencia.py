"""Cálculo y visualización separados del análisis de convergencia temporal."""

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Se identifica la carpeta src para localizar los módulos de los incisos anteriores.
DIRECTORIO_SRC = Path(__file__).resolve().parent


def _cargar_modulo(nombre, ruta):
    """Carga un archivo Python aunque su nombre comience con un dígito."""

    # Se crea la especificación que vincula un nombre interno con el archivo.
    especificacion = importlib.util.spec_from_file_location(nombre, ruta)
    # Se rechaza una especificación incompleta porque no permitiría ejecutar el módulo.
    if especificacion is None or especificacion.loader is None:
        raise ImportError(f"No se pudo cargar el módulo ubicado en {ruta}.")
    # Se crea el objeto módulo vacío a partir de la especificación.
    modulo = importlib.util.module_from_spec(especificacion)
    # Se ejecuta el código del archivo dentro del módulo creado.
    especificacion.loader.exec_module(modulo)
    # Se devuelve el módulo para acceder a sus funciones y constantes.
    return modulo


# Se carga la definición del sistema dinámico del inciso 1.a.
modulo_sistema = _cargar_modulo("sistema_dinamico", DIRECTORIO_SRC / "1a_sistema_dinamico.py")
# Se extrae la función que calcula el lado derecho F(t, X).
derivadas_sistema = modulo_sistema.derivadas_sistema
# Se carga el módulo que contiene los integradores Euler y RK4.
modulo_integradores = _cargar_modulo("integradores", DIRECTORIO_SRC / "1b_integradores.py")
# Se extrae la función maestra de integración temporal.
integrar_edo = modulo_integradores.integrar_edo


def _estado_en_tiempo(t_out, X_out, tiempo_objetivo):
    """Obtiene el estado asociado a un instante representado en la grilla."""

    # Se localiza el tiempo de la salida más cercano al instante solicitado.
    indice = int(np.argmin(np.abs(t_out - tiempo_objetivo)))
    # Se comprueba que la diferencia sea sólo el redondeo de punto flotante.
    if not np.isclose(t_out[indice], tiempo_objetivo, rtol=0.0, atol=1e-8):
        raise RuntimeError(
            f"El tiempo {tiempo_objetivo} s no está presente exactamente en la grilla."
        )
    # Se devuelve el estado y su índice temporal.
    return X_out[indice], indice


def _error_posicion(estado, posicion_referencia):
    """Calcula la norma euclídea del error de posición del dron."""

    # Se extraen las coordenadas del dron, componentes cuarta y quinta del estado.
    posicion_dron = estado[4:6]
    # Se calcula el vector diferencia respecto de la referencia.
    diferencia = posicion_dron - posicion_referencia
    # Se devuelve la norma euclídea como error escalar en metros.
    return float(np.linalg.norm(diferencia))


def calcular_errores_y_ordenes():
    """Calcula errores y órdenes empíricos sin producir salida por consola."""

    # Se fija el extremo inicial del intervalo de integración.
    t_inicial = 0.0
    # Se fija el extremo final de la misión.
    t_final = 60.0
    # Se fija el primer instante de evaluación.
    t1 = 15.0
    # Se fija el segundo instante de evaluación.
    t2 = 60.0
    # Se utiliza un paso ultra fino para generar la referencia numérica.
    dt_ref = 0.001
    # Se define la secuencia de pasos temporales del estudio.
    dt_lista = np.array([2.0, 1.0, 0.5, 0.25, 0.125, 0.0625], dtype=float)
    # Se calcula la solución de referencia mediante RK4.
    t_ref, X_ref = integrar_edo(
        derivadas_sistema,
        (t_inicial, t_final),
        modulo_sistema.X0,
        dt_ref,
        "rk4",
    )
    # Se obtiene el estado de referencia durante la persecución activa.
    estado_ref_t1, _ = _estado_en_tiempo(t_ref, X_ref, t1)
    # Se obtiene el estado de referencia al final de la misión.
    estado_ref_t2, _ = _estado_en_tiempo(t_ref, X_ref, t2)
    # Se guardan sólo las posiciones de referencia del dron en t1.
    posicion_ref_t1 = estado_ref_t1[4:6]
    # Se guardan sólo las posiciones de referencia del dron en t2.
    posicion_ref_t2 = estado_ref_t2[4:6]
    # Se reservan vectores de error para cada método y cada instante.
    errores = {
        "euler": {"t1": np.zeros(dt_lista.size), "t2": np.zeros(dt_lista.size)},
        "rk4": {"t1": np.zeros(dt_lista.size), "t2": np.zeros(dt_lista.size)},
    }

    # Se recorren los métodos de integración que serán comparados.
    for metodo in ("euler", "rk4"):
        # Se recorren todos los pasos temporales definidos.
        for indice_dt, dt in enumerate(dt_lista):
            # Se integra hasta el final de la misión con el método actual.
            t_out, X_out = integrar_edo(
                derivadas_sistema,
                (t_inicial, t_final),
                modulo_sistema.X0,
                dt,
                metodo,
            )
            # Se identifica el punto más cercano a t1 en la grilla calculada.
            indice_t1 = int(np.argmin(np.abs(t_out - t1)))
            # Se comprueba si t1 pertenece exactamente a la grilla temporal.
            if np.isclose(t_out[indice_t1], t1, rtol=0.0, atol=1e-8):
                # Se usa directamente el estado almacenado en t1.
                estado_t1 = X_out[indice_t1]
            else:
                # Se integra hasta t1 para aplicar allí un último paso parcial exacto.
                _, estados_hasta_t1 = integrar_edo(
                    derivadas_sistema,
                    (t_inicial, t1),
                    modulo_sistema.X0,
                    dt,
                    metodo,
                )
                # Se toma el último estado, que corresponde exactamente a t1.
                estado_t1 = estados_hasta_t1[-1]
            # Se obtiene directamente el estado de t2, que siempre es el extremo final.
            estado_t2, _ = _estado_en_tiempo(t_out, X_out, t2)
            # Se guarda el error de posición en t1 para el método y paso actuales.
            errores[metodo]["t1"][indice_dt] = _error_posicion(estado_t1, posicion_ref_t1)
            # Se guarda el error de posición en t2 para el método y paso actuales.
            errores[metodo]["t2"][indice_dt] = _error_posicion(estado_t2, posicion_ref_t2)

    # Se reserva un diccionario para las pendientes log-log de cada curva.
    ordenes = {"t1": {}, "t2": {}}
    # Se recorren los dos instantes usados para medir la convergencia.
    for nombre_tiempo in ("t1", "t2"):
        # Se recorren Euler y RK4 para ajustar una recta a cada conjunto de errores.
        for metodo in ("euler", "rk4"):
            # Se transforma el paso temporal a escala logarítmica.
            log_dt = np.log(dt_lista)
            # Se transforma el error a escala logarítmica.
            log_error = np.log(errores[metodo][nombre_tiempo])
            # La pendiente del ajuste lineal es el orden empírico p.
            pendiente, _ = np.polyfit(log_dt, log_error, 1)
            # Se almacena la pendiente como número real.
            ordenes[nombre_tiempo][metodo] = float(pendiente)

    # Se devuelve todo lo necesario para mostrar o graficar los resultados.
    return dt_lista, errores, ordenes, t1, t2


def mostrar_resultados_consola(ordenes):
    """Explica e imprime los órdenes empíricos de convergencia calculados."""

    # Se explica que p es la pendiente de la recta en escala log-log.
    print("\nInterpretación del orden de convergencia:")
    # Se relaciona el orden p con la ley asintótica del error numérico.
    print(
        "El valor p mide cómo disminuye el error global cuando se reduce el paso "
        "dt: aproximadamente E(dt) = K · dt^p."
    )
    # Se aclara que un p mayor implica una reducción más rápida del error.
    print(
        "Por lo tanto, cuanto mayor sea p, más rápidamente converge el método "
        "al disminuir dt.\n"
    )
    # Se imprime el encabezado de la tabla con espaciado legible.
    print("Órdenes empíricos obtenidos")
    # Se imprime una separación visual superior.
    print("=" * 72)
    # Se imprimen los títulos de las columnas evaluadas.
    print(f"{'Método':<18}{'Orden p en t = 15 s':>26}{'Orden p en t = 60 s':>26}")
    # Se imprime una separación visual bajo los títulos.
    print("-" * 72)
    # Se imprime el orden empírico de Euler en ambos instantes.
    print(f"{'Euler':<18}{ordenes['t1']['euler']:>26.6f}{ordenes['t2']['euler']:>26.6f}")
    # Se imprime el orden empírico de RK4 en ambos instantes.
    print(f"{'RK4':<18}{ordenes['t1']['rk4']:>26.6f}{ordenes['t2']['rk4']:>26.6f}")
    # Se imprime una separación visual inferior.
    print("=" * 72)
    # Se introduce la interpretación teórica del resultado de RK4 al final.
    print("\nConclusión teórica:")
    # Se explica la causa de la pérdida de orden en el instante final.
    print(
        "En t = 15 s, RK4 presenta un orden cercano a 4, como predice la teoría "
        "para un campo vectorial suficientemente suave."
    )
    # Se explica explícitamente el efecto de la captura sobre el campo dinámico.
    print(
        "En t = 60 s, el orden de RK4 baja aproximadamente a 1 porque, cuando "
        "la distancia entre el dron y la nube es menor que 1.2 m, se activa la "
        "captura: el barrido ortogonal se apaga."
    )
    # Se vincula la conmutación de régimen con la hipótesis de suavidad de RK4.
    print(
        "Esa conmutación introduce una discontinuidad en el campo vectorial "
        "F(t, X), rompiendo la hipótesis de suavidad necesaria para conservar "
        "el orden teórico cuatro de Runge-Kutta."
    )


def graficar_curvas_error(dt_lista, errores, ordenes, t1, t2):
    """Genera y guarda la figura log-log a partir de datos ya calculados."""

    # Se crea una figura con un panel para cada instante analizado.
    figura, ejes = plt.subplots(1, 2, figsize=(13.0, 5.5), constrained_layout=True)
    # Se vincula cada eje con la clave de sus errores y su instante físico.
    configuraciones = ((ejes[0], "t1", t1), (ejes[1], "t2", t2))
    # Se recorren las dos configuraciones de los subgráficos.
    for eje, nombre_tiempo, instante in configuraciones:
        # Se representa el error de Euler en ambas escalas logarítmicas.
        eje.loglog(
            dt_lista,
            errores["euler"][nombre_tiempo],
            marker="o",
            linewidth=1.8,
            label=f"Euler (p = {ordenes[nombre_tiempo]['euler']:.3f})",
        )
        # Se representa el error de RK4 en ambas escalas logarítmicas.
        eje.loglog(
            dt_lista,
            errores["rk4"][nombre_tiempo],
            marker="s",
            linewidth=1.8,
            label=f"RK4 (p = {ordenes[nombre_tiempo]['rk4']:.3f})",
        )
        # Se etiqueta el eje horizontal con el paso temporal.
        eje.set_xlabel("Paso temporal dt [s]")
        # Se etiqueta el eje vertical con el error de posición.
        eje.set_ylabel("Error global de posición [m]")
        # Se titula el panel con el instante correspondiente.
        eje.set_title(f"Convergencia en t = {instante:.0f} s")
        # Se agrega una cuadrícula para facilitar la lectura logarítmica.
        eje.grid(True, which="both", alpha=0.3)
        # Se agrega la leyenda con el método y su orden empírico.
        eje.legend()

    # Se calcula la raíz del proyecto a partir de la carpeta src.
    directorio_raiz = DIRECTORIO_SRC.parent
    # Se define la carpeta de salida para los gráficos.
    directorio_graficos = directorio_raiz / "graficos"
    # Se crea la carpeta de salida si todavía no existe.
    directorio_graficos.mkdir(parents=True, exist_ok=True)
    # Se define la ruta del gráfico solicitado.
    ruta_salida = directorio_graficos / "1c_convergencia_errores.png"
    # Se guarda la figura sin abrir ninguna ventana.
    figura.savefig(ruta_salida, dpi=200, bbox_inches="tight")
    # Se cierra la figura para liberar los recursos de Matplotlib.
    plt.close(figura)
    # Se devuelve la ruta para permitir verificar o reutilizar la salida.
    return ruta_salida
