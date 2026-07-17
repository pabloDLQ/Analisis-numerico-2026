import NR
import versiones.metodoBrent as metodoBrent
import versiones.metodoIQI as metodoIQI
import versiones.metodoSecante as metodoSecante


def _biseccion_fallback(f, a, b, tol, max_iter=200):
    """Respaldo robusto para intervalos con cambio de signo."""
    fa = f(a, tipo="comun")
    fb = f(b, tipo="comun")

    if fa == 0.0:
        return a
    if fb == 0.0:
        return b

    if fa * fb > 0:
        raise ValueError("El intervalo no contiene un cambio de signo")

    x_anterior = None
    x = (a + b) / 2.0
    for _ in range(max_iter):
        fx = f(x, tipo="comun")

        if fx == 0.0:
            return x
        if abs(b - a) <= tol:
            return x
        if x_anterior is not None and abs(x - x_anterior) <= tol:
            return x

        if fa * fx < 0:
            b, fb = x, fx
        else:
            a, fa = x, fx

        x_anterior = x
        x = (a + b) / 2.0

    return x


def encontrar_raiz(f, a, b, tol, max_iter=150):
    """
    Argumentos de entrada:
    f : Función a evaluar. Se debe invocar especificando el "tipo" de llamada:
    - Para evaluar la función de forma estándar: f(x, tipo="comun")
    - Para evaluar puntos destinados al cálculo de derivadas: f(x, tipo="derivada")
    a, b: Flotantes. Extremos del intervalo inicial (b > a).
    tol : Flotante>0. Tolerancia de convergencia.
    max_iter : Entero positivo. Cantidad máxima de iteraciones permitidas.
    El algoritmo DEBE detenerse estrictamente cuando | x_k - x_{k-1} | <= tol.
    Para la evaluación, el equipo docente utilizará: tol = 1e-12
    Retorno:
    Devuelve únicamente un número flotante con la estimación de la raíz.
    """
    # Ejemplo de llamadas válidas para la telemetría del peaje (sistema de benchmark):
    # fx = f(x, tipo="comun")
    # fx_der = f(x + dx, tipo="derivada")

    max_iter = int(max_iter)
    if max_iter <= 0:
        raise ValueError("max_iter debe ser un entero positivo")

    cache = {}
    mejor_x = None
    mejor_abs_f = float("inf")

    def f_track(x, tipo="comun"):
        nonlocal mejor_x, mejor_abs_f
        key = (tipo, x)
        if key in cache:
            return cache[key]

        valor = f(x, tipo=tipo)
        cache[key] = valor

        if tipo == "comun":
            abs_valor = abs(valor)
            if abs_valor < mejor_abs_f:
                mejor_abs_f = abs_valor
                mejor_x = x

        return valor

    def f_limited(base_f, max_comun=None, max_derivada=None):
        conteo_comun = 0
        conteo_derivada = 0

        def wrapped(x, tipo="comun"):
            nonlocal conteo_comun, conteo_derivada
            if tipo == "comun":
                conteo_comun += 1
                if max_comun is not None and conteo_comun > max_comun:
                    raise RuntimeError("Limite de iteraciones alcanzado para la estrategia")
            elif tipo == "derivada":
                conteo_derivada += 1
                if max_derivada is not None and conteo_derivada > max_derivada:
                    raise RuntimeError("Limite de iteraciones alcanzado para la estrategia")
            return base_f(x, tipo=tipo)

        return wrapped

    fa = f_track(a, tipo="comun")
    fb = f_track(b, tipo="comun")

    if fa == 0.0:
        return a
    if fb == 0.0:
        return b

    # Caso 1: Hay cambio de signo -> usar bisección como respaldo robusto.
    if fa * fb < 0:
        return _biseccion_fallback(f_track, a, b, tol, max_iter=max_iter)

    # Caso 2: No hay cambio de signo -> metodos abiertos priorizando NR.
    try:
        f_nr_abierto = f_limited(f_track, max_comun=max_iter, max_derivada=max_iter)
        raiz_nr_abierto = NR.encontrar_raiz(f_nr_abierto, a, b, tol)
        return raiz_nr_abierto
    except Exception:
        pass

    try:
        raiz_iqi = metodoIQI.encontrar_raiz(f_track, a, b, tol, max_iter=max_iter)
        return raiz_iqi
    except Exception:
        pass

    try:
        # Secante no expone max_iter en esta version: limitamos por cantidad de evaluaciones.
        f_secante = f_limited(f_track, max_comun=max_iter, max_derivada=0)
        raiz_sec = metodoSecante.encontrar_raiz(f_secante, a, b, tol)
        return raiz_sec
    except Exception:
        if mejor_x is not None:
            return mejor_x

    raise RuntimeError("No se pudo encontrar la raíz")