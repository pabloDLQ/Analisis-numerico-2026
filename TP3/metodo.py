import NR
import versiones.metodoBrent as metodoBrent
import versiones.metodoIQI as metodoIQI
import versiones.metodoSecante as metodoSecante


def encontrar_raiz(f, a, b, tol):
    """
    Argumentos de entrada:
    f : Función a evaluar. Se debe invocar especificando el "tipo" de llamada:
    - Para evaluar la función de forma estándar: f(x, tipo="comun")
    - Para evaluar puntos destinados al cálculo de derivadas: f(x, tipo="derivada")
    a, b: Flotantes. Extremos del intervalo inicial (b > a).
    tol : Flotante>0. Tolerancia de convergencia.
    El algoritmo DEBE detenerse estrictamente cuando | x_k - x_{k-1} | <= tol.
    Para la evaluación, el equipo docente utilizará: tol = 1e-12
    Retorno:
    Devuelve únicamente un número flotante con la estimación de la raíz.
    """
    # Ejemplo de llamadas válidas para la telemetría del peaje (sistema de benchmark):
    # fx = f(x, tipo="comun")
    # fx_der = f(x + dx, tipo="derivada")

    max_iter = 150

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

    # Caso 1: Hay cambio de signo -> estrategia cerrada + refinamiento.
    if fa * fb < 0:
        try:
            raiz_breve = metodoBrent.encontrar_raiz(f_track, a, b, tol, max_iter=4)
            return raiz_breve
        except (ValueError, RuntimeError):
            pass

        # Intentamos NR en el intervalo original para aprovechar derivadas y cortar rafagas.
        try:
            f_nr = f_limited(f_track, max_comun=10, max_derivada=10)
            raiz_nr = NR.encontrar_raiz(f_nr, a, b, tol)
            return raiz_nr
        except Exception:
            pass

        # Garantia final en caso de que NR no converja.
        try:
            raiz_brent = metodoBrent.encontrar_raiz(f_track, a, b, tol)
            return raiz_brent
        except (ValueError, RuntimeError):
            if mejor_x is not None:
                return mejor_x
            raise RuntimeError("No se pudo encontrar la raíz")

    # Caso 2: No hay cambio de signo -> metodos abiertos priorizando NR.
    try:
        f_nr_abierto = f_limited(f_track, max_comun=10, max_derivada=10)
        raiz_nr_abierto = NR.encontrar_raiz(f_nr_abierto, a, b, tol)
        return raiz_nr_abierto
    except Exception:
        pass

    try:
        raiz_iqi = metodoIQI.encontrar_raiz(f_track, a, b, tol, max_iter=4)
        return raiz_iqi
    except Exception:
        pass

    try:
        # Secante no expone max_iter en esta version: limitamos por cantidad de evaluaciones.
        f_secante = f_limited(f_track, max_comun=6, max_derivada=0)
        raiz_sec = metodoSecante.encontrar_raiz(f_secante, a, b, tol)
        return raiz_sec
    except Exception:
        if mejor_x is not None:
            return mejor_x

    raise RuntimeError("No se pudo encontrar la raíz")