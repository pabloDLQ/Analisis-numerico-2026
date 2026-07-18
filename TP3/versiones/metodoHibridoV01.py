import math


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


    fa = f(a, tipo="comun")
    fb = f(b, tipo="comun")

    if fa == 0.0:
        return float(a)
    if fb == 0.0:
        return float(b)

    raiz_estimada = (a + b) / 2.0
    x_anterior = None

    # Fase 1: bisección si hay cambio de signo.
    if fa * fb < 0:
        for _ in range(max_iter):
            raiz_estimada = (a + b) / 2.0
            fx = f(raiz_estimada, tipo="comun")

            if fx == 0.0:
                return float(raiz_estimada)
            if x_anterior is not None and abs(raiz_estimada - x_anterior) <= tol:
                return float(raiz_estimada)

            if fa * fx < 0:
                b, fb = raiz_estimada, fx
            else:
                a, fa = raiz_estimada, fx

            x_anterior = raiz_estimada

        return float(raiz_estimada)

    # Fase 2: no hay cambio de signo, usar Newton amortiguado con derivada.
    if abs(fa) <= abs(fb):
        raiz_estimada = float(a)
    else:
        raiz_estimada = float(b)

    for _ in range(max_iter):
        fx = f(raiz_estimada, tipo="comun")
        if fx == 0.0:
            return float(raiz_estimada)

        f_der = f(raiz_estimada, tipo="derivada")
        if f_der == 0.0 or not math.isfinite(f_der):
            break

        candidato = raiz_estimada - fx / f_der

        if not math.isfinite(candidato):
            break

        if x_anterior is not None and abs(candidato - raiz_estimada) <= tol:
            return float(candidato)

        if candidato < a or candidato > b:
            break

        x_anterior = raiz_estimada
        raiz_estimada = candidato

    # Fase 3: secante acotada dentro del intervalo, como respaldo final.
    x0 = float(a)
    x1 = float(b)
    f0 = fa
    f1 = fb

    for _ in range(max_iter):
        denominador = f1 - f0
        if denominador == 0.0:
            break

        candidato = x1 - f1 * (x1 - x0) / denominador
        if not math.isfinite(candidato):
            break

        if x_anterior is not None and abs(candidato - x1) <= tol:
            return float(candidato)

        if candidato < a or candidato > b:
            candidato = (a + b) / 2.0

        fc = f(candidato, tipo="comun")
        if fc == 0.0:
            return float(candidato)

        x0, f0 = x1, f1
        x1, f1 = candidato, fc

        if abs(x1 - x0) <= tol:
            return float(x1)

        x_anterior = x0

    return float(raiz_estimada)