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

    if tol <= 0:
        raise ValueError("tol debe ser positiva")
    if b <= a:
        raise ValueError("Se espera que b > a")

    evaluaciones_comunes = 0
    newton_usado = False
    mejor_x = None
    mejor_abs_f = float("inf")

    def f_comun(x):
        nonlocal evaluaciones_comunes, mejor_x, mejor_abs_f
        evaluaciones_comunes += 1
        valor = f(x, tipo="comun")
        abs_valor = abs(valor)
        if abs_valor < mejor_abs_f:
            mejor_abs_f = abs_valor
            mejor_x = x
        return valor

    def f_derivada(x):
        return f(x, tipo="derivada")

    def secante(x0, f0, x1, f1):
        denominador = f1 - f0
        if denominador == 0.0:
            return None
        candidato = x1 - f1 * (x1 - x0) / denominador
        if not math.isfinite(candidato):
            return None
        return candidato

    def nuevo_x_por_newton(x_ref, fx_ref):
        nonlocal newton_usado
        if newton_usado or evaluaciones_comunes >= 4:
            return None

        f_der = f_derivada(x_ref)
        if f_der == 0.0 or not math.isfinite(f_der):
            return None

        candidato = x_ref - fx_ref / f_der
        if not math.isfinite(candidato):
            return None

        newton_usado = True
        return candidato

    fa = f_comun(a)
    fb = f_comun(b)

    if fa == 0.0:
        return float(a)
    if fb == 0.0:
        return float(b)

    if fa * fb < 0:
        x_anterior = None
        for _ in range(max_iter):
            punto_medio = (a + b) / 2.0
            fm = f_comun(punto_medio)
            if fm == 0.0:
                return float(punto_medio)

            discriminante = fm * fm - fa * fb
            if discriminante <= 0.0:
                raiz_estimada = punto_medio
            else:
                signo = 1.0 if (fa - fb) >= 0.0 else -1.0
                raiz_estimada = punto_medio + (punto_medio - a) * signo * fm / math.sqrt(discriminante)

            if not math.isfinite(raiz_estimada):
                raiz_estimada = punto_medio

            if x_anterior is not None and abs(raiz_estimada - x_anterior) <= tol:
                return float(raiz_estimada)

            fx = f_comun(raiz_estimada)
            if fx == 0.0:
                return float(raiz_estimada)

            if not newton_usado and evaluaciones_comunes >= 4:
                candidato_newton = nuevo_x_por_newton(raiz_estimada, fx)
                if candidato_newton is not None and a <= candidato_newton <= b:
                    raiz_estimada = candidato_newton
                    fx = f_comun(raiz_estimada)
                    if fx == 0.0:
                        return float(raiz_estimada)

            if fm * fx < 0:
                a, fa = punto_medio, fm
                b, fb = raiz_estimada, fx
            elif fa * fx < 0:
                b, fb = raiz_estimada, fx
            else:
                a, fa = raiz_estimada, fx

            x_anterior = raiz_estimada

        return float(mejor_x if mejor_x is not None else raiz_estimada)

    x0, f0 = a, fa
    x1, f1 = b, fb
    x_anterior = None

    for _ in range(max_iter):
        raiz_estimada = secante(x0, f0, x1, f1)
        if raiz_estimada is None or raiz_estimada < a or raiz_estimada > b:
            raiz_estimada = (a + b) / 2.0

        if x_anterior is not None and abs(raiz_estimada - x_anterior) <= tol:
            return float(raiz_estimada)

        fx = f_comun(raiz_estimada)
        if fx == 0.0:
            return float(raiz_estimada)

        if not newton_usado and evaluaciones_comunes >= 4:
            candidato_newton = nuevo_x_por_newton(raiz_estimada, fx)
            if candidato_newton is not None and a <= candidato_newton <= b:
                raiz_estimada = candidato_newton
                fx = f_comun(raiz_estimada)
                if fx == 0.0:
                    return float(raiz_estimada)

        x0, f0 = x1, f1
        x1, f1 = raiz_estimada, fx
        x_anterior = raiz_estimada

    return float(mejor_x if mejor_x is not None else raiz_estimada)