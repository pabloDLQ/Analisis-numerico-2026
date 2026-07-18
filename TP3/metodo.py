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


    evaluaciones_comunes = 0
    newton_usado = False
    mejor_x = None
    mejor_abs_f = float("inf")

    evaluaciones_comunes += 1
    fa = f(a, tipo="comun")
    abs_fa = abs(fa)
    if abs_fa < mejor_abs_f:
        mejor_abs_f = abs_fa
        mejor_x = a

    evaluaciones_comunes += 1
    fb = f(b, tipo="comun")
    abs_fb = abs(fb)
    if abs_fb < mejor_abs_f:
        mejor_abs_f = abs_fb
        mejor_x = b

    if fa == 0.0:
        return float(a)
    if fb == 0.0:
        return float(b)

    if fa * fb < 0:
        x_anterior = None
        for _ in range(max_iter):
            punto_medio = (a + b) / 2.0
            evaluaciones_comunes += 1
            fm = f(punto_medio, tipo="comun")
            abs_fm = abs(fm)
            if abs_fm < mejor_abs_f:
                mejor_abs_f = abs_fm
                mejor_x = punto_medio
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

            evaluaciones_comunes += 1
            fx = f(raiz_estimada, tipo="comun")
            abs_fx = abs(fx)
            if abs_fx < mejor_abs_f:
                mejor_abs_f = abs_fx
                mejor_x = raiz_estimada
            if fx == 0.0:
                return float(raiz_estimada)

            if not newton_usado and evaluaciones_comunes >= 4:
                f_der = f(raiz_estimada, tipo="derivada")
                if f_der != 0.0 and math.isfinite(f_der):
                    candidato_newton = raiz_estimada - fx / f_der
                    if math.isfinite(candidato_newton):
                        newton_usado = True
                    else:
                        candidato_newton = None
                else:
                    candidato_newton = None
                if candidato_newton is not None and a <= candidato_newton <= b:
                    raiz_estimada = candidato_newton
                    evaluaciones_comunes += 1
                    fx = f(raiz_estimada, tipo="comun")
                    abs_fx = abs(fx)
                    if abs_fx < mejor_abs_f:
                        mejor_abs_f = abs_fx
                        mejor_x = raiz_estimada
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
        denominador = f1 - f0
        if denominador == 0.0:
            raiz_estimada = None
        else:
            raiz_estimada = x1 - f1 * (x1 - x0) / denominador
            if not math.isfinite(raiz_estimada):
                raiz_estimada = None
        if raiz_estimada is None or raiz_estimada < a or raiz_estimada > b:
            raiz_estimada = (a + b) / 2.0

        if x_anterior is not None and abs(raiz_estimada - x_anterior) <= tol:
            return float(raiz_estimada)

        evaluaciones_comunes += 1
        fx = f(raiz_estimada, tipo="comun")
        abs_fx = abs(fx)
        if abs_fx < mejor_abs_f:
            mejor_abs_f = abs_fx
            mejor_x = raiz_estimada
        if fx == 0.0:
            return float(raiz_estimada)

        if not newton_usado and evaluaciones_comunes >= 4:
            f_der = f(raiz_estimada, tipo="derivada")
            if f_der != 0.0 and math.isfinite(f_der):
                candidato_newton = raiz_estimada - fx / f_der
                if math.isfinite(candidato_newton):
                    newton_usado = True
                else:
                    candidato_newton = None
            else:
                candidato_newton = None
            if candidato_newton is not None and a <= candidato_newton <= b:
                raiz_estimada = candidato_newton
                evaluaciones_comunes += 1
                fx = f(raiz_estimada, tipo="comun")
                abs_fx = abs(fx)
                if abs_fx < mejor_abs_f:
                    mejor_abs_f = abs_fx
                    mejor_x = raiz_estimada
                if fx == 0.0:
                    return float(raiz_estimada)

        x0, f0 = x1, f1
        x1, f1 = raiz_estimada, fx
        x_anterior = raiz_estimada

    return float(mejor_x if mejor_x is not None else raiz_estimada)