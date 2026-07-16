def encontrar_raiz(f, a, b, tol, use_illinois=True):
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
        return a
    if fb == 0.0:
        return b

    if fa * fb > 0:
        raise ValueError("El intervalo no contiene un cambio de signo")

    x_anterior = None
    lado_anterior = None

    for _ in range(max_iter):
        denominador = fb - fa
        if denominador == 0.0:
            raise RuntimeError("No se alcanzó la tolerancia en el máximo de iteraciones")

        c = b - fb * (b - a) / denominador
        fc = f(c, tipo="comun")

        if fc == 0.0:
            return c

        if x_anterior is not None and abs(c - x_anterior) <= tol:
            return c

        if fa * fc < 0:
            b = c
            fb = fc
            if use_illinois and lado_anterior == "b":
                fa *= 0.5
            lado_anterior = "b"
        else:
            a = c
            fa = fc
            if use_illinois and lado_anterior == "a":
                fb *= 0.5
            lado_anterior = "a"

        x_anterior = c

    raise RuntimeError("No se alcanzó la tolerancia en el máximo de iteraciones")