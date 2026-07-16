def encontrar_raiz(f, a, b, tol, max_iter=150):
    """
    Argumentos de entrada:
    f : Función a evaluar. Se debe invocar especificando el "tipo" de llamada:
    - Para evaluar la función de forma estándar: f(x, tipo="comun")
    - Para evaluar puntos destinados al cálculo de derivadas: f(x, tipo="derivada")
    a, b: Flotantes. Extremos del intervalo inicial (b > a).
    tol : Flotante>0. Tolerancia de convergencia.
    El algoritmo DEBE detenerse estrictamente cuando | x_k - x_{k-1} | <= tol.
    Para la evaluación, el equipo docente utilizará: tol = 1e-12
    max_iter : Entero opcional. Cantidad máxima de iteraciones permitidas.
    Retorno:
    Devuelve únicamente un número flotante con la estimación de la raíz.
    """
    # Ejemplo de llamadas válidas para la telemetría del peaje (sistema de benchmark):
    # fx = f(x, tipo="comun")
    # fx_der = f(x + dx, tipo="derivada")

    x0 = a
    x1 = (a + b) / 2.0
    x2 = b

    f0 = f(x0, tipo="comun")
    f1 = f(x1, tipo="comun")
    f2 = f(x2, tipo="comun")

    if f0 == 0.0:
        return x0
    if f1 == 0.0:
        return x1
    if f2 == 0.0:
        return x2

    puntos = [(x0, f0), (x1, f1), (x2, f2)]

    for _ in range(max_iter):
        puntos.sort(key=lambda item: abs(item[1]))
        (x0, f0), (x1, f1), (x2, f2) = puntos

        if abs(x0 - x1) <= tol:
            return x0

        if f0 != f1 and f0 != f2 and f1 != f2:
            denom0 = (f0 - f1) * (f0 - f2)
            denom1 = (f1 - f0) * (f1 - f2)
            denom2 = (f2 - f0) * (f2 - f1)

            if denom0 != 0.0 and denom1 != 0.0 and denom2 != 0.0:
                x3 = (
                    x0 * (f1 * f2) / denom0
                    + x1 * (f0 * f2) / denom1
                    + x2 * (f0 * f1) / denom2
                )
            else:
                x3 = x2 - f2 * (x2 - x1) / (f2 - f1) if f2 != f1 else x0
        else:
            x3 = x2 - f2 * (x2 - x1) / (f2 - f1) if f2 != f1 else x0

        if x3 == x0 or x3 == x1 or x3 == x2:
            x3 = x2 - f2 * (x2 - x1) / (f2 - f1) if f2 != f1 else (x0 + x1 + x2) / 3.0

        f3 = f(x3, tipo="comun")

        if f3 == 0.0:
            return x3
        if abs(x3 - x0) <= tol:
            return x3

        puntos[2] = (x3, f3)

    raise RuntimeError("No se alcanzó la tolerancia en el máximo de iteraciones")